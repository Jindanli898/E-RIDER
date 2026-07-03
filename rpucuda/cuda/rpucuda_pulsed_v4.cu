/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "forward_backward_pass.h"
#include "rpucuda_pulsed.h"
#include "rpucuda_transfer_device.h"

#include <iostream>
#include <cstdlib>   // getenv, atoi, atof
#include <cstdio> 
#include <cstdint>
   // fprintf, fflush
#include <chrono>
#include <memory>
#include <typeinfo>
#include "rpucuda_dynamic_transfer_device.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cstring>
#include <sstream>

#ifdef RPU_USE_MKL
  #include <mkl_lapacke.h>
#else
  #include <lapacke.h>
#endif
#define CHECK_RPU_DEVICE_INIT                                                                      \
  if (rpucuda_device_ == nullptr || rpu_device_ == nullptr) {                                      \
    RPU_FATAL("First populate rpu device (call Populate_Parameter())!");                           \
  }
template <typename T>
__global__ void kernelOuterFroSq(
    const T *x,   // [B, Dx]
    const T *d,   // [B, Dd]
    int B,
    int Dx,
    int Dd,
    float *out) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = Dd * Dx;

  double local = 0.0;

  for (int flat = idx; flat < total; flat += blockDim.x * gridDim.x) {
    int i = flat / Dx;   // row in d / output dim
    int j = flat % Dx;   // col in x / input dim

    double gij = 0.0;
    for (int b = 0; b < B; ++b) {
      gij += (double)d[b * Dd + i] * (double)x[b * Dx + j];
    }
    gij /= (double)B;
    local += gij * gij;
  }

  atomicAdd(out, (float)local);
}

template <typename T>
__global__ void kernelScaleArray(T *a, int n, T scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    a[i] = (T)((double)a[i] * (double)scale);
  }
}

template <typename T>
__global__ void kernelKFACResidualToTransfer(
    T *transfer_tmp,   // [n_rows, x_size], row-major
    T *residual,       // [d_size, x_size], col-major as weight layout
    int x_size,
    int d_size,
    int row_start,
    int n_rows,
    T granularity) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_rows * x_size;

  for (int k = idx; k < total; k += blockDim.x * gridDim.x) {
    int local_row = k / x_size;
    int col = k % x_size;
    int global_row = row_start + local_row;

    // residual is stored in col-major [d, x]
    int ridx = global_row + col * d_size;

        T value = residual[ridx];
    T q = -trunc(value / granularity);

    // transfer input is row-major [batch=n_rows, x_size]
    transfer_tmp[k] = q;

    // actual device-0 change follows approximately: actual ~= -granularity * q
    // so subtract the realized signed commit from the residual in the same convention.
    residual[ridx] = value + granularity * q;

  }
}


template <typename T>
__global__ void kernelUpdateCovEMA(
    const T *in,   // [m_batch, full_dim]
    T *S,          // [bs, bs]
    int full_dim,
    int start,
    int bs,
    int m_batch,
    T beta,
    T sample_scale,   // NEW: scale input only for statistics
    int initialize) { // NEW: 1 => S = C, 0 => EMA

  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (r >= bs || c >= bs) {
    return;
  }

  double acc = 0.0;
  for (int mb = 0; mb < m_batch; ++mb) {
    const T *xb = in + mb * full_dim + start;
    double vr = (double)sample_scale * (double)xb[r];
    double vc = (double)sample_scale * (double)xb[c];
    acc += vr * vc;
  }

  T C = (T)(acc / (double)m_batch);
  int idx = r * bs + c;

  if (initialize) {
    S[idx] = C;
  } else {
    S[idx] = ((T)1 - beta) * S[idx] + beta * C;
  }
}

template <typename T>
__global__ void kernelBlockMatVec(
    const T *in,
    T *out,
    const T *F,
    int full_dim,
    int start,
    int bs,
    int m_batch) {

  int mb = blockIdx.y;
  int r  = blockIdx.x * blockDim.x + threadIdx.x;

  if (mb >= m_batch || r >= bs) {
    return;
  }

  const T *x = in  + mb * full_dim + start;
  T *y       = out + mb * full_dim + start;

  double acc = 0.0;
  for (int c = 0; c < bs; ++c) {
    acc += (double)F[r * bs + c] * (double)x[c];
  }
  y[r] = (T)acc;
}
struct ScopedWallTimer {
  const char *name;
  std::chrono::high_resolution_clock::time_point t0;

  explicit ScopedWallTimer(const char *n)
      : name(n), t0(std::chrono::high_resolution_clock::now()) {}

  ~ScopedWallTimer() {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[TIME][WALL] %s: %.3f ms\n", name, ms);
    fflush(stderr);
  }
};
struct ScopedCudaTimer {
  const char *name;
  cudaStream_t stream;
  cudaEvent_t start{}, stop{};

  ScopedCudaTimer(const char *n, cudaStream_t s) : name(n), stream(s) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
  }

  ~ScopedCudaTimer() {
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    fprintf(stderr, "[TIME][CUDA] %s: %.3f ms\n", name, ms);
    fflush(stderr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};
namespace RPU {

/********************************************************************************
 * RPUCudaPulsed<T>
 *********************************************************************************/

template <typename T> void RPUCudaPulsed<T>::initialize() {

  int d_size = this->getDSize();
  int x_size = this->getXSize();

  CudaContextPtr c = this->context_;
  size_ = d_size * x_size;

  // forward arrays
  f_iom_ = RPU::make_unique<InputOutputManager<T>>(c, x_size, d_size);
  dev_f_x_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, x_size);
  dev_f_d_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, d_size);

  // backward arrays
  b_iom_ = RPU::make_unique<InputOutputManager<T>>(c, d_size, x_size);
  dev_b_x_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, x_size);
  dev_b_d_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, d_size);

  // fb pass
  fb_pass_ = RPU::make_unique<ForwardBackwardPassIOManagedCuda<T>>(c, x_size, d_size);

  // update arrays
    // update arrays
  up_pwu_ = RPU::make_unique<PulsedWeightUpdater<T>>(c, x_size, d_size);

  // separate updater for KFAC residual-transfer submit.
  // Keep this isolated from up_pwu_ so kernel/cache state such as BL/NK32
  // does not get mixed with the normal rank-update path.
  up_pwu_kfac_transfer_ = RPU::make_unique<PulsedWeightUpdater<T>>(c, x_size, d_size);

  dev_up_x_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, x_size);
  dev_up_d_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, d_size);

  this->context_->synchronize();

  rpu_device_ = nullptr;
  rpucuda_device_ = nullptr;
  kfac_buffer_mbatch_ = 0;
  DEBUG_OUT("RPUCudaPulsed constructed");
}

template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(CudaContextPtr c, int x_size, int d_size)
    : RPUCudaSimple<T>(c, x_size, d_size) {}

template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(cudaStream_t s, int x_size, int d_size)
    : RPUCudaSimple<T>(s, x_size, d_size) {}

template <typename T> void RPUCudaPulsed<T>::initFrom(RPUPulsed<T> &rpu) {
  // this is private and only for the construction from CPU

  initialize();

  // forward / backward
  fb_pass_->populateFrom(rpu.getFBParameter());

  // update
  rpu_device_ = rpu.cloneDevice();
  par_ = rpu.getMetaPar();
  if (rpu_device_) {
    rpucuda_device_ = AbstractRPUDeviceCuda<T>::createFromUnique(this->context_, *rpu_device_);
  } else {
    RPU_FATAL("Expect rpu_device to be populated!");
  }
  this->context_->synchronize();

  // NOTE: weight is already copied in RPUSimple constructor
}

template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(CudaContextPtr c, RPUPulsed<T> &o)
    : RPUCudaSimple<T>(c, static_cast<RPUSimple<T> &>(o)) {
  initFrom(o);
}

template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(cudaStream_t s, RPUPulsed<T> &o)
    : RPUCudaSimple<T>(s, static_cast<RPUSimple<T> &>(o)) {
  initFrom(o);
}

template <typename T> RPUCudaPulsed<T>::~RPUCudaPulsed() { DEBUG_OUT("RPUCudaPulsed destroyed."); }

// copy constructor
template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(const RPUCudaPulsed<T> &other) : RPUCudaSimple<T>(other) {
  // NOTE: we do not copy all the class members helper such as blm,
  // pwu etc. They get reconstructed. We only copy construct the
  // RPU_PULSED_, since all relevant parameters are set there, and
  // copy it to GPU
  if (up_pwu_ != nullptr) { // check whether it is initialized
    initialize();           // private
  }
  par_ = other.par_;

  if (other.rpu_device_) {
    rpu_device_ = other.rpu_device_->cloneUnique();
    rpucuda_device_ = AbstractRPUDeviceCuda<T>::createFromUnique(this->context_, *rpu_device_);
  } else {
    rpucuda_device_ = nullptr;
    rpu_device_ = nullptr;
  }
  par_ = other.par_;
  fb_pass_ = RPU::make_unique<ForwardBackwardPassIOManagedCuda<T>>(*other.fb_pass_);

  this->context_->synchronize();
  DEBUG_CALL(this->disp(););
  DEBUG_OUT("RPUCudaPulsed copy constructed.");
}

// copy assignment
template <typename T> RPUCudaPulsed<T> &RPUCudaPulsed<T>::operator=(const RPUCudaPulsed<T> &other) {

  RPUCudaPulsed<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
}

// move constructor
template <typename T> RPUCudaPulsed<T>::RPUCudaPulsed(RPUCudaPulsed<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T> RPUCudaPulsed<T> &RPUCudaPulsed<T>::operator=(RPUCudaPulsed<T> &&other) {

  RPUCudaSimple<T>::operator=(std::move(other));

  rpu_device_ = std::move(other.rpu_device_);
  rpucuda_device_ = std::move(other.rpucuda_device_);

  par_ = other.par_;
    f_iom_ = std::move(other.f_iom_);
  b_iom_ = std::move(other.b_iom_);
  up_pwu_ = std::move(other.up_pwu_);
  up_pwu_kfac_transfer_ = std::move(other.up_pwu_kfac_transfer_);
  fb_pass_ = std::move(other.fb_pass_);


  dev_up_x_vector_inc1_ = std::move(other.dev_up_x_vector_inc1_);
  dev_up_d_vector_inc1_ = std::move(other.dev_up_d_vector_inc1_);
  dev_f_x_vector_inc1_ = std::move(other.dev_f_x_vector_inc1_);
  dev_f_d_vector_inc1_ = std::move(other.dev_f_d_vector_inc1_);

  dev_b_x_vector_inc1_ = std::move(other.dev_b_x_vector_inc1_);
  dev_b_d_vector_inc1_ = std::move(other.dev_b_d_vector_inc1_);

  size_ = other.size_;

  this->context_->synchronize();

  return *this;
}

template <typename T>
void RPUCudaPulsed<T>::populateParameter(
    PulsedMetaParameter<T> *p, PulsedRPUDeviceMetaParameter<T> *dp) {
  RPUCudaSimple<T>::populateParameter(dp);

  // TODO: better to init the internal parameter only and pass this as const?
  p->initialize(this->x_size_, this->d_size_);

  if (up_pwu_ == nullptr) {
    initialize();
  }

  // use CPU populate for FB pass
  auto fb_pass_host = ForwardBackwardPassIOManaged<T>(this->x_size_, this->d_size_, this->rng_);
  
  p->f_io.mv_type = AnalogMVType::OnePass;
  p->b_io.mv_type = AnalogMVType::OnePass;
  // test for onepass MV

  fb_pass_host.populateFBParameter(p->f_io, p->b_io);
  fb_pass_->populateFrom(fb_pass_host.getFBParameter());

  if (p->up.pulse_type == PulseType::None) {

    if (dynamic_cast<SimpleRPUDeviceMetaParameter<T> *>(dp) == nullptr) {
      RPU_FATAL("For PulseType::None device needs to be castable to Simple.");
    }

    SimpleRPUDeviceMetaParameter<T> dp_simple(*static_cast<SimpleRPUDeviceMetaParameter<T> *>(dp));
    rpu_device_ = dp_simple.createDeviceUnique(this->x_size_, this->d_size_, nullptr);
  } else {
    // create and populate correct device
    RealWorldRNG<T> rng(dp->construction_seed);
    rpu_device_ = dp->createDeviceUnique(this->x_size_, this->d_size_, &rng);
  }
  rpucuda_device_ = AbstractRPUDeviceCuda<T>::createFromUnique(this->context_, *rpu_device_);


  // --- Enable pref only for network fb_pass_ (transfer path remains untouched) ---
  // {
  //   // TODO: decide where gamma comes from. For core bring-up, hardcode.
  //   const T gamma = (T)1.0;

  //   if (auto *dt = dynamic_cast<DynamicTransferRPUDeviceCuda<T> *>(rpucuda_device_.get())) {
  //     const T *q_ptr = dt->getPastMeanDataConst();
  //     fb_pass_->setPref(q_ptr, gamma, /*apply_fwd=*/true, /*apply_bwd=*/true);
  //   } else {
  //     fb_pass_->clearPref();
  //   }
  // }
  fprintf(stderr, "[CHOPPER][UNIQUE] HIT populateParameter\n");
fflush(stderr);
  #ifdef RPU_DEBUG_CHOPPER_CHECK

std::cout << "[CHOPPER] rpucuda_device_ RTTI = " << typeid(*rpucuda_device_).name() << std::endl;
#ifdef CHOPPER_PREF_ON
std::cout << "[CHOPPER] CHOPPER_PREF_ON is defined\n";
#else
std::cout << "[CHOPPER] CHOPPER_PREF_ON is NOT defined\n";
#endif
#endif

  #ifdef CHOPPER_PREF_ON
  const T gamma = (T)0.1;
  // if (auto *dt = dynamic_cast<DynamicTransferRPUDeviceCuda<T> *>(rpucuda_device_.get())) {
  if (auto *dt = dynamic_cast<RPU::DynamicTransferRPUDeviceCuda<T> *>(rpucuda_device_.get())) { 
  fb_pass_->setPref(dt->getPastMeanDataConst(), gamma, true, true);
  } else {
    fb_pass_->clearPref();
  }
#else
  fb_pass_->clearPref();
#endif

  this->setWeights(this->copyWeightsToHost()[0]); // set weights needs all populated

  par_ = *p; // only copy, read access with getMetaPar()
}

/*********************************************************************************/

template <typename T> void RPUCudaPulsed<T>::setLearningRate(T lr) {

  if (lr != this->getLearningRate()) {

    RPUCudaSimple<T>::setLearningRate(lr);

    if (rpucuda_device_ != nullptr && rpucuda_device_->isPulsedDevice()) {
      // some output
      int BL = 0;
      T A = 0;
      T B = 0;
      getMetaPar().up.calculateBlAB(
          BL, A, B, lr,
          static_cast<PulsedRPUDeviceCudaBase<T> &>(*rpucuda_device_).getWeightGranularity());
      DEBUG_OUT("\t BL = " << BL << ", A = " << A << ", B = " << B);
    }
  }
}

/*********************************************************************************/

template <typename T> void RPUCudaPulsed<T>::printToStream(std::stringstream &ss) const {

  CHECK_RPU_DEVICE_INIT;

  std::string name;
  name = rpucuda_device_->getPar().getName();

  ss << "RPUCudaPulsed<" << this->getDataTypeName() << ">[" << name << "](" << this->d_size_ << ","
     << this->x_size_ << ")" << std::endl;
};

/*********************************************************************************/
/** These functions use the rpu device and are basically a
    reimplimentation of RPU_PULSED for CUDA. Note that we cannot
    directly inherit from RPU_PULSED (triangle)!  **/

template <typename T> void RPUCudaPulsed<T>::decayWeights(T alpha, bool bias_no_decay) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->decayWeights(this->dev_weights_->getData(), alpha, bias_no_decay);
}

template <typename T> void RPUCudaPulsed<T>::decayWeights(bool bias_no_decay) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->decayWeights(this->dev_weights_->getData(), bias_no_decay);
}

template <typename T> void RPUCudaPulsed<T>::driftWeights(T time_since_last_call) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->driftWeights(this->dev_weights_->getData(), time_since_last_call);
}

template <typename T> void RPUCudaPulsed<T>::diffuseWeights() {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->diffuseWeights(this->dev_weights_->getData());
}

template <typename T> void RPUCudaPulsed<T>::clipWeights(T clip) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->clipWeights(this->dev_weights_->getData(), clip);
}

template <typename T> void RPUCudaPulsed<T>::clipWeights(const WeightClipParameter &wclpar) {

  if (wclpar.type == WeightClipType::FixedValue) {
    clipWeights(wclpar.fixed_value); // handle outside  to support devices
  } else if (rpu_device_->implements() == DeviceUpdateType::FloatingPoint) {
    RPUCudaSimple<T>::clipWeights(wclpar);
  } else {
    RPU_FATAL("Sophisticated clipping is NOT implemented for most training devices");
  }
}

template <typename T>
void RPUCudaPulsed<T>::remapWeights(const WeightRemapParameter &wrmpar, T *scales, T *biases) {
  // Same as Simple version, however, we can now suppport the exceeded channel BM

  ENFORCE_NO_DELAYED_UPDATE; // will get confused with the buffer

  if (this->wremapper_cuda_ == nullptr) {
    if (!up_pwu_->checkForFPUpdate(&*rpucuda_device_, getMetaPar().up)) {
      getMetaPar().print();
      RPU_FATAL("Remapping is NOT implemented for most devices");
    }
    this->wremapper_cuda_ =
        RPU::make_unique<WeightRemapperCuda<T>>(this->context_, this->x_size_, this->d_size_);
  }

  // remap weights
  this->wremapper_cuda_->apply(
      this->dev_weights_->getData(), this->getAlphaLearningRate(), wrmpar, scales, biases);
}

template <typename T>
bool RPUCudaPulsed<T>::swaWeights(
    const WeightRemapParameter &wrmpar, T *swa_weights, uint64_t iter, T *scales, T *biases) {

  CHECK_RPU_DEVICE_INIT;
  ENFORCE_NO_DELAYED_UPDATE;

  if (wrmpar.type != WeightRemapType::None &&
      !up_pwu_->checkForFPUpdate(&*rpucuda_device_, getMetaPar().up)) {
    getMetaPar().print();
    RPU_FATAL("SWA is NOT implemented for most devices");
  }

  bool modfied = RPUCudaSimple<T>::swaWeights(wrmpar, swa_weights, iter, scales, biases);

  if (modfied) {
    this->copyWeightsToHost();
    this->setWeights(this->getWeightsPtr()[0]);
  }
  return modfied;
}

template <typename T> void RPUCudaPulsed<T>::resetCols(int start_col, int n_cols, T reset_prob) {

  if (reset_prob) {
    CHECK_RPU_DEVICE_INIT;
    rpucuda_device_->resetCols(this->dev_weights_->getData(), start_col, n_cols, reset_prob);
  }
}

template <typename T> void RPUCudaPulsed<T>::printRPUParameter(int x_count, int d_count) const {
  // prints parameters from rpu_stoc without syncing ! However,
  // device should mirror rpu_pulsed_ any time anyway

  CHECK_RPU_DEVICE_INIT;
  rpu_device_->printDP(x_count, d_count);
}

template <typename T> void RPUCudaPulsed<T>::getWeightsReal(T *weightsptr) {

  CHECK_RPU_DEVICE_INIT;

  int x_sz = this->getXSize();
  int d_sz = this->getDSize();

  T **eye = Array_2D_Get_Eye<T>(x_sz);
  auto eye_d = CudaArray<T>(this->context_, x_sz * x_sz, eye[0]);
  auto w_buffer = CudaArray<T>(this->context_, d_sz * x_sz);
  this->context_->synchronize();

  bool is_test = true; // should not change anything

  T alpha = this->getFwdAlpha();
  this->setFwdAlpha(1.0, false);
  this->forwardMatrix(eye_d.getDataConst(), w_buffer.getData(), x_sz, false, true, is_test);
  this->setFwdAlpha(alpha, false);

  w_buffer.copyTo(weightsptr);
  this->context_->synchronize();

  Array_2D_Free<T>(eye);
}

template <typename T> void RPUCudaPulsed<T>::setWeightsReal(const T *weightsptr, int n_loops) {

  CHECK_RPU_DEVICE_INIT;

  int x_sz = this->getXSize();
  int d_sz = this->getDSize();

  /*==== slight hack to get the range right */
  T weight_granularity = 0.001;
  const auto *dpar =
      dynamic_cast<const PulsedRPUDeviceMetaParameter<T> *>(&rpucuda_device_->getPar());
  T w_min = -1;
  T w_max = 1;

  if (dpar != nullptr) {
    w_min = dpar->w_min;
    w_max = dpar->w_max;
    weight_granularity =
        static_cast<PulsedRPUDeviceCudaBase<T> &>(*rpucuda_device_).getWeightGranularity();
  }
  T A = 0;
  T B = 0;
  int BL = 0;
  getMetaPar().up.calculateBlAB(BL, A, B, this->getLearningRate(), weight_granularity);
  T mx_change = (T)BL * weight_granularity;
  T range = fabsf(w_max - w_min);
  int iter = ceilf((T)n_loops * range / mx_change);

  /*==== */

  DEBUG_OUT("RPUCudaPulsed: Set weights real [iter=" << iter << "]");

  T **eye = Array_2D_Get_Eye<T>(x_sz);
  auto eye_d = CudaArray<T>(this->context_, x_sz * x_sz, eye[0]);
  auto w_ref_trans = CudaArray<T>(this->context_, d_sz * x_sz, weightsptr);
  auto delta = CudaArray<T>(this->context_, d_sz * x_sz, weightsptr);

  this->context_->synchronize();
  bool is_test = false; // not used
  T fwd_alpha = this->getFwdAlpha();
  T bwd_alpha = this->getBwdAlpha();
  this->setFwdAlpha(1.0, false);
  this->setBwdAlpha(1.0, false);
  for (int k = 0; k < iter; ++k) {

    this->forwardMatrix(eye_d.getDataConst(), delta.getData(), x_sz, false, true, is_test);

    RPU::math::elemaddscale<T>(
        this->context_, delta.getData(), x_sz * d_sz, w_ref_trans.getDataConst(), (T)-1.0);

    this->updateMatrix(eye_d.getDataConst(), delta.getDataConst(), x_sz, false, true);
  }
  this->setFwdAlpha(fwd_alpha, false);
  this->setBwdAlpha(bwd_alpha, false);

  this->context_->synchronize();

  T avg_dev = 0.0;
  T *w_current = this->copyWeightsToHost()[0];
  for (int i = 0; i < x_sz * d_sz; ++i) {
    avg_dev += fabsf(weightsptr[i] - w_current[i]);
  }
  avg_dev /= x_sz * d_sz;
  DEBUG_OUT("Finished setting weights real [avg deviation=" << avg_dev << "]");

  Array_2D_Free<T>(eye);
}

template <typename T>
void RPUCudaPulsed<T>::getDeviceParameterNames(std::vector<std::string> &names) const {

  CHECK_RPU_DEVICE_INIT;
  rpu_device_->getDPNames(names);
}

template <typename T> void RPUCudaPulsed<T>::getDeviceParameter(std::vector<T *> &data_ptrs) {

  CHECK_RPU_DEVICE_INIT;
  rpu_device_->setHiddenWeights(rpucuda_device_->getHiddenWeights());
  this->copyWeightsToHost();
  rpu_device_->getDeviceParameter(this->getWeightsPtr(), data_ptrs);
};

template <typename T> void RPUCudaPulsed<T>::setDeviceParameter(const std::vector<T *> &data_ptrs) {
  // note that memory (x_sz*d_sz per ptr) assumed to be initialized from outside !!

  CHECK_RPU_DEVICE_INIT;

  // Note: for now setting the device just keeps the old meta parameter.
  // however weight_granularity is at least estimated.
  this->copyWeightsToHost();
  rpu_device_->setDeviceParameter(this->getWeightsPtr(), data_ptrs);

  rpucuda_device_->populateFrom(*rpu_device_);

  // set device weights which might have been updated because of the hidden parameters
  RPUCudaSimple<T>::setWeights(this->getWeightsPtr()[0]);
};

template <typename T> int RPUCudaPulsed<T>::getHiddenUpdateIdx() const {
  CHECK_RPU_DEVICE_INIT;
  return rpucuda_device_->getHiddenUpdateIdx();
};

template <typename T> void RPUCudaPulsed<T>::setHiddenUpdateIdx(int idx) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->setHiddenUpdateIdx(idx);
  rpu_device_->setHiddenUpdateIdx(idx);
};

/*********************************************************************************/
/* dump / load state */

template <typename T>
void RPUCudaPulsed<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  RPUCudaSimple<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  rpu_device_->dumpExtra(state, "rpu_device");
  rpucuda_device_->dumpExtra(state, "rpucuda_device");
    up_pwu_kfac_transfer_->dumpExtra(state, "up_pwu_kfac_transfer");

  f_iom_->dumpExtra(state, "f_iom");
  b_iom_->dumpExtra(state, "b_iom");

  up_pwu_->dumpExtra(state, "up_pwu");
  fb_pass_->dumpExtra(state, "fb_pass");

  // tmp vectors are ignored
  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void RPUCudaPulsed<T>::loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {
  RPUCudaSimple<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  rpu_device_->loadExtra(state, "rpu_device", strict);
  rpucuda_device_->loadExtra(state, "rpucuda_device", strict);
    up_pwu_kfac_transfer_->loadExtra(state, "up_pwu_kfac_transfer", strict);

  f_iom_->loadExtra(state, "f_iom", strict);
  b_iom_->loadExtra(state, "b_iom", strict);

  up_pwu_->loadExtra(state, "up_pwu", strict);
  fb_pass_->loadExtra(state, "fb_pass", strict);
}

/*********************************************************************************/
template <typename T> void RPUCudaPulsed<T>::setWeights(const T *host_source) {

  CHECK_RPU_DEVICE_INIT;
  RPUSimple<T>::setWeights(host_source); // sets host

  if (rpu_device_) {
    if (rpu_device_->onSetWeights(this->getWeightsPtr())) {
      // apply bounds etc to host
      rpucuda_device_->populateFrom(*rpu_device_); // device pars have changed (due to onSetWeights)
    }
  }
  RPUCudaSimple<T>::setWeights(this->getWeightsPtr()[0]); // set device weights
}

template <typename T> void RPUCudaPulsed<T>::applyWeightUpdate(T *dw_and_current_weight_out) {

  CHECK_RPU_DEVICE_INIT;

  if (rpu_device_) {
    rpucuda_device_->applyWeightUpdate(this->dev_weights_->getData(), dw_and_current_weight_out);
  } else {
    RPUCudaSimple<T>::applyWeightUpdate(dw_and_current_weight_out);
  }
}

/*********************************************************************************/
/*********************************************************************************/
/* FORWARD */

template <typename T>
void RPUCudaPulsed<T>::forwardMatrix(
    const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test) {
  this->forwardMatrixIterator(X_input, D_output, m_batch, x_trans, d_trans, is_test);
}

template <typename T>
void RPUCudaPulsed<T>::forwardIndexed(
    const T *X_input,
    T *D_output,
    int total_input_size,
    int m_batch,
    int dim3,
    bool trans,
    bool is_test) {

  const int *indices = this->getMatrixIndices();

  if (trans && (dim3 > 1)) {

    IndexReaderTransInputIterator<T> iter(
        X_input, indices, total_input_size / dim3, m_batch, this->x_size_ * m_batch,
        m_batch * dim3);

    PermuterTransOutputIterator<T> permute_iter(
        D_output, m_batch, this->d_size_ * m_batch, m_batch * dim3);

    this->forwardMatrixIterator(iter, permute_iter, m_batch * dim3, trans, trans, is_test);

  } else {
    IndexReaderInputIterator<T> iter(
        X_input, indices, total_input_size / dim3, this->x_size_ * m_batch);

    this->forwardMatrixIterator(iter, D_output, m_batch * dim3, trans, trans, is_test);
  }
}

template <typename T>
void RPUCudaPulsed<T>::forwardIndexedSlice(
    const T *X_input,
    T *D_output,
    int total_input_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices,
    bool is_test) {

  const int *indices = this->getMatrixIndices();
  int x_size = this->getXSize();
  int d_size = this->getDSize();

  if (trans && (dim3 > 1)) {

    IndexReaderSliceInputIterator<true, T> in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    SliceOutputIterator<true, T> out_iter(
        D_output, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    this->forwardMatrixIterator(in_iter, out_iter, m_batch_slice * dim3, trans, trans, is_test);

  } else {

    IndexReaderSliceInputIterator<false, T> in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    SliceOutputIterator<false, T> out_iter(
        D_output, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    this->forwardMatrixIterator(in_iter, out_iter, m_batch_slice * dim3, trans, trans, is_test);
  }
}

template <typename T>
template <typename InputIteratorT, typename OutputIteratorT>
void RPUCudaPulsed<T>::forwardMatrixIterator(
    InputIteratorT X_input,
    OutputIteratorT D_output,
    int m_batch,
    bool x_trans,
    bool d_trans,
    bool is_test) {

// #ifdef RPU_DEBUG_CHOPPER_CHECK
//   static int hit = 0;
//   if (hit++ < 5) {
//     fprintf(stderr,
//             "[CHOPPER][HIT] RPUCudaPulsed::forwardMatrixIterator this=%p fb_pass=%p\n",
//             (void *)this, (void *)fb_pass_.get());
//     fflush(stderr);
//   }
// #endif

  // -------- Pref runtime gate (no recompile needed) --------
  auto pref_on = []() -> bool {
    const char *e = std::getenv("AIHWKIT_PREF_ON");
    return (e != nullptr) && (std::atoi(e) != 0);
  };

  auto get_pref_gamma = []() -> T {
    const char *g = std::getenv("AIHWKIT_PREF_GAMMA");
    return (g != nullptr) ? (T)std::atof(g) : (T)0.1; // default 0.1
  };

  if (pref_on()) {

// #ifdef RPU_DEBUG_CHOPPER_CHECK
//     static int rtti_hit = 0;
//     if (rtti_hit++ < 5) {
//       fprintf(stderr, "[CHOPPER] rpucuda_device RTTI=%s\n", typeid(*rpucuda_device_).name());
//       fflush(stderr);
//     }
// #endif

    if (auto *dt = dynamic_cast<RPU::DynamicTransferRPUDeviceCuda<T> *>(rpucuda_device_.get())) {

      // const T *q_ptr = dt->getPastMeanDataConst();
      const T *cpq_ptr = dt->getCPQDataConst();
      const T pref_g = get_pref_gamma();

      // if (q_ptr != nullptr && pref_g != (T)0) {
      //   // Forward entry: enable only fwd pref (bwd set in backward entry).
      //   fb_pass_->setPref(q_ptr, pref_g, /*apply_fwd=*/true, /*apply_bwd=*/false);
      if (cpq_ptr != nullptr && pref_g != (T)0) {
        fb_pass_->setPref(cpq_ptr, pref_g, /*apply_fwd=*/true, /*apply_bwd=*/false);
  
// #ifdef RPU_DEBUG_CHOPPER_CHECK
//         static int q_dbg = 0;
//         if (q_dbg++ < 10) {

//           if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {

//             constexpr int K = 8;
//             T hq[K];

//             cudaStream_t s = this->context_->getStream();
//             cudaError_t err = cudaMemcpyAsync(hq, q_ptr, K * sizeof(T), cudaMemcpyDeviceToHost, s);
//             if (err != cudaSuccess) {
//               fprintf(stderr, "[CHOPPER][QDBG] cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
//               fflush(stderr);
//             } else {
//               err = cudaStreamSynchronize(s);
//               if (err != cudaSuccess) {
//                 fprintf(stderr, "[CHOPPER][QDBG] cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
//                 fflush(stderr);
//               } else {
//                 double mean = 0.0;
//                 double mx = -1e300;
//                 double amx = 0.0;

//                 for (int i = 0; i < K; ++i) {
//                   double v = (double)hq[i];
//                   mean += v;
//                   mx = std::max(mx, v);
//                   amx = std::max(amx, std::fabs(v));
//                 }
//                 mean /= (double)K;

//                 fprintf(stderr,
//                         "[CHOPPER][QDBG] FWD setPref fb_pass=%p Q=%p gamma=%g | "
//                         "Q[0:%d] mean=%+.4e max=%+.4e absmax=%+.4e | vals=",
//                         (void*)fb_pass_.get(), (const void*)q_ptr, (double)pref_g,
//                         K, mean, mx, amx);

//                 for (int i = 0; i < K; ++i) {
//                   fprintf(stderr, "%+.3e%s", (double)hq[i], (i + 1 == K ? "" : ","));
//                 }
//                 fprintf(stderr, "\n");
//                 fflush(stderr);
//               }
//             }

//           } else {
//             fprintf(stderr, "[CHOPPER][QDBG] skip Q sample stats for this T (not float/double)\n");
//             fflush(stderr);
//           }
//         }
// #endif
#ifdef RPU_DEBUG_CHOPPER_CHECK
      // ---- QDBG periodic print ----
      static long long q_call = 0;
      static int q_every = -1;
      static int q_left  = -1;

      if (q_every < 0) {
        const char *e = std::getenv("AIHWKIT_QDBG_EVERY");
        q_every = (e != nullptr) ? std::atoi(e) : 5000;   // default: every 5000 forward calls

        const char *m = std::getenv("AIHWKIT_QDBG_MAX");
        q_left = (m != nullptr) ? std::atoi(m) : 50;
          fprintf(stderr,
          "[CHOPPER][QDBG_INIT][FWD] env_every=%s env_max=%s => q_every=%d q_left=%d is_test=%d cpq_ptr=%p pref_g=%g\n",
          e ? e : "(null)",
          m ? m : "(null)",
          q_every, q_left,
          (int)is_test,
          (const void*)cpq_ptr,
          (double)pref_g);
  fflush(stderr);      // default: print at most 50 lines
      }

      q_call++;

      const bool do_print =  (q_left != 0) && (q_every > 0) && (q_call % q_every == 0);

      if (do_print) {
        if (q_left > 0) {
          q_left--;
        }

        if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
          constexpr int K = 80;
          T hq[K];

          cudaStream_t s = this->context_->getStream();
          cudaError_t err = cudaMemcpyAsync(hq, cpq_ptr, K * sizeof(T), cudaMemcpyDeviceToHost, s);

          if (err != cudaSuccess) {
            fprintf(stderr, "[CHOPPER][QDBG] cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
            fflush(stderr);
          } else {
            err = cudaStreamSynchronize(s);
            if (err != cudaSuccess) {
              fprintf(stderr, "[CHOPPER][QDBG] cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
              fflush(stderr);
            } else {
              double mean = 0.0;
              double mx = -1e300;
              double amx = 0.0;

              for (int i = 0; i < K; ++i) {
                double v = (double)hq[i];
                mean += v;
                mx = std::max(mx, v);
                amx = std::max(amx, std::fabs(v));
              }
              mean /= (double)K;

              fprintf(stderr,
                      "[CHOPPER][QDBG] FWD Q=%p gamma=%g | "
                      "Q[0:%d] mean=%+.4e max=%+.4e absmax=%+.4e | vals=",
                      (const void*)cpq_ptr, (double)pref_g,
                      K, mean, mx, amx);

              for (int i = 0; i < K; ++i) {
                fprintf(stderr, "%+.3e%s", (double)hq[i], (i + 1 == K ? "" : ","));
              }
              fprintf(stderr, "\n");
              fflush(stderr);
            }
          }
        } else {
          fprintf(stderr, "[CHOPPER][QDBG] skip Q sample stats for this T (not float/double)\n");
          fflush(stderr);
        }
      }
#endif

      } else {
        fb_pass_->clearPref();
      }

    } else {
      fb_pass_->clearPref();
    }

  } else {
    fb_pass_->clearPref();
  }

  fb_pass_->forwardMatrixIterator(
      this->getFBWeightsCuda(is_test), X_input, this->getXSize(), x_trans, D_output,
      this->getDSize(), d_trans, m_batch, this->getFwdAlpha(), *f_iom_, getMetaPar().f_io, is_test);
}


template <typename T>
void RPUCudaPulsed<T>::forwardVector(
    const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) {
  T *d_output_inc1 = d_output;
  const T *x_input_inc1 = x_input;

  if (d_inc != 1) {
    d_output_inc1 = dev_f_d_vector_inc1_->getData();
  }
  if (x_inc != 1) {
    // just copy for now. Only needed for looped matrix versions anyway
    RPU::math::copy<T>(
        this->context_, this->x_size_, x_input, x_inc, dev_f_x_vector_inc1_->getData(), 1);
    x_input_inc1 = dev_f_x_vector_inc1_->getDataConst();
  }

  forwardMatrixIterator(x_input_inc1, d_output_inc1, 1, false, false, is_test);

  if (d_inc != 1) {
    RPU::math::copy<T>(this->context_, this->d_size_, d_output_inc1, 1, d_output, d_inc);
  }
}

/*********************************************************************************/
/*********************************************************************************/
/* BACKWARD */

// template <typename T>
// void RPUCudaPulsed<T>::backwardMatrix(
//     const T *D_input, T *X_output, int m_batch, bool d_trans, bool x_trans) {
//       backwardMatrixIterator(D_input, X_output, m_batch, d_trans, x_trans);
// }
template <typename T>
void RPUCudaPulsed<T>::backwardMatrix(
    const T *D_input, T *X_output, int m_batch, bool d_trans, bool x_trans) {

  // -------- Pref runtime gate --------
  auto pref_on = []() -> bool {
    const char *e = std::getenv("AIHWKIT_PREF_ON");
    return (e != nullptr) && (std::atoi(e) != 0);
  };

  auto get_pref_gamma = []() -> T {
    const char *g = std::getenv("AIHWKIT_PREF_GAMMA");
    return (g != nullptr) ? (T)std::atof(g) : (T)0.1; // default 0.1
  };

  if (pref_on()) {
#ifdef RPU_DEBUG_CHOPPER_CHECK
    static int rtti_hit = 0;
    if (rtti_hit++ < 5) {
      fprintf(stderr, "[CHOPPER] (BWD) rpucuda_device RTTI=%s\n", typeid(*rpucuda_device_).name());
      fflush(stderr);
    }
#endif

    const T pref_g = get_pref_gamma();

    if (auto *dt = dynamic_cast<RPU::DynamicTransferRPUDeviceCuda<T> *>(rpucuda_device_.get())) {
      // const T *q_ptr = dt->getPastMeanDataConst();
      const T *cpq_ptr = dt->getCPQDataConst();
      // if (q_ptr != nullptr && pref_g != (T)0) {
      //   fb_pass_->setPref(q_ptr, pref_g, /*apply_fwd=*/false, /*apply_bwd=*/true);
      if (cpq_ptr != nullptr && pref_g != (T)0) {
        fb_pass_->setPref(cpq_ptr, pref_g, /*apply_fwd=*/false, /*apply_bwd=*/true);
#ifdef RPU_DEBUG_CHOPPER_CHECK
        static int q_dbg = 0;
        if (q_dbg++ < 10) {

          if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
            constexpr int K = 8;
            T hq[K];

            cudaStream_t s = this->context_->getStream();
            cudaError_t err = cudaMemcpyAsync(hq, cpq_ptr, K * sizeof(T), cudaMemcpyDeviceToHost, s);
            if (err != cudaSuccess) {
              fprintf(stderr, "[CHOPPER][QDBG] (BWD) cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
              fflush(stderr);
            } else {
              err = cudaStreamSynchronize(s);
              if (err != cudaSuccess) {
                fprintf(stderr, "[CHOPPER][QDBG] (BWD) cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
                fflush(stderr);
              } else {
                double mean = 0.0;
                double mx = -1e300;
                double amx = 0.0;

                for (int i = 0; i < K; ++i) {
                  double v = (double)hq[i];
                  mean += v;
                  mx = std::max(mx, v);
                  amx = std::max(amx, std::fabs(v));
                }
                mean /= (double)K;

                fprintf(stderr,
                        "[CHOPPER][QDBG] BWD setPref fb_pass=%p Q=%p gamma=%g | "
                        "Q[0:%d] mean=%+.4e max=%+.4e absmax=%+.4e | vals=",
                        (void*)fb_pass_.get(), (const void*)cpq_ptr, (double)pref_g,
                        K, mean, mx, amx);

                for (int i = 0; i < K; ++i) {
                  fprintf(stderr, "%+.3e%s", (double)hq[i], (i + 1 == K ? "" : ","));
                }
                fprintf(stderr, "\n");
                fflush(stderr);
              }
            }
          } else {
            fprintf(stderr, "[CHOPPER][QDBG] (BWD) skip Q sample stats for this T (not float/double)\n");
            fflush(stderr);
          }
        }
#endif

      } else {
        fb_pass_->clearPref();
      }
    } else {
      fb_pass_->clearPref();
    }

  } else {
    fb_pass_->clearPref();
  }

  backwardMatrixIterator(D_input, X_output, m_batch, d_trans, x_trans);
}


template <typename T>
void RPUCudaPulsed<T>::backwardIndexed(
    const T *D_input, T *X_output, int total_output_size, int m_batch, int dim3, bool trans) {

  int x_size = this->getXSize();
  const int *indices = this->getMatrixIndices();

  // need to set X_output to all zero for the atomics
  this->setZero(X_output, total_output_size);

  if ((dim3 == 1) || (!trans)) {

    IndexReaderOutputIterator<T> out_iter(
        X_output, indices, total_output_size / dim3, x_size * m_batch);

    backwardMatrixIterator(D_input, out_iter, m_batch * dim3, trans, trans);

  } else {

    IndexReaderTransOutputIterator<T> out_iter(
        X_output, indices, total_output_size / dim3, m_batch, x_size * m_batch, m_batch * dim3);

    PermuterTransInputIterator<T> permute_iter(
        D_input, m_batch, this->getDSize() * m_batch, m_batch * dim3);

    backwardMatrixIterator(permute_iter, out_iter, m_batch * dim3, trans, trans);
  }
}

template <typename T>
void RPUCudaPulsed<T>::backwardIndexedSlice(
    const T *D_input,
    T *X_output,
    int total_output_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices) {

  int x_size = this->getXSize();
  int d_size = this->getDSize();
  const int *indices = this->getMatrixIndices();

  // CAUTION: need X_output to be set to zero!
  if ((dim3 == 1) || (!trans)) {

    SliceInputIterator<false, T> in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    IndexReaderSliceOutputIterator<false, T> out_iter(
        X_output, indices, total_output_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->backwardMatrixIterator(in_iter, out_iter, m_batch_slice * dim3, trans, trans);

  } else {
    SliceInputIterator<true, T> in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    IndexReaderSliceOutputIterator<true, T> out_iter(
        X_output, indices, total_output_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->backwardMatrixIterator(in_iter, out_iter, m_batch_slice * dim3, trans, trans);
  }
}

template <typename T>
template <typename InputIteratorT, typename OutputIteratorT>
void RPUCudaPulsed<T>::backwardMatrixIterator(
    InputIteratorT D_input, OutputIteratorT X_output, int m_batch, bool d_trans, bool x_trans) {

  fb_pass_->backwardMatrixIterator(
      this->getFBWeightsCuda(false), D_input, this->getDSize(), d_trans, X_output, this->getXSize(),
      x_trans, m_batch, this->getBwdAlpha(), *b_iom_, getMetaPar().b_io);
};

template <typename T>
void RPUCudaPulsed<T>::backwardVector(const T *d_input, T *x_output, int d_inc, int x_inc) {
  const T *d_input_inc1 = d_input;
  T *x_output_inc1 = x_output;

  if (x_inc != 1) {
    x_output_inc1 = dev_b_x_vector_inc1_->getData();
  }

  if (d_inc != 1) { // only needed for looped updates anyways
    RPU::math::copy<T>(
        this->context_, this->d_size_, d_input, d_inc, dev_b_d_vector_inc1_->getData(), 1);
    d_input_inc1 = dev_b_d_vector_inc1_->getDataConst();
  }
  this->backwardMatrixIterator(d_input_inc1, x_output_inc1, 1, false, false);

  if (x_inc != 1) {
    RPU::math::copy<T>(this->context_, this->x_size_, x_output_inc1, 1, x_output, x_inc);
  }
}
/*********************************************************************************/
/* K-FAC */
namespace {

template <typename T>
inline void cuda_copy_d2h(cudaStream_t s, const T *src_dev, T *dst_host, size_t n) {
  if (n == 0) {
    return;
  }
  cudaError_t err = cudaMemcpyAsync(dst_host, src_dev, n * sizeof(T), cudaMemcpyDeviceToHost, s);
  if (err != cudaSuccess) {
    RPU_FATAL("cudaMemcpyAsync D2H failed in KFAC helper.");
  }
  err = cudaStreamSynchronize(s);
  if (err != cudaSuccess) {
    RPU_FATAL("cudaStreamSynchronize failed in KFAC helper (D2H).");
  }
}

template <typename T>
inline void cuda_copy_h2d(cudaStream_t s, const T *src_host, T *dst_dev, size_t n) {
  if (n == 0) {
    return;
  }
  cudaError_t err = cudaMemcpyAsync(dst_dev, src_host, n * sizeof(T), cudaMemcpyHostToDevice, s);
  if (err != cudaSuccess) {
    RPU_FATAL("cudaMemcpyAsync H2D failed in KFAC helper.");
  }
  err = cudaStreamSynchronize(s);
  if (err != cudaSuccess) {
    RPU_FATAL("cudaStreamSynchronize failed in KFAC helper (H2D).");
  }
}

template <typename T>
inline void host_eye(std::vector<T> &A, int n, T diag = (T)1) {
  A.assign(n * n, (T)0);
  for (int i = 0; i < n; ++i) {
    A[i * n + i] = diag;
  }
}

template <typename T>
bool host_invert_spd_block(
    const std::vector<T> &A_in,
    int n,
    T lambda,
    std::vector<T> &A_inv_out) {

  // Do the numerics in double for robustness, then cast back to T.
  std::vector<double> A(n * n, 0.0);
  for (int i = 0; i < n * n; ++i) {
    A[i] = static_cast<double>(A_in[i]);
  }

  // Enforce symmetry to match Python:
  // A <- 0.5 * (A + A^T)
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double v = 0.5 * (A[i * n + j] + A[j * n + i]);
      A[i * n + j] = v;
      A[j * n + i] = v;
    }
  }

  // Add damping
  for (int i = 0; i < n; ++i) {
    A[i * n + i] += static_cast<double>(lambda);
  }

  std::vector<double> L(n * n, 0.0);
  bool ok = false;

  // Escalating jitter
  for (int t = 0; t < 8; ++t) {
    std::vector<double> M = A;
    double jitter = 1e-12 * std::pow(10.0, t);
    for (int i = 0; i < n; ++i) {
      M[i * n + i] += jitter;
    }

    std::fill(L.begin(), L.end(), 0.0);
    ok = true;

    for (int i = 0; i < n && ok; ++i) {
      for (int j = 0; j <= i; ++j) {
        double sum = M[i * n + j];
        for (int k = 0; k < j; ++k) {
          sum -= L[i * n + k] * L[j * n + k];
        }
        if (i == j) {
          if (sum <= 0.0) {
            ok = false;
            break;
          }
          L[i * n + j] = std::sqrt(sum);
        } else {
          L[i * n + j] = sum / L[j * n + j];
        }
      }
    }

    if (ok) {
      break;
    }
  }

  if (!ok) {
    return false;
  }

  A_inv_out.assign(n * n, (T)0);
  std::vector<double> y(n, 0.0), x(n, 0.0);

  // Solve A X = I via LL^T X = I
  for (int col = 0; col < n; ++col) {
    for (int i = 0; i < n; ++i) {
      double rhs = (i == col) ? 1.0 : 0.0;
      for (int k = 0; k < i; ++k) {
        rhs -= L[i * n + k] * y[k];
      }
      y[i] = rhs / L[i * n + i];
    }

    for (int i = n - 1; i >= 0; --i) {
      double rhs = y[i];
      for (int k = i + 1; k < n; ++k) {
        rhs -= L[k * n + i] * x[k];
      }
      x[i] = rhs / L[i * n + i];
    }

    for (int row = 0; row < n; ++row) {
      A_inv_out[row * n + col] = static_cast<T>(x[row]);
    }
  }

  return true;
}

template <typename T>
inline void host_block_matvec(
    const std::vector<T> &F,
    const T *x,
    T *y,
    int n) {
  for (int r = 0; r < n; ++r) {
    double acc = 0.0;
    for (int c = 0; c < n; ++c) {
      acc += static_cast<double>(F[r * n + c]) * static_cast<double>(x[c]);
    }
    y[r] = static_cast<T>(acc);
  }
}

struct DeltaWCompareMetrics {
  double actual_norm = 0.0;
  double ideal_norm = 0.0;
  double rel_err = 0.0;
  double cosine = 1.0;
  double angle_deg = 0.0;
  double proj_scale = 0.0;
  double mag_ratio = 0.0;
  double orth_rel = 0.0;
  double max_abs_err = 0.0;
  double mean_abs_err = 0.0;
};

template <typename T>
void host_build_outer_update_col_major(
    const T *x,
    const T *d,
    int m_batch,
    int x_size,
    int d_size,
    double lr,
    std::vector<double> &dw_out) {

  dw_out.assign((size_t)d_size * (size_t)x_size, 0.0);
  const double alpha = -lr;

  for (int b = 0; b < m_batch; ++b) {
    const T *xb = x + (size_t)b * (size_t)x_size;
    const T *db = d + (size_t)b * (size_t)d_size;

    for (int col = 0; col < x_size; ++col) {
      const double xv = static_cast<double>(xb[col]);
      double *dw_col = dw_out.data() + (size_t)col * (size_t)d_size;
      for (int row = 0; row < d_size; ++row) {
        dw_col[row] += alpha * static_cast<double>(db[row]) * xv;
      }
    }
  }
}

template <typename T>
DeltaWCompareMetrics host_compare_weight_updates(
    const std::vector<T> &actual,
    const std::vector<double> &ideal) {

  if (actual.size() != ideal.size()) {
    RPU_FATAL("KFAC delta-W diagnostic size mismatch.");
  }

  DeltaWCompareMetrics metrics;

  double dot = 0.0;
  double actual_norm_sq = 0.0;
  double ideal_norm_sq = 0.0;
  double err_sq = 0.0;
  double abs_err_sum = 0.0;

  for (size_t i = 0; i < actual.size(); ++i) {
    const double a = static_cast<double>(actual[i]);
    const double b = ideal[i];
    const double diff = a - b;

    dot += a * b;
    actual_norm_sq += a * a;
    ideal_norm_sq += b * b;
    err_sq += diff * diff;
    abs_err_sum += std::abs(diff);
    metrics.max_abs_err = std::max(metrics.max_abs_err, std::abs(diff));
  }

  metrics.actual_norm = std::sqrt(std::max(actual_norm_sq, 0.0));
  metrics.ideal_norm = std::sqrt(std::max(ideal_norm_sq, 0.0));
  metrics.rel_err = std::sqrt(std::max(err_sq, 0.0)) / (metrics.ideal_norm + 1e-24);
  metrics.mean_abs_err = abs_err_sum / (double)std::max<size_t>(actual.size(), 1);
  metrics.proj_scale = dot / (ideal_norm_sq + 1e-24);
  metrics.mag_ratio = metrics.actual_norm / (metrics.ideal_norm + 1e-24);

  if (metrics.actual_norm > 0.0 && metrics.ideal_norm > 0.0) {
    metrics.cosine = dot / (metrics.actual_norm * metrics.ideal_norm + 1e-24);
    metrics.cosine = std::max(-1.0, std::min(1.0, metrics.cosine));
  } else if (metrics.actual_norm == 0.0 && metrics.ideal_norm == 0.0) {
    metrics.cosine = 1.0;
  } else {
    metrics.cosine = 0.0;
  }

  constexpr double kRadToDeg = 57.2957795130823208768;
  metrics.angle_deg = std::acos(metrics.cosine) * kRadToDeg;

  const double orth_sq =
      std::max(actual_norm_sq - dot * dot / (ideal_norm_sq + 1e-24), 0.0);
  metrics.orth_rel = std::sqrt(orth_sq) / (metrics.ideal_norm + 1e-24);

  return metrics;
}

inline bool host_compute_singular_values_col_major(
    const std::vector<double> &matrix_in,
    int rows,
    int cols,
    std::vector<double> &s_out) {

  const int rank = std::min(rows, cols);
  s_out.assign(rank, 0.0);

  if (rank == 0) {
    return true;
  }

  std::vector<double> matrix = matrix_in;
  std::vector<double> u_dummy(1, 0.0);
  std::vector<double> vt_dummy(1, 0.0);

  const int info = LAPACKE_dgesdd(
      LAPACK_COL_MAJOR, 'N', rows, cols, matrix.data(), rows, s_out.data(), u_dummy.data(), 1,
      vt_dummy.data(), 1);

  return info == 0;
}

inline std::string host_format_top_values(const std::vector<double> &values, int top_k) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(3);

  const int k = std::min((int)values.size(), std::max(top_k, 0));
  for (int i = 0; i < k; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  return oss.str();
}

template <typename T>
void dump_host_matrix_col_major(
    const char *fname,
    const std::vector<T> &buf,
    int rows,
    int cols,
    const char *tag,
    uint64_t tile_id,
    uint64_t step,
    int x_size,
    int d_size) {

  std::ofstream ofs(fname);
  ofs << "# tag=" << tag
      << " rows=" << rows
      << " cols=" << cols
      << " tile_id=" << (unsigned long long)tile_id
      << " step=" << (unsigned long long)step
      << " x_size=" << x_size
      << " d_size=" << d_size
      << "\n";
  ofs << std::setprecision(18);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      ofs << static_cast<double>(buf[(size_t)c * (size_t)rows + (size_t)r]);
      if (c + 1 < cols) {
        ofs << " ";
      }
    }
    ofs << "\n";
  }
}

} // namespace
template <typename T>
void RPUCudaPulsed<T>::setKFACConfig(
    bool enable,
    int block_x,
    int block_d,
    int update_freq,
    T beta,
    T eps) {   // rename lambda -> eps if possible

  kfac_enable_ = enable;
  kfac_block_x_ = std::max(0, block_x);
  kfac_block_d_ = std::max(0, block_d);
  kfac_update_freq_ = std::max(1, update_freq);
  kfac_beta_ = beta;
  kfac_eps_ = eps;
  kfac_step_ = 0;
  kfac_ready_ = false;
  kfac_stats_initialized_ = false;

  if (kfac_enable_) {
    initKFACBlocks();
    zeroKFACStats();   // zero S, identity F
  } else {
    kfac_x_block_starts_.clear();
    kfac_x_block_sizes_.clear();
    kfac_d_block_starts_.clear();
    kfac_d_block_sizes_.clear();

    dev_kfac_Sx_blocks_.clear();
    dev_kfac_Sd_blocks_.clear();
    dev_kfac_Fx_blocks_.clear();
    dev_kfac_Fd_blocks_.clear();

    dev_kfac_x_raw_.reset();
    dev_kfac_d_raw_.reset();
    dev_kfac_x_pre_.reset();
    dev_kfac_d_pre_.reset();

    kfac_buffer_mbatch_ = 0;
    kfac_stats_initialized_ = false;
  }
}
template <typename T>
void RPUCudaPulsed<T>::initKFACBlocks() {
  kfac_x_block_starts_.clear();
  kfac_x_block_sizes_.clear();
  kfac_d_block_starts_.clear();
  kfac_d_block_sizes_.clear();

  dev_kfac_Sx_blocks_.clear();
  dev_kfac_Sd_blocks_.clear();
  dev_kfac_Fx_blocks_.clear();
  dev_kfac_Fd_blocks_.clear();

  auto make_partition = [](int dim, int blk, std::vector<int> &starts, std::vector<int> &sizes) {
    if (blk <= 0 || blk >= dim) {
      starts.push_back(0);
      sizes.push_back(dim);
      return;
    }
    for (int s = 0; s < dim; s += blk) {
      starts.push_back(s);
      sizes.push_back(std::min(blk, dim - s));
    }
  };

  make_partition(this->x_size_, kfac_block_x_, kfac_x_block_starts_, kfac_x_block_sizes_);
  make_partition(this->d_size_, kfac_block_d_, kfac_d_block_starts_, kfac_d_block_sizes_);

  CudaContextPtr c = this->context_;

  for (size_t i = 0; i < kfac_x_block_sizes_.size(); ++i) {
    int bs = kfac_x_block_sizes_[i];
    dev_kfac_Sx_blocks_.emplace_back(RPU::make_unique<CudaArray<T>>(c, bs * bs));
    dev_kfac_Fx_blocks_.emplace_back(RPU::make_unique<CudaArray<T>>(c, bs * bs));
  }

  for (size_t i = 0; i < kfac_d_block_sizes_.size(); ++i) {
    int bs = kfac_d_block_sizes_[i];
    dev_kfac_Sd_blocks_.emplace_back(RPU::make_unique<CudaArray<T>>(c, bs * bs));
    dev_kfac_Fd_blocks_.emplace_back(RPU::make_unique<CudaArray<T>>(c, bs * bs));
  }
}

template <typename T>
void RPUCudaPulsed<T>::ensureKFACBuffers(int m_batch) {
  if (!kfac_enable_) {
    return;
  }
  if (m_batch <= kfac_buffer_mbatch_) {
    return;
  }

  CudaContextPtr c = this->context_;
  dev_kfac_x_raw_ = RPU::make_unique<CudaArray<T>>(c, this->x_size_ * m_batch);
  dev_kfac_d_raw_ = RPU::make_unique<CudaArray<T>>(c, this->d_size_ * m_batch);
  dev_kfac_x_pre_ = RPU::make_unique<CudaArray<T>>(c, this->x_size_ * m_batch);
  dev_kfac_d_pre_ = RPU::make_unique<CudaArray<T>>(c, this->d_size_ * m_batch);
  if (!dev_kfac_gnorm_buf_) {
  dev_kfac_gnorm_buf_ = RPU::make_unique<CudaArray<float>>(c, 2);
}
  kfac_buffer_mbatch_ = m_batch;
}


template <typename T>
void RPUCudaPulsed<T>::initKFACTrickRuntime() {
  auto env_on = [](const char *name, bool defval) -> bool {
    const char *e = std::getenv(name);
    return (e != nullptr) ? (std::atoi(e) != 0) : defval;
  };
  auto env_int = [](const char *name, int defval) -> int {
    const char *e = std::getenv(name);
    return (e != nullptr) ? std::atoi(e) : defval;
  };

  kfac_trick_enable_ = env_on("AIHWKIT_KFAC_TRICK_ON", false);
  kfac_trick_scaling_ = env_on("AIHWKIT_KFAC_TRICK_SCALING", true);
  kfac_trick_split_signal_ = env_on("AIHWKIT_KFAC_TRICK_SPLIT_SIGNAL", true);
  kfac_trick_n_grad_sample_ =
      std::max(1, env_int("AIHWKIT_KFAC_TRICK_N_GRAD_SAMPLE", 1));
  kfac_trick_grad_desired_bl_ =
      std::max(1, env_int("AIHWKIT_KFAC_TRICK_GRAD_DESIRED_BL", 255));
  kfac_trick_group_max_batch_ =
      std::max(1, env_int("AIHWKIT_KFAC_TRICK_GROUP_MAX_BATCH", 500));
}

template <typename T>
void RPUCudaPulsed<T>::initKFACTransferRuntime() {
  auto env_int = [](const char *name, int defval) -> int {
    const char *e = std::getenv(name);
    return (e != nullptr) ? std::atoi(e) : defval;
  };
  auto env_val = [](const char *name, T defval) -> T {
    const char *e = std::getenv(name);
    return (e != nullptr) ? (T)std::atof(e) : defval;
  };

  kfac_transfer_rows_per_step_ =
      std::max(1, env_int("AIHWKIT_KFAC_TRANSFER_ROWS_PER_STEP", 1));

  kfac_transfer_up_ = getMetaPar().up;
  kfac_transfer_up_.update_management = true;
  kfac_transfer_up_.update_bl_management = true;

  // optional: allow separate BL for transfer submit
  kfac_transfer_up_.desired_BL =
      std::max(1, env_int("AIHWKIT_KFAC_TRANSFER_DESIRED_BL", kfac_transfer_up_.desired_BL));

  T env_gran = env_val("AIHWKIT_KFAC_TRANSFER_GRANULARITY", (T)-1.0);
  if (env_gran > (T)0.0) {
    kfac_transfer_granularity_base_ = env_gran;
  } else {
    kfac_transfer_granularity_base_ = (T)1.0;
    if (auto *pulsed =
            dynamic_cast<PulsedRPUDeviceCudaBase<T> *>(rpucuda_device_.get())) {
      kfac_transfer_granularity_base_ = pulsed->getWeightGranularity();
    }
  }
  kfac_transfer_gran_shape_exp_ =
    env_val("AIHWKIT_KFAC_TRANSFER_GRAN_SHAPE_EXP", (T)0.5);
kfac_transfer_gran_min_mult_ =
    env_val("AIHWKIT_KFAC_TRANSFER_GRAN_MIN_MULT", (T)0.125);
kfac_transfer_gran_max_mult_ =
    env_val("AIHWKIT_KFAC_TRANSFER_GRAN_MAX_MULT", (T)8.0);

  kfac_transfer_row_idx_ = 0;
}


template <typename T>
void RPUCudaPulsed<T>::ensureKFACTrickBuffers(int m_batch) {
  if (!kfac_trick_enable_) {
    return;
  }

  const int x_need = this->x_size_ * m_batch;
  const int d_need = this->d_size_ * m_batch;
  const int w_need = this->x_size_ * this->d_size_;
  const int eye_need = this->x_size_ * this->x_size_;
  CudaContextPtr c = this->context_;
  cudaStream_t s = c->getStream();

  if (!dev_kfac_x_work_ || dev_kfac_x_work_->getSize() < x_need) {
    dev_kfac_x_work_ = RPU::make_unique<CudaArray<T>>(c, x_need);
  }
  if (!dev_kfac_d_work_ || dev_kfac_d_work_->getSize() < d_need) {
    dev_kfac_d_work_ = RPU::make_unique<CudaArray<T>>(c, d_need);
  }
  if (!dev_kfac_dw_sample_ || dev_kfac_dw_sample_->getSize() < w_need) {
    dev_kfac_dw_sample_ = RPU::make_unique<CudaArray<T>>(c, w_need);
  }
  if (!dev_kfac_dw_acc_ || dev_kfac_dw_acc_->getSize() < w_need) {
    dev_kfac_dw_acc_ = RPU::make_unique<CudaArray<T>>(c, w_need);
  }
  if (!dev_kfac_eye_ || dev_kfac_eye_->getSize() < eye_need) {
    dev_kfac_eye_ = RPU::make_unique<CudaArray<T>>(c, eye_need);
    std::vector<T> h_eye(eye_need, (T)0);
    for (int i = 0; i < this->x_size_; ++i) {
      h_eye[i * this->x_size_ + i] = (T)1;
    }
    cuda_copy_h2d(s, h_eye.data(), dev_kfac_eye_->getData(), eye_need);
  }
}
template <typename T>
void RPUCudaPulsed<T>::ensureKFACTransferBuffers(int n_rows) {
  if (!kfac_trick_enable_) {
    return;
  }

  CudaContextPtr c = this->context_;
  cudaStream_t s = c->getStream();

  const int w_need = this->x_size_ * this->d_size_;
  const int tmp_need = std::max(1, n_rows) * this->x_size_;
  const int dvec_need = this->d_size_ * this->d_size_;

  if (!dev_kfac_dw_residual_ || dev_kfac_dw_residual_->getSize() < w_need) {
    dev_kfac_dw_residual_ = RPU::make_unique<CudaArray<T>>(c, w_need);
    dev_kfac_dw_residual_->setConst((T)0);
  }

  if (!dev_kfac_transfer_tmp_ || dev_kfac_transfer_tmp_->getSize() < tmp_need) {
    dev_kfac_transfer_tmp_ = RPU::make_unique<CudaArray<T>>(c, tmp_need);
  }

  if (!dev_kfac_transfer_d_vecs_ || dev_kfac_transfer_d_vecs_->getSize() < dvec_need) {
    dev_kfac_transfer_d_vecs_ = RPU::make_unique<CudaArray<T>>(c, dvec_need);

    std::vector<T> h_eye(dvec_need, (T)0);
    for (int i = 0; i < this->d_size_; ++i) {
      h_eye[i * this->d_size_ + i] = (T)1;
    }
    cuda_copy_h2d(s, h_eye.data(), dev_kfac_transfer_d_vecs_->getData(), dvec_need);
  }
}


template <typename T>
template <typename InputIteratorT>
void RPUCudaPulsed<T>::materializeUpdateInput(InputIteratorT in, T *buf, int size) {
  RPU::math::copyWithIterator(this->context_, buf, in, size);
}

template <typename T>
void RPUCudaPulsed<T>::zeroKFACStats() {
  cudaStream_t s = this->context_->getStream();

  for (size_t bi = 0; bi < dev_kfac_Sx_blocks_.size(); ++bi) {
    int bs = kfac_x_block_sizes_[bi];
    std::vector<T> S(bs * bs, (T)0), F;
    host_eye(F, bs, (T)1);
    cuda_copy_h2d(s, S.data(), dev_kfac_Sx_blocks_[bi]->getData(), bs * bs);
    cuda_copy_h2d(s, F.data(), dev_kfac_Fx_blocks_[bi]->getData(), bs * bs);
  }

  for (size_t bj = 0; bj < dev_kfac_Sd_blocks_.size(); ++bj) {
    int bs = kfac_d_block_sizes_[bj];
    std::vector<T> S(bs * bs, (T)0), F;
    host_eye(F, bs, (T)1);
    cuda_copy_h2d(s, S.data(), dev_kfac_Sd_blocks_[bj]->getData(), bs * bs);
    cuda_copy_h2d(s, F.data(), dev_kfac_Fd_blocks_[bj]->getData(), bs * bs);
  }
}
template <typename T>
void RPUCudaPulsed<T>::updateKFACStats(const T *x_ptr, const T *d_ptr, int m_batch) {
  if (!kfac_enable_ || m_batch <= 0) {
    return;
  }

  cudaStream_t s = this->context_->getStream();
  dim3 block(16, 16);

  const int initialize = kfac_stats_initialized_ ? 0 : 1;

  // ---- X-side blocks ----
  for (size_t bi = 0; bi < kfac_x_block_sizes_.size(); ++bi) {
    const int start = kfac_x_block_starts_[bi];
    const int bs    = kfac_x_block_sizes_[bi];

    dim3 grid((bs + block.x - 1) / block.x,
              (bs + block.y - 1) / block.y);

    kernelUpdateCovEMA<T><<<grid, block, 0, s>>>(
        x_ptr,
        dev_kfac_Sx_blocks_[bi]->getData(),
        this->x_size_,
        start,
        bs,
        m_batch,
        kfac_beta_,
        (T)1,          // X-side: no extra scaling
        initialize);
  }

  // ---- D-side blocks ----
  for (size_t bj = 0; bj < kfac_d_block_sizes_.size(); ++bj) {
    const int start = kfac_d_block_starts_[bj];
    const int bs    = kfac_d_block_sizes_[bj];

    dim3 grid((bs + block.x - 1) / block.x,
              (bs + block.y - 1) / block.y);

    kernelUpdateCovEMA<T><<<grid, block, 0, s>>>(
        d_ptr,
        dev_kfac_Sd_blocks_[bj]->getData(),
        this->d_size_,
        start,
        bs,
        m_batch,
        kfac_beta_,
        (T)m_batch,    // D-side: undo mean-loss scaling for statistics only
        initialize);
  }

  kfac_stats_initialized_ = true;
}
template <typename T>
void RPUCudaPulsed<T>::refreshKFACFactors() {
  if (!kfac_enable_ || !kfac_stats_initialized_) {
    return;
  }

  cudaStream_t s = this->context_->getStream();

  // Python parity (pi=False, num_locations=1):
  const T damp_x = (T)std::sqrt(std::max((double)kfac_eps_, 0.0));
  const T damp_d = (T)std::sqrt(std::max((double)kfac_eps_, 0.0));

  for (size_t bi = 0; bi < kfac_x_block_sizes_.size(); ++bi) {
    const int bs = kfac_x_block_sizes_[bi];
    std::vector<T> S(bs * bs), F;
    cuda_copy_d2h(s, dev_kfac_Sx_blocks_[bi]->getDataConst(), S.data(), S.size());

    bool ok = host_invert_spd_block(S, bs, damp_x, F);
    if (!ok) {
      RPU_FATAL("KFAC X-block inversion failed.");
    }
    cuda_copy_h2d(s, F.data(), dev_kfac_Fx_blocks_[bi]->getData(), F.size());
  }

  for (size_t bj = 0; bj < kfac_d_block_sizes_.size(); ++bj) {
    const int bs = kfac_d_block_sizes_[bj];
    std::vector<T> S(bs * bs), F;
    cuda_copy_d2h(s, dev_kfac_Sd_blocks_[bj]->getDataConst(), S.data(), S.size());

    bool ok = host_invert_spd_block(S, bs, damp_d, F);
    if (!ok) {
      RPU_FATAL("KFAC D-block inversion failed.");
    }
    cuda_copy_h2d(s, F.data(), dev_kfac_Fd_blocks_[bj]->getData(), F.size());
  }

  kfac_ready_ = true;
}
template <typename T>
void RPUCudaPulsed<T>::applyBlockKFACX(const T *x_in, T *x_out, int m_batch) {
  this->setZero(x_out, this->x_size_ * m_batch);

  cudaStream_t s = this->context_->getStream();

  for (size_t bi = 0; bi < kfac_x_block_sizes_.size(); ++bi) {
    const int start = kfac_x_block_starts_[bi];
    const int bs    = kfac_x_block_sizes_[bi];

    const T *F_dev = dev_kfac_Fx_blocks_[bi]->getDataConst();

    dim3 block(128);
    dim3 grid((bs + block.x - 1) / block.x, m_batch);

    kernelBlockMatVec<T><<<grid, block, 0, s>>>(
        x_in, x_out, F_dev, this->x_size_, start, bs, m_batch);
  }
}
template <typename T>
void RPUCudaPulsed<T>::applyBlockKFACD(const T *d_in, T *d_out, int m_batch) {
  // 先清零输出
  this->setZero(d_out, this->d_size_ * m_batch);

  cudaStream_t s = this->context_->getStream();

  for (size_t bi = 0; bi < kfac_d_block_sizes_.size(); ++bi) {
    const int start = kfac_d_block_starts_[bi];
    const int bs    = kfac_d_block_sizes_[bi];

    const T *F_dev = dev_kfac_Fd_blocks_[bi]->getDataConst();

    dim3 block(128);
    dim3 grid((bs + block.x - 1) / block.x, m_batch);

    kernelBlockMatVec<T><<<grid, block, 0, s>>>(
        d_in, d_out, F_dev, this->d_size_, start, bs, m_batch);
  }
}


template <typename T>
void RPUCudaPulsed<T>::buildKFACDeltaWithAnalogGradTricks(
    const T *x_pre, const T *d_pre, int m_batch, T lr, T *dw_out) {

  const auto &up = getMetaPar().up;
  cudaStream_t s = this->context_->getStream();
  const int X = this->x_size_;
  const int D = this->d_size_;
  const int W = X * D;
  const T eps = (T)1e-11;

  std::vector<T> hx(m_batch * X), hd(m_batch * D);
  cuda_copy_d2h(s, x_pre, hx.data(), hx.size());
  cuda_copy_d2h(s, d_pre, hd.data(), hd.size());

  auto build_groups = [&](const std::vector<T> &hd_host) {
    std::vector<std::vector<int>> groups;
    if (!kfac_trick_split_signal_ || m_batch <= kfac_trick_group_max_batch_) {
      groups.emplace_back();
      groups.back().reserve(m_batch);
      for (int i = 0; i < m_batch; ++i) groups.back().push_back(i);
      return groups;
    }

    std::vector<double> grad_magnitude(m_batch, 0.0);
    double max_grad_norm = 0.0;
    double min_grad_norm = std::numeric_limits<double>::infinity();
    for (int b = 0; b < m_batch; ++b) {
      double n2 = 0.0;
      for (int j = 0; j < D; ++j) {
        const double v = (double)hd_host[b * D + j];
        n2 += v * v;
      }
      const double g = std::sqrt(n2);
      grad_magnitude[b] = g;
      max_grad_norm = std::max(max_grad_norm, g);
      min_grad_norm = std::min(min_grad_norm, g);
    }

    const double gamma = 0.5;
    const double min_grad = 1e-15;
    std::vector<double> delimiters = {max_grad_norm};
    while (!delimiters.empty() && delimiters.back() > min_grad_norm &&
           delimiters.back() > min_grad) {
      delimiters.push_back(delimiters.back() * gamma);
    }
    delimiters.push_back(0.0);

    const int max_bin_size = kfac_trick_group_max_batch_;
    for (size_t i = 0; i + 1 < delimiters.size(); ++i) {
      std::vector<int> indices;
      indices.reserve(m_batch);
      for (int b = 0; b < m_batch; ++b) {
        const double g = grad_magnitude[b];
        if (g <= delimiters[i] && g > delimiters[i + 1]) {
          indices.push_back(b);
        }
      }
      if (indices.empty()) {
        continue;
      }
      if ((int)indices.size() > max_bin_size) {
        for (int start = 0; start < (int)indices.size(); start += max_bin_size) {
          const int end = std::min(start + max_bin_size, (int)indices.size());
          groups.emplace_back(indices.begin() + start, indices.begin() + end);
        }
      } else {
        groups.push_back(std::move(indices));
      }
    }
    return groups;
  };

  auto groups = build_groups(hd);
  std::vector<double> h_dw_acc(W, 0.0);

  PulsedUpdateMetaParameter<T> up_tmp(up);
  up_tmp.desired_BL = kfac_trick_grad_desired_bl_;
  up_tmp.fixed_BL = true;
  up_tmp.update_bl_management = false;
  up_tmp.update_management = true;
  up_tmp.pulse_type = PulseType::StochasticCompressed;

  for (int sample_idx = 0; sample_idx < kfac_trick_n_grad_sample_; ++sample_idx) {
    auto *tmp_device = rpucuda_device_->clone();
    CudaArray<T> dev_tmp_weights(this->context_, W);
    this->setZero(dev_tmp_weights.getData(), W);

    std::vector<double> h_dw_sample(W, 0.0);

    for (const auto &g : groups) {
      const int B = (int)g.size();
      std::vector<T> hxg(B * X), hdg(B * D);

      for (int bi = 0; bi < B; ++bi) {
        const int src = g[bi];
        std::copy_n(&hx[src * X], X, &hxg[bi * X]);
        std::copy_n(&hd[src * D], D, &hdg[bi * D]);
      }


      double x_mean = 0.0, d_mean = 0.0, x_max = 0.0, d_scale = 0.0;
      std::vector<double> x_sum(X, 0.0), d_sum(D, 0.0);

      for (int i = 0; i < B * X; ++i) {
        const double v = (double)hxg[i];
        x_mean += v;
        x_max = std::max(x_max, std::abs(v));
      }
      for (int i = 0; i < B * D; ++i) {
        const double v = (double)hdg[i];
        d_mean += v;
      }
      x_mean /= std::max(1, B * X);
      d_mean /= std::max(1, B * D);

      // d-side robust scale: use centered RMS instead of global max
      double d_sq_sum = 0.0;
      for (int i = 0; i < B * D; ++i) {
        const double vc = (double)hdg[i] - d_mean;
        d_sq_sum += vc * vc;
      }
      d_scale = std::sqrt(d_sq_sum / std::max(1, B * D));

      for (int b = 0; b < B; ++b) {
        for (int i = 0; i < X; ++i) x_sum[i] += (double)hxg[b * X + i];
        for (int j = 0; j < D; ++j) d_sum[j] += (double)hdg[b * D + j];
      }

      T C = (T)1.0;
      if (kfac_trick_scaling_) {
        T dw_min = (T)1.0;
        if (auto *pulsed =
                dynamic_cast<PulsedRPUDeviceCudaBase<T> *>(rpucuda_device_.get())) {
          dw_min = pulsed->getWeightGranularity();
        }
        C = (T)std::sqrt((double)kfac_trick_grad_desired_bl_ * (double)dw_min);
        for (int i = 0; i < B * X; ++i) {
          hxg[i] = (T)(C * (((double)hxg[i] - x_mean) / (x_max + (double)eps)));
        }
        for (int i = 0; i < B * D; ++i) {
          hdg[i] = (T)(C * (((double)hdg[i] - d_mean) / (d_scale + (double)eps)));
        }
      }

      cuda_copy_h2d(s, hxg.data(), dev_kfac_x_work_->getData(), B * X);
      cuda_copy_h2d(s, hdg.data(), dev_kfac_d_work_->getData(), B * D);
      this->setZero(dev_tmp_weights.getData(), W);

      up_pwu_->update(
          dev_kfac_x_work_->getDataConst(),
          dev_kfac_d_work_->getDataConst(),
          dev_tmp_weights.getData(),
          tmp_device,
          up_tmp,
          lr,
          B,
          false,
          false);

      std::vector<T> h_dw_group(W);
      cuda_copy_d2h(s, dev_tmp_weights.getDataConst(), h_dw_group.data(), W);

      for (int x = 0; x < X; ++x) {
        for (int d = 0; d < D; ++d) {
          const int idx = d + x * D; // col-major [d, x]
          double v = (double)h_dw_group[idx];

          if (kfac_trick_scaling_) {
            v /= ((double)C * (double)C);
            v *= (x_max + (double)eps) * (d_scale + (double)eps);
            // h_dw_group already lives in update space (roughly -lr * grad'),
            // so the mean-offset restoration terms also need the update-space
            // scale and sign, unlike the original Python gradient-space formula.
            v -= (double)lr * d_mean * x_sum[x];
            v -= (double)lr * x_mean * d_sum[d];
            v += (double)lr * (double)B * x_mean * d_mean;
          }

          h_dw_sample[idx] += v;
        }
      }

    }
    delete tmp_device;
    for (int i = 0; i < W; ++i) h_dw_acc[i] += h_dw_sample[i];
  }

  std::vector<T> h_out(W);
  for (int i = 0; i < W; ++i) {
    h_out[i] = (T)(h_dw_acc[i] / (double)kfac_trick_n_grad_sample_);
  }
  cuda_copy_h2d(s, h_out.data(), dw_out, W);
}

template <typename T>
void RPUCudaPulsed<T>::submitKFACDenseDelta(const T *dw, T lr) {
  (void)lr;
  const T fwd_alpha = this->getFwdAlpha();
  const T bwd_alpha = this->getBwdAlpha();

  this->setFwdAlpha((T)1.0, false);
  this->setBwdAlpha((T)1.0, false);

  // 如果后面发现这里多吃了一次 lr，再把 tile lr 临时设成 1
  this->updateMatrix(dev_kfac_eye_->getDataConst(), dw, this->x_size_, false, true);

  this->setFwdAlpha(fwd_alpha, false);
  this->setBwdAlpha(bwd_alpha, false);
}

template <typename T>
void RPUCudaPulsed<T>::accumulateKFACResidual(const T *dw_step) {
  const int w_size = this->x_size_ * this->d_size_;
  RPU::math::elemaddscale<T>(
      this->context_,
      dev_kfac_dw_residual_->getData(),
      w_size,
      dw_step,
      (T)1.0);
}

template <typename T>
void RPUCudaPulsed<T>::submitKFACResidualTransfer(
    T *dev_weights, uint64_t step_id, uint64_t tile_id, bool diag_match) {
  int n_rows = std::min(kfac_transfer_rows_per_step_, this->d_size_);
  int i_row = kfac_transfer_row_idx_;
  cudaStream_t s = this->context_->getStream();
    auto env_on = [](const char *name, bool defval = false) -> bool {
    const char *e = std::getenv(name);
    return (e != nullptr) ? (std::atoi(e) != 0) : defval;
  };
  auto env_int = [](const char *name, int defval) -> int {
    const char *e = std::getenv(name);
    return (e != nullptr) ? std::atoi(e) : defval;
  };

  const bool xfer_cum_on = env_on("AIHWKIT_KFAC_XFER_CUM_DIAG", false);
  const int xfer_cum_freq = std::max(1, env_int("AIHWKIT_KFAC_XFER_CUM_FREQ", 100));
  const size_t weight_size = (size_t)this->x_size_ * (size_t)this->d_size_;
  std::vector<T> h_dw_build_full;
  if (xfer_cum_on) {
    h_dw_build_full.resize(weight_size);
    cuda_copy_d2h(s, dev_kfac_dw_acc_->getDataConst(), h_dw_build_full.data(), weight_size);
  }


  auto submit_chunk = [&](int row_start, int rows) {
    if (rows <= 0) {
      return;
    }

    int total = rows * this->x_size_;
    int nthreads = this->context_->getNThreads();
    int nblocks = this->context_->getNBlocks(total, nthreads);
T gran_tile = kfac_transfer_granularity_base_ *
    (T)std::pow((double)this->d_size_ / (double)this->x_size_,
                (double)kfac_transfer_gran_shape_exp_);

gran_tile = std::max(gran_tile,
                     kfac_transfer_granularity_base_ * kfac_transfer_gran_min_mult_);
gran_tile = std::min(gran_tile,
                     kfac_transfer_granularity_base_ * kfac_transfer_gran_max_mult_);

    kernelKFACResidualToTransfer<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
        dev_kfac_transfer_tmp_->getData(),
        dev_kfac_dw_residual_->getData(),
        this->x_size_,
        this->d_size_,
        row_start,
        rows,
        gran_tile);

        auto *transfer_dev_dbg = dynamic_cast<TransferRPUDeviceCuda<T> *>(rpucuda_device_.get());
    const bool actual_from_device0 =
        (transfer_dev_dbg != nullptr) && transfer_dev_dbg->isFullyHiddenDebug();

    const T *actual_weights_ptr_before =
        actual_from_device0 ? transfer_dev_dbg->getDeviceWeightsPtrDebug(0) : dev_weights;

    std::vector<T> h_commit_rowmajor;
    std::vector<T> h_w_before;
    if (diag_match || xfer_cum_on) {
      h_commit_rowmajor.resize((size_t)rows * (size_t)this->x_size_);
      h_w_before.resize((size_t)this->x_size_ * (size_t)this->d_size_);
      cuda_copy_d2h(s, dev_kfac_transfer_tmp_->getDataConst(), h_commit_rowmajor.data(), total);
      cuda_copy_d2h(s, actual_weights_ptr_before, h_w_before.data(), h_w_before.size());
    }


    const T *transfer_d =
        dev_kfac_transfer_d_vecs_->getDataConst() + (size_t)row_start * (size_t)this->d_size_;

        if (auto *transfer_dev = dynamic_cast<TransferRPUDeviceCuda<T> *>(rpucuda_device_.get())) {
      transfer_dev->applyExternalUpdateToDevice0(
          dev_weights,
          dev_kfac_transfer_tmp_->getDataConst(),
          transfer_d,
          rows,
          gran_tile,
          kfac_transfer_up_);
    } else {
      up_pwu_kfac_transfer_->update(
          dev_kfac_transfer_tmp_->getDataConst(),  // x side: integer pulse counts
          transfer_d,                              // d side: one-hot rows
          dev_weights,
          &*rpucuda_device_,
          kfac_transfer_up_,
          gran_tile,
          rows,
          false,
          false);
    }


    if (diag_match || xfer_cum_on) {
            const T *actual_weights_ptr_after =
          actual_from_device0 ? transfer_dev_dbg->getDeviceWeightsPtrDebug(0) : dev_weights;

      std::vector<T> h_w_after((size_t)this->x_size_ * (size_t)this->d_size_);
      cuda_copy_d2h(s, actual_weights_ptr_after, h_w_after.data(), h_w_after.size());

      std::vector<T> h_commit_colmajor((size_t)rows * (size_t)this->x_size_, (T)0);
      std::vector<T> h_actual_chunk((size_t)rows * (size_t)this->x_size_, (T)0);

      int nnz_q = 0;
      T max_abs_q = (T)0;
      for (int local_row = 0; local_row < rows; ++local_row) {
        for (int col = 0; col < this->x_size_; ++col) {
          const size_t rowmajor_idx =
              (size_t)local_row * (size_t)this->x_size_ + (size_t)col;
          const T q = h_commit_rowmajor[rowmajor_idx];
          if (q != (T)0) {
            nnz_q++;
          }
          max_abs_q = std::max(max_abs_q, (T)std::abs((double)q));
          const size_t colmajor_idx = (size_t)col * (size_t)rows + (size_t)local_row;
          h_commit_colmajor[colmajor_idx] = (T)(-(double)gran_tile * (double)q);

          // const size_t colmajor_idx = (size_t)col * (size_t)rows + (size_t)local_row;
          // h_commit_colmajor[colmajor_idx] = (T)((double)gran_tile * (double)q);

          const size_t full_idx =
              (size_t)col * (size_t)this->d_size_ + (size_t)(row_start + local_row);
          h_actual_chunk[colmajor_idx] = h_w_after[full_idx] - h_w_before[full_idx];
        }
      }
            if (xfer_cum_on) {
        auto &state = kfac_xfer_cum_states_[tile_id];
        if (state.ideal.size() != weight_size) {
          state.ideal.assign(weight_size, 0.0);
          state.commit.assign(weight_size, 0.0);
          state.actual.assign(weight_size, 0.0);
          state.window_start_step = step_id;
          state.n_accum = 0;
        }
        if (state.n_accum == 0) {
          state.window_start_step = step_id;
        }

        for (int local_row = 0; local_row < rows; ++local_row) {
          for (int col = 0; col < this->x_size_; ++col) {
            const size_t colmajor_chunk_idx =
                (size_t)col * (size_t)rows + (size_t)local_row;
            const size_t full_idx =
                (size_t)col * (size_t)this->d_size_ + (size_t)(row_start + local_row);

            state.ideal[full_idx] += (double)h_dw_build_full[full_idx] * ((double)this->d_size_ / (double)rows);
            state.commit[full_idx] += (double)h_commit_colmajor[colmajor_chunk_idx];
            state.actual[full_idx] += (double)h_actual_chunk[colmajor_chunk_idx];
          }
        }

        state.last_step = step_id;
        state.n_accum++;
      }

    if (diag_match) {
      std::vector<double> h_commit_ideal(h_commit_colmajor.size(), 0.0);
      for (size_t i = 0; i < h_commit_colmajor.size(); ++i) {
        h_commit_ideal[i] = static_cast<double>(h_commit_colmajor[i]);
      }

      DeltaWCompareMetrics xfer_metrics =
          host_compare_weight_updates(h_actual_chunk, h_commit_ideal);

            fprintf(
          stderr,
          "[KFAC-XFER] step=%llu tile_id=%llu row_start=%d rows=%d x_size=%d d_size=%d "
          "actual_src=%s gran=%.6e nnz_q=%d max_abs_q=%.6e ||actual||=%.6e ||commit||=%.6e "
          "cos=%.6f angle_deg=%.4f proj_scale=%.6e mag_ratio=%.6e rel_err=%.6e "
          "max_abs_err=%.6e mean_abs_err=%.6e\n",
          (unsigned long long)step_id,
          (unsigned long long)tile_id,
          row_start,
          rows,
          this->x_size_,
          this->d_size_,
          actual_from_device0 ? "device0" : "visible",
          (double)gran_tile,
          nnz_q,
          (double)max_abs_q,
          xfer_metrics.actual_norm,
          xfer_metrics.ideal_norm,
          xfer_metrics.cosine,
          xfer_metrics.angle_deg,
          xfer_metrics.proj_scale,
          xfer_metrics.mag_ratio,
          xfer_metrics.rel_err,
          xfer_metrics.max_abs_err,
          xfer_metrics.mean_abs_err);

      fflush(stderr);
    }
    }
  };

  int n_rest = this->d_size_ - i_row;
  if (n_rest < n_rows) {
    submit_chunk(i_row, n_rest);
    submit_chunk(0, n_rows - n_rest);
  } else {
    submit_chunk(i_row, n_rows);
  }
  if (xfer_cum_on && ((step_id + 1) % (uint64_t)xfer_cum_freq) == 0) {
    auto it = kfac_xfer_cum_states_.find(tile_id);
    if (it != kfac_xfer_cum_states_.end() && it->second.n_accum > 0) {
      auto &state = it->second;

      DeltaWCompareMetrics actual_vs_commit =
          host_compare_weight_updates(state.actual, state.commit);
      DeltaWCompareMetrics commit_vs_ideal =
          host_compare_weight_updates(state.commit, state.ideal);
      DeltaWCompareMetrics actual_vs_ideal =
          host_compare_weight_updates(state.actual, state.ideal);

      fprintf(
          stderr,
          "[KFAC-XFER-CUM] step=%llu tile_id=%llu x_size=%d d_size=%d "
          "window=[%llu,%llu] n_accum=%d "
          "A_vs_C cos=%.6f mag=%.6e rel=%.6e "
          "C_vs_I cos=%.6f mag=%.6e rel=%.6e "
          "A_vs_I cos=%.6f mag=%.6e rel=%.6e\n",
          (unsigned long long)step_id,
          (unsigned long long)tile_id,
          this->x_size_,
          this->d_size_,
          (unsigned long long)state.window_start_step,
          (unsigned long long)state.last_step,
          state.n_accum,
          actual_vs_commit.cosine,
          actual_vs_commit.mag_ratio,
          actual_vs_commit.rel_err,
          commit_vs_ideal.cosine,
          commit_vs_ideal.mag_ratio,
          commit_vs_ideal.rel_err,
          actual_vs_ideal.cosine,
          actual_vs_ideal.mag_ratio,
          actual_vs_ideal.rel_err);

      fflush(stderr);

      // std::fill(state.ideal.begin(), state.ideal.end(), 0.0);
      // std::fill(state.commit.begin(), state.commit.end(), 0.0);
      // std::fill(state.actual.begin(), state.actual.end(), 0.0);
      // state.window_start_step = step_id + 1;
      // state.last_step = step_id;
      // state.n_accum = 0;
    }
  }

  kfac_transfer_row_idx_ = (i_row + n_rows) % this->d_size_;
}


/*********************************************************************************/
/* UPDATE */
template <typename T> void RPUCudaPulsed<T>::finishUpdateCalculations() {
  if (getMetaPar().up.pulse_type != PulseType::None) {
    up_pwu_->waitForUpdateCalculations();
  }
}

template <typename T> void RPUCudaPulsed<T>::makeUpdateAsync() {
  if (getMetaPar().up.pulse_type != PulseType::None) {
    up_pwu_->makeUpdateAsync();
  }
}

template <typename T>
void RPUCudaPulsed<T>::updateMatrix(
    const T *X_input, const T *D_input, int m_batch, bool x_trans, bool d_trans) {
  updateMatrixIterator(X_input, D_input, m_batch, x_trans, d_trans);
}

template <typename T>
void RPUCudaPulsed<T>::updateIndexed(
    const T *X_input, const T *D_input, int total_input_size, int m_batch, int dim3, bool trans) {

  const int *indices = this->getMatrixIndices();
  int x_size = this->getXSize();

  if (trans && (dim3 > 1)) {
    IndexReaderTransInputIterator<T> in_iter(
        X_input, indices, total_input_size / dim3, m_batch, x_size * m_batch, m_batch * dim3);

    PermuterTransInputIterator<T> permute_iter(
        D_input, m_batch, this->getDSize() * m_batch, m_batch * dim3);
    updateMatrixIterator(in_iter, permute_iter, m_batch * dim3, trans, trans);

  } else {

    IndexReaderInputIterator<T> in_iter(
        X_input, indices, total_input_size / dim3, x_size * m_batch);
    updateMatrixIterator(in_iter, D_input, m_batch * dim3, trans, trans);
  }
}

template <typename T>
void RPUCudaPulsed<T>::updateIndexedSlice(
    const T *X_input,
    const T *D_input,
    int total_input_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices) {

  auto env_on = [](const char *name) -> bool {
    const char *e = std::getenv(name);
    return (e != nullptr) && (std::atoi(e) != 0);
  };

  auto env_int = [](const char *name, int defval) -> int {
    const char *e = std::getenv(name);
    return (e != nullptr) ? std::atoi(e) : defval;
  };

  auto env_ull = [](const char *name, unsigned long long defval) -> unsigned long long {
    const char *e = std::getenv(name);
    return (e != nullptr) ? std::strtoull(e, nullptr, 0) : defval;
  };

  const int *indices = this->getMatrixIndices();
  int x_size = this->getXSize();
  int d_size = this->getDSize();
  const uint64_t this_tile_id =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));
  const int dbg_rows = m_batch_slice * dim3;
  const bool use_raw_trick_indexed_d_materialize_fix =
      env_on("AIHWKIT_KFAC_RAW_TRICK_ON") && !trans;
  const bool use_kfac_indexed_d_materialize_fix =
      (kfac_enable_ || use_raw_trick_indexed_d_materialize_fix) && !trans;
  const bool iter_debug = env_on("AIHWKIT_KFAC_ITER_DEBUG");
  const int iter_debug_xsize = env_int("AIHWKIT_KFAC_ITER_DEBUG_XSIZE", -1);
  const int iter_debug_dsize = env_int("AIHWKIT_KFAC_ITER_DEBUG_DSIZE", -1);
  const int iter_debug_step = env_int("AIHWKIT_KFAC_ITER_DEBUG_STEP", -1);
  const int iter_debug_every = env_int("AIHWKIT_DEBUG_EVERY", 0);
  const int opt_step = env_int("AIHWKIT_DEBUG_OPT_STEP", -1);
  const int min_step = env_int("AIHWKIT_DEBUG_MIN_STEP", 0);
  const int iter_debug_head = std::max(0, env_int("AIHWKIT_KFAC_ITER_DEBUG_HEAD", 8));
  const unsigned long long iter_debug_tile =
      env_ull("AIHWKIT_KFAC_ITER_DEBUG_TILE", 0ULL);
  const bool iter_match =
      iter_debug &&
      (iter_debug_xsize < 0 || iter_debug_xsize == x_size) &&
      (iter_debug_dsize < 0 || iter_debug_dsize == d_size) &&
      (iter_debug_tile == 0ULL || iter_debug_tile == (unsigned long long)this_tile_id) &&
      (iter_debug_step < 0 || iter_debug_step == opt_step) &&
      (iter_debug_every <= 0 || (opt_step >= min_step && (opt_step % iter_debug_every) == 0));

  auto emit_iter_summary = [&](auto &iter, const char *tag) {
    if (!iter_match) {
      return;
    }

    const int total = d_size * dbg_rows;
    CudaArray<T> dev_tmp(this->context_, total);
    this->materializeUpdateInput(iter, dev_tmp.getData(), total);

    std::vector<T> hbuf(total);
    cuda_copy_d2h(this->context_->getStream(), dev_tmp.getDataConst(), hbuf.data(), total);

    double norm2 = 0.0;
    double max_abs = 0.0;
    int nnz = 0;
    for (int i = 0; i < total; ++i) {
      const double v = static_cast<double>(hbuf[i]);
      norm2 += v * v;
      max_abs = std::max(max_abs, std::abs(v));
      if (v != 0.0) {
        nnz++;
      }
    }

    fprintf(
        stderr,
        "[KFAC-%s] tile=0x%llx opt_step=%d rows=%d cols=%d norm=%.6e max=%.6e nnz=%d/%d head=",
        tag,
        (unsigned long long)this_tile_id,
        opt_step,
        dbg_rows,
        d_size,
        std::sqrt(norm2),
        max_abs,
        nnz,
        total);
    const int show = std::min(total, iter_debug_head);
    for (int i = 0; i < show; ++i) {
      fprintf(stderr, "%.6e%s", static_cast<double>(hbuf[i]), (i + 1 < show) ? " " : "");
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  };

  if (trans && (dim3 > 1)) {

    SliceInputIterator<true, T> d_in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);
    emit_iter_summary(d_in_iter, "DITER");

    IndexReaderSliceInputIterator<true, T> x_in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->updateMatrixIterator(x_in_iter, d_in_iter, m_batch_slice * dim3, trans, trans);

  } else {

    SliceInputIterator<false, T> d_in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);
    emit_iter_summary(d_in_iter, "DITER");

    IndexReaderSliceInputIterator<false, T> x_in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    if (use_kfac_indexed_d_materialize_fix) {
      const int d_total = d_size * dbg_rows;
      CudaArray<T> d_contig(this->context_, d_total);
      this->materializeUpdateInput(d_in_iter, d_contig.getData(), d_total);

      // Old path kept here for reference while we verify the D-side iterator/capture bug:
      // this->updateMatrixIterator(x_in_iter, d_in_iter, m_batch_slice * dim3, trans, trans);

      this->updateMatrixIterator(
          x_in_iter, d_contig.getDataConst(), m_batch_slice * dim3, trans, false);
    } else {
      this->updateMatrixIterator(x_in_iter, d_in_iter, m_batch_slice * dim3, trans, trans);
    }
  }
}

template <typename T>
void RPUCudaPulsed<T>::updateVector(const T *x_input, const T *d_input, int x_inc, int d_inc) {

  const T *x_input_inc1 = x_input;
  const T *d_input_inc1 = d_input;

  if (x_inc !=
      1) { // could make iterators here. But never hit anyway (because matrix is used for inc>1)
    RPU::math::copy<T>(
        this->context_, this->x_size_, x_input, x_inc, dev_up_x_vector_inc1_->getData(), 1);
    x_input_inc1 = dev_up_x_vector_inc1_->getDataConst();
  }

  if (d_inc != 1) {
    RPU::math::copy<T>(
        this->context_, this->d_size_, d_input, d_inc, dev_up_d_vector_inc1_->getData(), 1);
    d_input_inc1 = dev_up_d_vector_inc1_->getDataConst();
  }

  updateMatrixIterator(x_input_inc1, d_input_inc1, 1, false, false);
}

// template <typename T>
// template <typename XInputIteratorT, typename DInputIteratorT>
// void RPUCudaPulsed<T>::updateMatrixIterator(
//     XInputIteratorT X_input, DInputIteratorT D_input, int m_batch, bool x_trans, bool d_trans) {
//   this->last_update_m_batch_ = m_batch;

//   const auto &up = getMetaPar().up;

//   if (up_pwu_->checkForFPUpdate(&*rpucuda_device_, up)) {
//     // we take a short-cut in case that FP update is requested:
//     up_pwu_->doFPupdate(
//         X_input, D_input, this->getUpWeightsCuda(), this->getAlphaLearningRate(), m_batch, x_trans,
//         d_trans, this->getUpBeta());

//   } else {
//     T *local_dw = this->getDeltaWeights();

//     if (local_dw) {
//       // cannot do directly compute DW. Thus first copy weights and then apply
//       this->setDeltaWeights(nullptr); // reset to normal up weights
//       int sz = this->x_size_ * this->d_size_;
//       RPU::math::copy<T>(this->context_, sz, this->getUpWeightsCuda(), 1, local_dw, 1);
//     }
//     T lr = this->getAlphaLearningRate();
//     T *weights = this->getUpWeightsCuda();
//     up_pwu_->update(
//         X_input, D_input, weights, &*rpucuda_device_, up, lr, m_batch, x_trans, d_trans);

//     if (local_dw) {
//       this->setDeltaWeights(local_dw); // this might change the value of LR
//       this->getAndResetWeightUpdate(local_dw, this->getAlphaLearningRate() / lr);
//     }
//   }
// }
template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void RPUCudaPulsed<T>::updateMatrixIterator(
    XInputIteratorT X_input, DInputIteratorT D_input, int m_batch, bool x_trans, bool d_trans) {

  this->last_update_m_batch_ = m_batch;
  const auto &up = getMetaPar().up;
  cudaStream_t s = this->context_->getStream();
  auto env_on = [](const char *name) -> bool {
  const char *e = std::getenv(name);
  return (e != nullptr) && (std::atoi(e) != 0);
};

auto env_int = [](const char *name, int defval) -> int {
  const char *e = std::getenv(name);
  return (e != nullptr) ? std::atoi(e) : defval;
};

auto env_ull = [](const char *name, unsigned long long defval) -> unsigned long long {
  const char *e = std::getenv(name);
  return (e != nullptr) ? std::strtoull(e, nullptr, 0) : defval;
};

  // ---- one-time runtime KFAC init from environment ----
  if (!kfac_runtime_inited_) {
    auto kfac_on = []() -> bool {
      const char *e = std::getenv("AIHWKIT_KFAC_ON");
      return (e != nullptr) && (std::atoi(e) != 0);
    };

    auto kfac_get_int = [](const char *name, int defval) -> int {
      const char *e = std::getenv(name);
      return (e != nullptr) ? std::atoi(e) : defval;
    };

    auto kfac_get_val = [](const char *name, T defval) -> T {
      const char *e = std::getenv(name);
      return (e != nullptr) ? (T)std::atof(e) : defval;
    };

    if (kfac_on()) {
      const int block_x = kfac_get_int("AIHWKIT_KFAC_BLOCK_X", 128);
      const int block_d = kfac_get_int("AIHWKIT_KFAC_BLOCK_D", 128);
      const int freq    = kfac_get_int("AIHWKIT_KFAC_FREQ", 16);
      const T beta      = kfac_get_val("AIHWKIT_KFAC_BETA", (T)0.05);
      const T eps = kfac_get_val("AIHWKIT_KFAC_LAMBDA", (T)1e-3);
      this->setKFACConfig(true, block_x, block_d, freq, beta, eps);
    } else {
      this->setKFACConfig(false, 0, 0, 1, (T)0.0, (T)0.0);
    }
    initKFACTrickRuntime();
    initKFACTrickRuntime();
    initKFACTransferRuntime();


    kfac_runtime_inited_ = true;
  }

  // ---- First-cut KFAC path: only support the standard layout ----
  const bool use_kfac = kfac_enable_ && !x_trans && !d_trans;
  const bool use_raw_trick =
      !use_kfac && kfac_trick_enable_ && env_on("AIHWKIT_KFAC_RAW_TRICK_ON") && !x_trans &&
      !d_trans;

  if (use_kfac || use_raw_trick) {
    {
      // ScopedWallTimer tw("ensureKFACBuffers");
      ensureKFACBuffers(m_batch);
    }

    {
      // ScopedWallTimer tw("materialize X");
      // ScopedCudaTimer tg("materialize X", s);
      materializeUpdateInput(X_input, dev_kfac_x_raw_->getData(), this->x_size_ * m_batch);
    }

    {
      // ScopedWallTimer tw("materialize D");
      // ScopedCudaTimer tg("materialize D", s);
      materializeUpdateInput(D_input, dev_kfac_d_raw_->getData(), this->d_size_ * m_batch);
    }

    const T *x_raw = dev_kfac_x_raw_->getDataConst();
    const T *d_raw = dev_kfac_d_raw_->getDataConst();
    const uint64_t this_tile_id =
    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));


const bool debug_dump = env_on("AIHWKIT_KFAC_DEBUG_DUMP");
const bool debug_list = env_on("AIHWKIT_KFAC_DEBUG_LIST_TILES");
const int target_tile_id = env_int("AIHWKIT_KFAC_DEBUG_TILE_ID", -1);
const int debug_every = env_int("AIHWKIT_DEBUG_EVERY", 0);
const int opt_step = env_int("AIHWKIT_DEBUG_OPT_STEP", -1);
const int min_step = env_int("AIHWKIT_DEBUG_MIN_STEP", 0);
const bool global_debug_match =
    (debug_every <= 0) || (opt_step >= min_step && (opt_step % debug_every) == 0);

if (debug_list && kfac_step_ < 3) {
  fprintf(stderr,
          "[KFAC-TILE] tile_id=%llu step=%llu x_size=%d d_size=%d\n",
          (unsigned long long)this_tile_id,
          (unsigned long long)kfac_step_,
          this->x_size_, this->d_size_);
  fflush(stderr);
}

const bool match_tile =
    (target_tile_id < 0) || ((int)this_tile_id == target_tile_id);

const bool mat_debug = env_on("AIHWKIT_KFAC_MAT_DEBUG");
const int mat_debug_xsize = env_int("AIHWKIT_KFAC_MAT_DEBUG_XSIZE", -1);
const int mat_debug_dsize = env_int("AIHWKIT_KFAC_MAT_DEBUG_DSIZE", -1);
const int mat_debug_step = env_int("AIHWKIT_KFAC_MAT_DEBUG_STEP", -1);
const int mat_debug_head = std::max(0, env_int("AIHWKIT_KFAC_MAT_DEBUG_HEAD", 8));
const unsigned long long mat_debug_tile =
    env_ull("AIHWKIT_KFAC_MAT_DEBUG_TILE", 0ULL);
const bool mat_debug_match =
    mat_debug &&
    (mat_debug_xsize < 0 || mat_debug_xsize == this->x_size_) &&
    (mat_debug_dsize < 0 || mat_debug_dsize == this->d_size_) &&
    (mat_debug_step < 0 || mat_debug_step == opt_step) &&
    global_debug_match &&
    (mat_debug_tile == 0ULL || mat_debug_tile == (unsigned long long)this_tile_id);

auto emit_operand_summary = [&](const char *tag, const T *ptr, int rows, int cols) {
  if (!mat_debug_match) {
    return;
  }

  const int total = rows * cols;
  std::vector<T> hbuf(total);
  cuda_copy_d2h(s, ptr, hbuf.data(), total);

  double norm2 = 0.0;
  double max_abs = 0.0;
  int nnz = 0;
  for (int i = 0; i < total; ++i) {
    const double v = static_cast<double>(hbuf[i]);
    norm2 += v * v;
    max_abs = std::max(max_abs, std::abs(v));
    if (v != 0.0) {
      nnz++;
    }
  }

  fprintf(
      stderr,
      "[KFAC-%s] tile=0x%llx opt_step=%d kfac_step=%llu rows=%d cols=%d norm=%.6e max=%.6e nnz=%d/%d head=",
      tag,
      (unsigned long long)this_tile_id,
      opt_step,
      (unsigned long long)kfac_step_,
      rows,
      cols,
      std::sqrt(norm2),
      max_abs,
      nnz,
      total);

  const int show = std::min(total, mat_debug_head);
  for (int i = 0; i < show; ++i) {
    fprintf(stderr, "%.6e%s", static_cast<double>(hbuf[i]), (i + 1 < show) ? " " : "");
  }
  fprintf(stderr, "\n");
  fflush(stderr);
};

auto dump_matrix = [&](const char *fname, const T *ptr, int rows, int cols) {
  std::vector<T> hbuf(rows * cols);
  cuda_copy_d2h(s, ptr, hbuf.data(), rows * cols);

  std::ofstream ofs(fname);
  ofs << "# rows=" << rows
      << " cols=" << cols
      << " tile_id=" << (unsigned long long)this_tile_id
      << " step=" << (unsigned long long)kfac_step_
      << " x_size=" << this->x_size_
      << " d_size=" << this->d_size_
      << "\n";
  ofs << std::setprecision(18);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      ofs << static_cast<double>(hbuf[r * cols + c]);
      if (c + 1 < cols) {
        ofs << " ";
      }
    }
    ofs << "\n";
  }
  ofs.close();
};

if (debug_dump && kfac_step_ == 0 && match_tile) {
  dump_matrix("/tmp/cpp_x_raw_step0.txt", x_raw, m_batch, this->x_size_);
  dump_matrix("/tmp/cpp_d_raw_step0.txt", d_raw, m_batch, this->d_size_);

  fprintf(stderr,
          "[KFAC-DUMP] dumped raw operands for tile_id=%llu step=%llu x_size=%d d_size=%d\n",
          (unsigned long long)this_tile_id,
          (unsigned long long)kfac_step_,
          this->x_size_, this->d_size_);
  fflush(stderr);
}

    emit_operand_summary("XRAW", x_raw, m_batch, this->x_size_);
    emit_operand_summary("DRAW", d_raw, m_batch, this->d_size_);
//     auto kfac_debug_dump = []() -> bool {
//   const char *e = std::getenv("AIHWKIT_KFAC_DEBUG_DUMP");
//   return (e != nullptr) && (std::atoi(e) != 0);
// };

// auto kfac_debug_xsize = []() -> int {
//   const char *e = std::getenv("AIHWKIT_KFAC_DEBUG_XSIZE");
//   return (e != nullptr) ? std::atoi(e) : -1;
// };

// auto kfac_debug_dsize = []() -> int {
//   const char *e = std::getenv("AIHWKIT_KFAC_DEBUG_DSIZE");
//   return (e != nullptr) ? std::atoi(e) : -1;
// };

// // 只在 step 0 dump 一次；可用 x_size / d_size 过滤目标层
// if (kfac_debug_dump() && kfac_step_ == 0) {
//   const int dbg_x = kfac_debug_xsize();
//   const int dbg_d = kfac_debug_dsize();
//   const bool match_x = (dbg_x < 0) || (dbg_x == this->x_size_);
//   const bool match_d = (dbg_d < 0) || (dbg_d == this->d_size_);

//   if (match_x && match_d) {
//     const int nd = this->d_size_ * m_batch;
//     std::vector<T> hd_raw(nd);
//     cuda_copy_d2h(s, d_raw, hd_raw.data(), nd);

//     // 导出成 [m_batch, d_size_] 的文本矩阵
//     const char *fname = "/tmp/cpp_d_raw_step0.txt";
//     std::ofstream ofs(fname);
//     ofs << "# m_batch=" << m_batch
//         << " d_size=" << this->d_size_
//         << " x_size=" << this->x_size_
//         << " step=" << (unsigned long long)kfac_step_ << "\n";
//     ofs << std::setprecision(18);

//     for (int mb = 0; mb < m_batch; ++mb) {
//       for (int j = 0; j < this->d_size_; ++j) {
//         ofs << static_cast<double>(hd_raw[mb * this->d_size_ + j]);
//         if (j + 1 < this->d_size_) {
//           ofs << " ";
//         }
//       }
//       ofs << "\n";
//     }
//     ofs.close();

//     double total_norm2 = 0.0;
//     for (int i = 0; i < nd; ++i) {
//       double v = static_cast<double>(hd_raw[i]);
//       total_norm2 += v * v;
//     }

//     fprintf(stderr,
//             "[C++ D_RAW DUMP] wrote %s | step=%llu | m_batch=%d | d_size=%d | x_size=%d | norm=%.6e\n",
//             fname,
//             (unsigned long long)kfac_step_,
//             m_batch, this->d_size_, this->x_size_,
//             std::sqrt(total_norm2));
//     fflush(stderr);
//   }

    const T *x_pre = nullptr;
    const T *d_pre = nullptr;
    const uint64_t step_id = kfac_step_;
    const bool do_gnorm_rescale = use_kfac && env_on("AIHWKIT_KFAC_GNORM_RESCALE");

    if (use_kfac) {
      {
        // ScopedWallTimer tw("updateKFACStats");
        // ScopedCudaTimer tg("updateKFACStats", s);
        updateKFACStats(x_raw, d_raw, m_batch);
      }

      if ((kfac_step_ % (uint64_t)kfac_update_freq_) == 0 || !kfac_ready_) {
        // ScopedWallTimer tw("refreshKFACFactors");
        refreshKFACFactors();
      }

      {
        // ScopedWallTimer tw("applyBlockKFACX");
        // ScopedCudaTimer tg("applyBlockKFACX", s);
        applyBlockKFACX(x_raw, dev_kfac_x_pre_->getData(), m_batch);
      }

      {
        // ScopedWallTimer tw("applyBlockKFACD");
        // ScopedCudaTimer tg("applyBlockKFACD", s);
        applyBlockKFACD(d_raw, dev_kfac_d_pre_->getData(), m_batch);
      }

      x_pre = dev_kfac_x_pre_->getDataConst();
      d_pre = dev_kfac_d_pre_->getDataConst();
    } else {
      // Raw-trace trick path: bypass block-KFAC and feed the raw operands directly
      // into build/residual/commit.
      x_pre = x_raw;
      d_pre = d_raw;
    }

if (do_gnorm_rescale) {
  const int B  = m_batch;
  const int Dx = this->x_size_;
  const int Dd = this->d_size_;
  const int nx = B * Dx;
  const int nd = B * Dd;

  // zero 2 doubles
  cudaError_t err = cudaMemsetAsync(
      dev_kfac_gnorm_buf_->getData(), 0, 2 * sizeof(float), s);
  if (err != cudaSuccess) {
    RPU_FATAL("cudaMemsetAsync failed for KFAC G-norm rescale.");
  }

  float *buf = dev_kfac_gnorm_buf_->getData();

  dim3 block(256);
  dim3 grid(std::min((Dd * Dx + block.x - 1) / block.x, 4096u));

  // buf[0] = ||G_raw||_F^2
  kernelOuterFroSq<T><<<grid, block, 0, s>>>(x_raw, d_raw, B, Dx, Dd, buf + 0);

  // buf[1] = ||G_pre||_F^2
  kernelOuterFroSq<T><<<grid, block, 0, s>>>(
      dev_kfac_x_pre_->getDataConst(),
      dev_kfac_d_pre_->getDataConst(),
      B, Dx, Dd, buf + 1);

  float hbuf[2] = {0.0f, 0.0f};
  cuda_copy_d2h(s, buf, hbuf, 2);

  const double graw = std::sqrt(std::max((double)hbuf[0], 0.0));
  const double gpre = std::sqrt(std::max((double)hbuf[1], 0.0));

  // alpha rescales G_pre: G_tilde = alpha * G_pre
  double alpha = graw / (gpre + 1e-24);

  // clip for stability
  alpha = std::max(0.1, std::min(alpha, 10.0));

  // scale x_pre and d_pre by sqrt(alpha), so G scales by alpha
  const double r = std::sqrt(alpha);

  dim3 grid_x(std::min((nx + block.x - 1) / block.x, 4096u));
  dim3 grid_d(std::min((nd + block.x - 1) / block.x, 4096u));

  kernelScaleArray<T><<<grid_x, block, 0, s>>>(dev_kfac_x_pre_->getData(), nx, (T)r);
  kernelScaleArray<T><<<grid_d, block, 0, s>>>(dev_kfac_d_pre_->getData(), nd, (T)r);

  // if (kfac_step_ % 200 == 0) {
  //   fprintf(stderr,
  //           "[KFAC-GRESCALE] step=%llu alpha=%.6e r=%.6e "
  //           "||G_raw||=%.6e ||G_pre||=%.6e\n",
  //           (unsigned long long)kfac_step_,
  //           alpha, r, graw, gpre);
  //   fflush(stderr);
  // }
}

    emit_operand_summary("XPRE", x_pre, m_batch, this->x_size_);
    emit_operand_summary("DPRE", d_pre, m_batch, this->d_size_);

    if (debug_dump && kfac_step_ == 0 && match_tile) {
  dump_matrix("/tmp/cpp_x_pre_step0.txt", x_pre, m_batch, this->x_size_);
  dump_matrix("/tmp/cpp_d_pre_step0.txt", d_pre, m_batch, this->d_size_);

  fprintf(stderr,
          "[KFAC-DUMP] dumped preconditioned operands for tile_id=%llu step=%llu\n",
          (unsigned long long)this_tile_id,
          (unsigned long long)kfac_step_);
  fflush(stderr);
}

    const bool dw_diag_on = env_on("AIHWKIT_KFAC_DW_DIAG");
    const int dw_diag_freq = std::max(1, env_int("AIHWKIT_KFAC_DW_DIAG_FREQ", 1));
    const int dw_diag_xsize = env_int("AIHWKIT_KFAC_DW_DIAG_XSIZE", -1);
    const int dw_diag_dsize = env_int("AIHWKIT_KFAC_DW_DIAG_DSIZE", -1);
    const int dw_diag_step = env_int("AIHWKIT_KFAC_DW_DIAG_STEP", -1);
    const bool dw_diag_dump = env_on("AIHWKIT_KFAC_DW_DIAG_DUMP");
    const bool dw_diag_svd = env_on("AIHWKIT_KFAC_DW_DIAG_SVD");
    const int dw_diag_svd_max_rank =
        std::max(1, env_int("AIHWKIT_KFAC_DW_DIAG_SVD_MAX_RANK", 512));
    const int dw_diag_svd_topk =
        std::max(1, env_int("AIHWKIT_KFAC_DW_DIAG_SVD_TOPK", 8));
    const int weight_size = this->x_size_ * this->d_size_;

    const bool dw_diag_match =
        dw_diag_on &&
        (dw_diag_xsize < 0 || dw_diag_xsize == this->x_size_) &&
        (dw_diag_dsize < 0 || dw_diag_dsize == this->d_size_) &&
        (dw_diag_step < 0 || dw_diag_step == opt_step) &&
        global_debug_match &&
        ((step_id % (uint64_t)dw_diag_freq) == 0);

    std::unique_ptr<CudaArray<T>> dw_diag_prev = nullptr;
    if (dw_diag_match) {
      dw_diag_prev = RPU::make_unique<CudaArray<T>>(this->context_, weight_size);
      RPU::math::copy<T>(
          this->context_, weight_size, this->getUpWeightsCuda(), 1, dw_diag_prev->getData(), 1);
    }

    auto emit_dw_build_diag = [&](const char *mode, T lr) {
      if (!dw_diag_match) {
        return;
      }

      const int nx = this->x_size_ * m_batch;
      const int nd = this->d_size_ * m_batch;

      std::vector<T> hx_pre(nx), hd_pre(nd), h_dw_build(weight_size);
      cuda_copy_d2h(s, x_pre, hx_pre.data(), nx);
      cuda_copy_d2h(s, d_pre, hd_pre.data(), nd);
      cuda_copy_d2h(s, dev_kfac_dw_acc_->getDataConst(), h_dw_build.data(), weight_size);

      std::vector<double> h_dw_ideal;
      host_build_outer_update_col_major(
          hx_pre.data(), hd_pre.data(), m_batch, this->x_size_, this->d_size_, (double)lr,
          h_dw_ideal);

      DeltaWCompareMetrics dw_metrics =
          host_compare_weight_updates(h_dw_build, h_dw_ideal);

      fprintf(
          stderr,
          "[KFAC-BUILD] opt_step=%d kfac_step=%llu tile_id=%llu mode=%s m_batch=%d x_size=%d d_size=%d "
          "||build||=%.6e ||ideal||=%.6e cos=%.6f angle_deg=%.4f proj_scale=%.6e "
          "mag_ratio_build=%.6e rel_err_build=%.6e max_abs_err=%.6e mean_abs_err=%.6e\n",
          opt_step,
          (unsigned long long)step_id,
          (unsigned long long)this_tile_id,
          mode,
          m_batch,
          this->x_size_,
          this->d_size_,
          dw_metrics.actual_norm,
          dw_metrics.ideal_norm,
          dw_metrics.cosine,
          dw_metrics.angle_deg,
          dw_metrics.proj_scale,
          dw_metrics.mag_ratio,
          dw_metrics.rel_err,
          dw_metrics.max_abs_err,
          dw_metrics.mean_abs_err);
      fflush(stderr);
    };

    auto emit_dw_diag = [&](const char *mode, T lr) {
      if (!dw_diag_prev) {
        return;
      }

      const int nx = this->x_size_ * m_batch;
      const int nd = this->d_size_ * m_batch;

      std::vector<T> hx_pre(nx), hd_pre(nd), h_w_before(weight_size), h_w_after(weight_size);
      cuda_copy_d2h(s, x_pre, hx_pre.data(), nx);
      cuda_copy_d2h(s, d_pre, hd_pre.data(), nd);
      cuda_copy_d2h(s, dw_diag_prev->getDataConst(), h_w_before.data(), weight_size);
      cuda_copy_d2h(s, this->getUpWeightsCuda(), h_w_after.data(), weight_size);

      std::vector<T> h_dw_actual(weight_size);
      for (int i = 0; i < weight_size; ++i) {
        h_dw_actual[i] = h_w_after[i] - h_w_before[i];
      }

      std::vector<double> h_dw_ideal;
      host_build_outer_update_col_major(
          hx_pre.data(), hd_pre.data(), m_batch, this->x_size_, this->d_size_, (double)lr,
          h_dw_ideal);

      DeltaWCompareMetrics dw_metrics =
          host_compare_weight_updates(h_dw_actual, h_dw_ideal);

      fprintf(
          stderr,
          "[KFAC-DW] opt_step=%d kfac_step=%llu tile_id=%llu mode=%s m_batch=%d x_size=%d d_size=%d "
          "||actual||=%.6e ||ideal||=%.6e cos=%.6f angle_deg=%.4f proj_scale=%.6e "
          "mag_ratio=%.6e rel_err=%.6e orth_rel=%.6e max_abs_err=%.6e mean_abs_err=%.6e\n",
          opt_step,
          (unsigned long long)step_id,
          (unsigned long long)this_tile_id,
          mode,
          m_batch,
          this->x_size_,
          this->d_size_,
          dw_metrics.actual_norm,
          dw_metrics.ideal_norm,
          dw_metrics.cosine,
          dw_metrics.angle_deg,
          dw_metrics.proj_scale,
          dw_metrics.mag_ratio,
          dw_metrics.rel_err,
          dw_metrics.orth_rel,
          dw_metrics.max_abs_err,
          dw_metrics.mean_abs_err);
      fflush(stderr);

      if (dw_diag_svd) {
        const int sv_rank = std::min(this->d_size_, this->x_size_);
        if (sv_rank <= dw_diag_svd_max_rank) {
          std::vector<double> h_dw_actual_double(weight_size, 0.0);
          for (int i = 0; i < weight_size; ++i) {
            h_dw_actual_double[i] = static_cast<double>(h_dw_actual[i]);
          }

          std::vector<double> sv_actual, sv_ideal;
          const bool actual_ok = host_compute_singular_values_col_major(
              h_dw_actual_double, this->d_size_, this->x_size_, sv_actual);
          const bool ideal_ok = host_compute_singular_values_col_major(
              h_dw_ideal, this->d_size_, this->x_size_, sv_ideal);

          if (actual_ok && ideal_ok) {
            double sv_err_sq = 0.0;
            double sv_ref_sq = 0.0;
            for (size_t i = 0; i < sv_ideal.size(); ++i) {
              const double diff = sv_actual[i] - sv_ideal[i];
              sv_err_sq += diff * diff;
              sv_ref_sq += sv_ideal[i] * sv_ideal[i];
            }

            fprintf(
                stderr,
                "[KFAC-DW-SVD] step=%llu tile_id=%llu mode=%s rel_sv_err=%.6e top_actual=%s "
                "top_ideal=%s\n",
                (unsigned long long)step_id,
                (unsigned long long)this_tile_id,
                mode,
                std::sqrt(std::max(sv_err_sq, 0.0)) /
                    (std::sqrt(std::max(sv_ref_sq, 0.0)) + 1e-24),
                host_format_top_values(sv_actual, dw_diag_svd_topk).c_str(),
                host_format_top_values(sv_ideal, dw_diag_svd_topk).c_str());
            fflush(stderr);
          } else {
            fprintf(
                stderr,
                "[KFAC-DW-SVD] step=%llu tile_id=%llu mode=%s skipped actual_ok=%d "
                "ideal_ok=%d\n",
                (unsigned long long)step_id,
                (unsigned long long)this_tile_id,
                mode,
                (int)actual_ok,
                (int)ideal_ok);
            fflush(stderr);
          }
        } else {
          fprintf(
              stderr,
              "[KFAC-DW-SVD] step=%llu tile_id=%llu mode=%s skipped min_dim=%d exceeds "
              "limit=%d\n",
              (unsigned long long)step_id,
              (unsigned long long)this_tile_id,
              mode,
              sv_rank,
              dw_diag_svd_max_rank);
          fflush(stderr);
        }
      }

      if (dw_diag_dump) {
        char fname_actual[256];
        char fname_ideal[256];

        std::snprintf(
            fname_actual, sizeof(fname_actual),
            "/tmp/kfac_dw_actual_tile%llu_step%llu_d%d_x%d.txt",
            (unsigned long long)this_tile_id, (unsigned long long)step_id, this->d_size_,
            this->x_size_);
        std::snprintf(
            fname_ideal, sizeof(fname_ideal),
            "/tmp/kfac_dw_ideal_tile%llu_step%llu_d%d_x%d.txt",
            (unsigned long long)this_tile_id, (unsigned long long)step_id, this->d_size_,
            this->x_size_);

        dump_host_matrix_col_major(
            fname_actual, h_dw_actual, this->d_size_, this->x_size_, "dw_actual", this_tile_id,
            step_id, this->x_size_, this->d_size_);
        dump_host_matrix_col_major(
            fname_ideal, h_dw_ideal, this->d_size_, this->x_size_, "dw_ideal", this_tile_id,
            step_id, this->x_size_, this->d_size_);

        fprintf(
            stderr,
            "[KFAC-DW-DUMP] step=%llu tile_id=%llu mode=%s actual=%s ideal=%s\n",
            (unsigned long long)step_id,
            (unsigned long long)this_tile_id,
            mode,
            fname_actual,
            fname_ideal);
        fflush(stderr);
      }
    };

    kfac_step_++;

    const int trick_start_step = env_int("AIHWKIT_KFAC_TRICK_START_STEP", 0);
    const bool trick_active_now =
        kfac_trick_enable_ && (opt_step < 0 || opt_step >= trick_start_step);

    if (trick_active_now) {
      T *local_dw = this->getDeltaWeights();

      if (local_dw) {
        this->setDeltaWeights(nullptr);
        RPU::math::copy<T>(
            this->context_, weight_size, this->getUpWeightsCuda(), 1, local_dw, 1);
      }

      T lr = this->getAlphaLearningRate();
      ensureKFACTrickBuffers(m_batch);
      ensureKFACTransferBuffers(kfac_transfer_rows_per_step_);

      buildKFACDeltaWithAnalogGradTricks(
          x_pre, d_pre, m_batch, lr, dev_kfac_dw_acc_->getData());

      emit_dw_build_diag("trick_build", lr);
      accumulateKFACResidual(dev_kfac_dw_acc_->getDataConst());
      submitKFACResidualTransfer(this->getUpWeightsCuda(), step_id, this_tile_id, dw_diag_match);

      emit_dw_diag("trick_transfer", lr);

      if (local_dw) {
        this->setDeltaWeights(local_dw);
        this->getAndResetWeightUpdate(local_dw, this->getAlphaLearningRate() / lr);
      }
      return;
    }


    const char *force_fp_env = std::getenv("AIHWKIT_KFAC_FORCE_FP");
    const bool force_fp =
        (force_fp_env != nullptr) && (std::atoi(force_fp_env) != 0);
    if (force_fp || up_pwu_->checkForFPUpdate(&*rpucuda_device_, up)) {
      const T lr = this->getAlphaLearningRate();
      up_pwu_->doFPupdate(
          x_pre, d_pre, this->getUpWeightsCuda(), lr,
          m_batch, false, false, this->getUpBeta());
      emit_dw_diag("fp", lr);
    } else {
      T *local_dw = this->getDeltaWeights();

      if (local_dw) {
        this->setDeltaWeights(nullptr);
        RPU::math::copy<T>(
            this->context_, weight_size, this->getUpWeightsCuda(), 1, local_dw, 1);
      }

      T lr = this->getAlphaLearningRate();
      T *weights = this->getUpWeightsCuda();
      up_pwu_->update(x_pre, d_pre, weights, &*rpucuda_device_, up, lr, m_batch, false, false);

      emit_dw_diag("pulsed", lr);

      if (local_dw) {
        this->setDeltaWeights(local_dw);
        this->getAndResetWeightUpdate(local_dw, this->getAlphaLearningRate() / lr);
      }
    }
return;
    // if (up_pwu_->checkForFPUpdate(&*rpucuda_device_, up)) {
    //   // ScopedWallTimer tw("doFPupdate");
    //   // ScopedCudaTimer tg("doFPupdate", s);
    //   up_pwu_->doFPupdate(
    //       x_pre, d_pre, this->getUpWeightsCuda(), this->getAlphaLearningRate(),
    //       m_batch, false, false, this->getUpBeta());
    // } else {
    //   // ScopedWallTimer tw("pulsed update");
    //   // ScopedCudaTimer tg("pulsed update", s);

    //   T *local_dw = this->getDeltaWeights();
    //   if (local_dw) {
    //     this->setDeltaWeights(nullptr);
    //     int sz = this->x_size_ * this->d_size_;
    //     RPU::math::copy<T>(this->context_, sz, this->getUpWeightsCuda(), 1, local_dw, 1);
    //   }

    //   T lr = this->getAlphaLearningRate();
    //   T *weights = this->getUpWeightsCuda();
    //   up_pwu_->update(x_pre, d_pre, weights, &*rpucuda_device_, up, lr, m_batch, false, false);

    //   if (local_dw) {
    //     this->setDeltaWeights(local_dw);
    //     this->getAndResetWeightUpdate(local_dw, this->getAlphaLearningRate() / lr);
    //   }
    // }
    // return;
  }

  // ---- Fallback: original behavior ----
  if (up_pwu_->checkForFPUpdate(&*rpucuda_device_, up)) {
    up_pwu_->doFPupdate(
        X_input, D_input, this->getUpWeightsCuda(), this->getAlphaLearningRate(), m_batch, x_trans,
        d_trans, this->getUpBeta());

  } else {
    T *local_dw = this->getDeltaWeights();

    if (local_dw) {
      this->setDeltaWeights(nullptr);
      int sz = this->x_size_ * this->d_size_;
      RPU::math::copy<T>(this->context_, sz, this->getUpWeightsCuda(), 1, local_dw, 1);
    }

    T lr = this->getAlphaLearningRate();
    T *weights = this->getUpWeightsCuda();
    up_pwu_->update(
        X_input, D_input, weights, &*rpucuda_device_, up, lr, m_batch, x_trans, d_trans);

    if (local_dw) {
      this->setDeltaWeights(local_dw);
      this->getAndResetWeightUpdate(local_dw, this->getAlphaLearningRate() / lr);
    }
  }
}
template class RPUCudaPulsed<float>;
#ifdef RPU_USE_DOUBLE
template class RPUCudaPulsed<double>;
#endif
#ifdef RPU_USE_FP16
template class RPUCudaPulsed<half_t>;
#endif

#undef CHECK_RPU_DEVICE_INIT

} // namespace RPU
