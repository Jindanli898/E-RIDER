/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "forward_backward_pass.h"
#include "rpucuda_pulsed.h"
#include <iostream>
#include <cstdlib>   // getenv, atoi, atof
#include <cstdio>    // fprintf, fflush
#include <chrono>
#include <memory>
#include <typeinfo>
#include "rpucuda_dynamic_transfer_device.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#define CHECK_RPU_DEVICE_INIT                                                                      \
  if (rpucuda_device_ == nullptr || rpu_device_ == nullptr) {                                      \
    RPU_FATAL("First populate rpu device (call Populate_Parameter())!");                           \
  }
template <typename T>
__global__ void kernelBlockMatVec(
    const T *in,          // [m_batch, full_dim]
    T *out,               // [m_batch, full_dim]
    const T *F,           // [bs, bs]
    int full_dim,
    int start,
    int bs,
    int m_batch) {

  int mb = blockIdx.y;   // sample index
  int r  = blockIdx.x * blockDim.x + threadIdx.x;  // row inside block

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
  up_pwu_ = RPU::make_unique<PulsedWeightUpdater<T>>(c, x_size, d_size);

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
    // Forward solve: L y = e_col
    for (int i = 0; i < n; ++i) {
      double rhs = (i == col) ? 1.0 : 0.0;
      for (int k = 0; k < i; ++k) {
        rhs -= L[i * n + k] * y[k];
      }
      y[i] = rhs / L[i * n + i];
    }

    // Backward solve: L^T x = y
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

} // namespace

template <typename T>
void RPUCudaPulsed<T>::setKFACConfig(
    bool enable,
    int block_x,
    int block_d,
    int update_freq,
    T beta,
    T lambda) {

  kfac_enable_ = enable;
  kfac_block_x_ = std::max(0, block_x);
  kfac_block_d_ = std::max(0, block_d);
  kfac_update_freq_ = std::max(1, update_freq);
  kfac_beta_ = beta;
  kfac_lambda_ = lambda;
  kfac_step_ = 0;
  kfac_ready_ = false;

  if (kfac_enable_) {
    initKFACBlocks();
    zeroKFACStats();
    refreshKFACFactors();
    kfac_ready_ = true;
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

  kfac_buffer_mbatch_ = m_batch;
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
    std::vector<T> S, F;
    host_eye(S, bs, (T)1);
    host_eye(F, bs, (T)(1.0 / (1.0 + static_cast<double>(kfac_lambda_))));
    cuda_copy_h2d(s, S.data(), dev_kfac_Sx_blocks_[bi]->getData(), bs * bs);
    cuda_copy_h2d(s, F.data(), dev_kfac_Fx_blocks_[bi]->getData(), bs * bs);
  }

  for (size_t bj = 0; bj < dev_kfac_Sd_blocks_.size(); ++bj) {
    int bs = kfac_d_block_sizes_[bj];
    std::vector<T> S, F;
    host_eye(S, bs, (T)1);
    host_eye(F, bs, (T)(1.0 / (1.0 + static_cast<double>(kfac_lambda_))));
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

  std::vector<T> hx(this->x_size_ * m_batch);
  std::vector<T> hd(this->d_size_ * m_batch);
  cuda_copy_d2h(s, x_ptr, hx.data(), hx.size());
  cuda_copy_d2h(s, d_ptr, hd.data(), hd.size());

  // ---- X-side blocks ----
  for (size_t bi = 0; bi < kfac_x_block_sizes_.size(); ++bi) {
    const int start = kfac_x_block_starts_[bi];
    const int bs = kfac_x_block_sizes_[bi];

    std::vector<T> S(bs * bs), C(bs * bs, (T)0);
    cuda_copy_d2h(s, dev_kfac_Sx_blocks_[bi]->getDataConst(), S.data(), S.size());

    for (int mb = 0; mb < m_batch; ++mb) {
      const T *xb = &hx[mb * this->x_size_ + start];
      for (int r = 0; r < bs; ++r) {
        for (int c = 0; c < bs; ++c) {
          C[r * bs + c] += xb[r] * xb[c];
        }
      }
    }

    T inv_mb = (T)(1.0 / static_cast<double>(m_batch));
    for (int i = 0; i < bs * bs; ++i) {
      C[i] *= inv_mb;
      S[i] = ((T)1 - kfac_beta_) * S[i] + kfac_beta_ * C[i];
    }

    cuda_copy_h2d(s, S.data(), dev_kfac_Sx_blocks_[bi]->getData(), S.size());
  }

  // ---- D-side blocks ----
  for (size_t bj = 0; bj < kfac_d_block_sizes_.size(); ++bj) {
    const int start = kfac_d_block_starts_[bj];
    const int bs = kfac_d_block_sizes_[bj];

    std::vector<T> S(bs * bs), C(bs * bs, (T)0);
    cuda_copy_d2h(s, dev_kfac_Sd_blocks_[bj]->getDataConst(), S.data(), S.size());

    for (int mb = 0; mb < m_batch; ++mb) {
      const T *db = &hd[mb * this->d_size_ + start];
      for (int r = 0; r < bs; ++r) {
        for (int c = 0; c < bs; ++c) {
          C[r * bs + c] += db[r] * db[c];
        }
      }
    }

    T inv_mb = (T)(1.0 / static_cast<double>(m_batch));
    for (int i = 0; i < bs * bs; ++i) {
      C[i] *= inv_mb;
      S[i] = ((T)1 - kfac_beta_) * S[i] + kfac_beta_ * C[i];
    }

    cuda_copy_h2d(s, S.data(), dev_kfac_Sd_blocks_[bj]->getData(), S.size());
  }
}

template <typename T>
void RPUCudaPulsed<T>::refreshKFACFactors() {
  if (!kfac_enable_) {
    return;
  }

  cudaStream_t s = this->context_->getStream();

  for (size_t bi = 0; bi < kfac_x_block_sizes_.size(); ++bi) {
    const int bs = kfac_x_block_sizes_[bi];
    std::vector<T> S(bs * bs), F;
    cuda_copy_d2h(s, dev_kfac_Sx_blocks_[bi]->getDataConst(), S.data(), S.size());

    bool ok = host_invert_spd_block(S, bs, kfac_lambda_, F);
    if (!ok) {
      RPU_FATAL("KFAC X-block inversion failed.");
    }
    cuda_copy_h2d(s, F.data(), dev_kfac_Fx_blocks_[bi]->getData(), F.size());
  }

  for (size_t bj = 0; bj < kfac_d_block_sizes_.size(); ++bj) {
    const int bs = kfac_d_block_sizes_[bj];
    std::vector<T> S(bs * bs), F;
    cuda_copy_d2h(s, dev_kfac_Sd_blocks_[bj]->getDataConst(), S.data(), S.size());

    bool ok = host_invert_spd_block(S, bs, kfac_lambda_, F);
    if (!ok) {
      RPU_FATAL("KFAC D-block inversion failed.");
    }
    cuda_copy_h2d(s, F.data(), dev_kfac_Fd_blocks_[bj]->getData(), F.size());
  }

  kfac_ready_ = true;
}

template <typename T>
void RPUCudaPulsed<T>::applyBlockKFACX(const T *x_in, T *x_out, int m_batch) {
  // 先清零输出
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

  const int *indices = this->getMatrixIndices();
  int x_size = this->getXSize();
  int d_size = this->getDSize();

  if (trans && (dim3 > 1)) {

    SliceInputIterator<true, T> d_in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    IndexReaderSliceInputIterator<true, T> x_in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->updateMatrixIterator(x_in_iter, d_in_iter, m_batch_slice * dim3, trans, trans);

  } else {

    SliceInputIterator<false, T> d_in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    IndexReaderSliceInputIterator<false, T> x_in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->updateMatrixIterator(x_in_iter, d_in_iter, m_batch_slice * dim3, trans, trans);
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
      const T lambda    = kfac_get_val("AIHWKIT_KFAC_LAMBDA", (T)1e-3);

      this->setKFACConfig(true, block_x, block_d, freq, beta, lambda);

      // fprintf(stderr,
      //         "[KFAC][RUNTIME INIT] enabled: block_x=%d block_d=%d freq=%d beta=%g lambda=%g\n",
      //         block_x, block_d, freq, (double)beta, (double)lambda);
      // fflush(stderr);
    } else {
      this->setKFACConfig(false, 0, 0, 1, (T)0.0, (T)0.0);
      // fprintf(stderr, "[KFAC][RUNTIME INIT] disabled\n");
      // fflush(stderr);
    }

    kfac_runtime_inited_ = true;
  }
  // ---- First-cut KFAC path: only support the standard layout ----
  const bool use_kfac = kfac_enable_ && !x_trans && !d_trans;

  if (use_kfac) {
    // std::cout << "[KFAC] using KFAC path, m_batch=" << m_batch << std::endl;
    ensureKFACBuffers(m_batch);

    // 1) Materialize generic iterators into dense contiguous buffers.
    materializeUpdateInput(X_input, dev_kfac_x_raw_->getData(), this->x_size_ * m_batch);
    materializeUpdateInput(D_input, dev_kfac_d_raw_->getData(), this->d_size_ * m_batch);

    const T *x_raw = dev_kfac_x_raw_->getDataConst();
    const T *d_raw = dev_kfac_d_raw_->getDataConst();

    // 2) Update raw KFAC statistics from the ORIGINAL x,d.
    updateKFACStats(x_raw, d_raw, m_batch);

    // 3) Refresh inverse blocks every T steps.
    if ((kfac_step_ % (uint64_t)kfac_update_freq_) == 0 || !kfac_ready_) {
      refreshKFACFactors();
    }

    // 4) Apply block preconditioners:
    //      x_hat = K^{-1} x
    //      d_hat = C^{-1} d
    applyBlockKFACX(x_raw, dev_kfac_x_pre_->getData(), m_batch);
    applyBlockKFACD(d_raw, dev_kfac_d_pre_->getData(), m_batch);

    const T *x_pre = dev_kfac_x_pre_->getDataConst();
    const T *d_pre = dev_kfac_d_pre_->getDataConst();

    kfac_step_++;

    // 5) Reuse the original AIHWKIT update primitive with preconditioned inputs.
    if (up_pwu_->checkForFPUpdate(&*rpucuda_device_, up)) {
      up_pwu_->doFPupdate(
          x_pre, d_pre, this->getUpWeightsCuda(), this->getAlphaLearningRate(), m_batch,
          false, false, this->getUpBeta());

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
          x_pre, d_pre, weights, &*rpucuda_device_, up, lr, m_batch, false, false);

      if (local_dw) {
        this->setDeltaWeights(local_dw);
        this->getAndResetWeightUpdate(local_dw, this->getAlphaLearningRate() / lr);
      }
    }
    return;
  }

  // ---- Fallback: original behavior ----
  if (up_pwu_->checkForFPUpdate(&*rpucuda_device_, up)) {
    up_pwu_->doFPupdate(
        X_input, D_input, this->getUpWeightsCuda(), this->getAlphaLearningRate(), m_batch, x_trans,
        d_trans, this->getUpBeta());

  } else { 
    // std::cout << "[KFAC] fallback path, enable=" << kfac_enable_
    //         << " x_trans=" << x_trans
    //         << " d_trans=" << d_trans << std::endl;
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
