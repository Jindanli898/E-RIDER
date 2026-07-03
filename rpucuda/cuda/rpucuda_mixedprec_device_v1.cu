
/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "cuda_fp16_util.h"
#include "cuda_math_util.h"
#include "io_iterator.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_mixedprec_device.h"
#include <memory>
#include <cmath>
#include <type_traits>
#include <vector>
#include <iostream>
#include <cmath>

#ifdef RPU_USE_MKL
  #include <mkl_lapacke.h>
#else
  #include <lapacke.h>
#endif

namespace RPU {

/******************************************************************************************/
/* MixedPrecRPUDeviceCuda

   CUDA implementation of MixedPrecRPUDevice

*/
template <typename T>
MixedPrecRPUDeviceCuda<T>::MixedPrecRPUDeviceCuda(CudaContextPtr c, int x_size, int d_size)
    : MixedPrecRPUDeviceBaseCuda<T>(c, x_size, d_size){};

template <typename T>
MixedPrecRPUDeviceCuda<T>::MixedPrecRPUDeviceCuda(
    CudaContextPtr c, const MixedPrecRPUDevice<T> &rpu_device)
    : MixedPrecRPUDeviceCuda<T>(c, rpu_device.getXSize(), rpu_device.getDSize()) {
  populateFrom(rpu_device);
};

template <typename T> void MixedPrecRPUDeviceCuda<T>::allocateContainers() {
  this->context_->synchronizeDevice();
  dev_chi_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
}
template <typename T>
void MixedPrecRPUDeviceCuda<T>::allocateMuonContainers_() {
  if (dev_m_ == nullptr) {
    dev_m_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
    dev_m_->setConst((T)0);
  }
  if (dev_ns_X_ == nullptr) dev_ns_X_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
  if (dev_ns_B_ == nullptr) dev_ns_B_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
  if (dev_ns_C_ == nullptr) dev_ns_C_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
  if (dev_ns_O_ == nullptr) dev_ns_O_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);

  int dd = this->d_size_ * this->d_size_;
  if (dev_ns_A_ == nullptr)  dev_ns_A_  = RPU::make_unique<CudaArray<T>>(this->context_, dd);
  if (dev_ns_G2_ == nullptr) dev_ns_G2_ = RPU::make_unique<CudaArray<T>>(this->context_, dd);
}



// copy
template <typename T>
MixedPrecRPUDeviceCuda<T>::MixedPrecRPUDeviceCuda(const MixedPrecRPUDeviceCuda<T> &other)
    : MixedPrecRPUDeviceBaseCuda<T>(other) {

  allocateContainers();
  dev_chi_->assign(*other.dev_chi_);
  if (other.dev_m_ != nullptr) {
  allocateMuonContainers_();
  dev_m_->assign(*other.dev_m_);
}
  this->context_->synchronize();
};

template <typename T>
MixedPrecRPUDeviceCuda<T> &
MixedPrecRPUDeviceCuda<T>::operator=(const MixedPrecRPUDeviceCuda<T> &other) {
  MixedPrecRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
};

template <typename T>
MixedPrecRPUDeviceCuda<T>::MixedPrecRPUDeviceCuda(MixedPrecRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

template <typename T>
MixedPrecRPUDeviceCuda<T> &MixedPrecRPUDeviceCuda<T>::operator=(MixedPrecRPUDeviceCuda<T> &&other) {

  MixedPrecRPUDeviceBaseCuda<T>::operator=(std::move(other));

  dev_chi_  = std::move(other.dev_chi_);
  dev_m_    = std::move(other.dev_m_);
  dev_ns_X_ = std::move(other.dev_ns_X_);
  dev_ns_A_ = std::move(other.dev_ns_A_);
  dev_ns_B_ = std::move(other.dev_ns_B_);
  dev_ns_O_ = std::move(other.dev_ns_O_);

  return *this;
}

template <typename T>
void MixedPrecRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const MixedPrecRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects MixedPrecRPUDevice.");
  }

  MixedPrecRPUDeviceBaseCuda<T>::populateFrom(rpu_device_in); // will set sizes
  allocateContainers();
  const auto &par = this->getPar();

  std::vector<T> v;
  v.resize(this->size_);
  rpu_device.getChi(v.data());
  dev_chi_->assign(v.data()); // both in x-major

  this->context_->synchronize();
  if (par.use_muon) {
  allocateMuonContainers_();
  dev_m_->setConst((T)0); // simplest: start fresh
}

}
template <typename T>
__global__ void kernelWeightDecayInPlace(T *W, int n, T decay_mul) {
  RPU_CUDA_1D_KERNEL_LOOP(i, n) { W[i] *= decay_mul; }
}

template <typename T>
__global__ void kernelQuantizeBatch(
    T *quantized_values,
    const T *values,
    const T *nm_values,
    const int n_bins,
    const int size_in,
    const int m_batch_in,
    const bool trans_in) {

  int size = size_in;
  int m_batch = m_batch_in;
  int total_size = size * m_batch;
  bool trans = trans_in;
  T half_bins = (T)(n_bins / 2); // floor
  T res = (T)1.0 / ((T)half_bins);

  RPU_CUDA_1D_KERNEL_LOOP(idx, total_size) {

    T value = values[idx];

    int sidx = trans ? (idx % m_batch) : (idx / size);
    T amax = nm_values[sidx]; // amax from noise management
    value = amax > (T)0.0 ? value / amax : value;
    value = RPU_ROUNDFUN(value / res);
    value = MIN(MAX(value, -half_bins), half_bins) * amax * res;

    quantized_values[idx] = value;
  }
}
template <typename T>
__global__ void kernelScaleInPlace(T *x, int size, T alpha) {
  RPU_CUDA_1D_KERNEL_LOOP(i, size) { x[i] *= alpha; }
}

template <typename T>
__global__ void kernelDotAtomic(float *out, const T *A, const T *B, int n) {
  RPU_CUDA_1D_KERNEL_LOOP(i, n) {
    atomicAdd(out, (float)A[i] * (float)B[i]);
  }
}

template <typename T>
__global__ void kernelSumSquaresAtomic(float *out, const T *A, int n) {
  RPU_CUDA_1D_KERNEL_LOOP(i, n) {
    float v = (float)A[i];
    atomicAdd(out, v * v);
  }
}

template <typename T>
__global__ void kernelAxpyInPlace(T *y, const T *x, int size, T alpha) {
  RPU_CUDA_1D_KERNEL_LOOP(i, size) { y[i] += alpha * x[i]; }
}

// X = 1.5 X - 0.5 B
template <typename T>
__global__ void kernelNSUpdateX(T *X, const T *B, int size) {
  RPU_CUDA_1D_KERNEL_LOOP(i, size) { X[i] = (T)1.5 * X[i] - (T)0.5 * B[i]; }
}

template <typename T>
__global__ void kernelCopy(T *dst, const T *src, int size) {
  RPU_CUDA_1D_KERNEL_LOOP(i, size) { dst[i] = src[i]; }
}
template <typename T>
static void debug_check_uvt_svd(
    const T* hM,       // host M, col-major, (x_size x d_size)
    const T* hO,       // host O, col-major, (x_size x d_size)
    int x_size,
    int d_size,
    int dbg_id) {

  if constexpr (!std::is_same<T, float>::value && !std::is_same<T, double>::value) {
    std::cout << "[SVD-CHK] only supports float/double.\n";
    return;
  }

  const int m = x_size;
  const int n = d_size;
  const int r = std::min(m, n);

  // LAPACK overwrites input matrix, so copy
  std::vector<T> A((size_t)m * (size_t)n);
  std::copy(hM, hM + (size_t)m * (size_t)n, A.begin());

  std::vector<T> S(r);
  std::vector<T> U((size_t)m * (size_t)r);     // m x r
  std::vector<T> VT((size_t)r * (size_t)n);    // r x n  (this is V^T)

  // jobz='S': thin SVD
  int info = 0;
  if constexpr (std::is_same<T, float>::value) {
    info = LAPACKE_sgesdd(
        LAPACK_COL_MAJOR, 'S',
        m, n,
        (float*)A.data(), m,
        (float*)S.data(),
        (float*)U.data(), m,
        (float*)VT.data(), r);
  } else {
    info = LAPACKE_dgesdd(
        LAPACK_COL_MAJOR, 'S',
        m, n,
        (double*)A.data(), m,
        (double*)S.data(),
        (double*)U.data(), m,
        (double*)VT.data(), r);
  }

  if (info != 0) {
    std::cout << "[SVD-CHK " << dbg_id << "] LAPACKE_*gesdd failed, info=" << info << "\n";
    return;
  }

  // Compute UV^T (m x n), col-major
  std::vector<T> Osvd((size_t)m * (size_t)n, (T)0);
  // Osvd = U (m x r) * VT (r x n)
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      double acc = 0.0;
      for (int k = 0; k < r; ++k) {
        // U(row,k) in col-major => U[row + k*m]
        // VT(k,col) in col-major with ld=r => VT[k + col*r]
        acc += (double)U[row + k * m] * (double)VT[k + col * r];
      }
      Osvd[row + col * m] = (T)acc;
    }
  }

  // Compare O (NS) vs Osvd (SVD)
  double fro2 = 0.0;
  double fro2_ref = 0.0;
  double max_abs = 0.0;

  const size_t MN = (size_t)m * (size_t)n;
  for (size_t i = 0; i < MN; ++i) {
    double a = (double)hO[i];
    double b = (double)Osvd[i];
    double diff = a - b;
    fro2 += diff * diff;
    fro2_ref += b * b;
    max_abs = std::max(max_abs, std::abs(diff));
  }

  double fro = std::sqrt(fro2);
  double rel = fro / (std::sqrt(fro2_ref) + 1e-12);

  std::cout
    << "[SVD-CHK " << dbg_id << "] "
    << "||O - U V^T||_F=" << fro
    << "  rel=" << rel
    << "  max|diff|=" << max_abs
    << "\n";
}

// sum_{i} X[i]^2  (accumulate into a single float)
template <typename T>
__global__ void kernelSumSquares(float *out, const T *X, int size) {
  // atomic into one scalar (slow but OK for first working version)
  RPU_CUDA_1D_KERNEL_LOOP(i, size) {
    float v = (float)X[i];
    atomicAdd(out, v * v);
  }
}

template <typename T>
__global__ void kernelQuantizeBatchStochasticRounding(
    T *quantized_values,
    const T *values,
    const T *nm_values,
    const int n_bins,
    const int size_in,
    const int m_batch_in,
    const bool trans_in,
    curandState *random_states) {

  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  curandState local_state = random_states[tid];
  int size = size_in;
  int m_batch = m_batch_in;
  int total_size = size * m_batch;
  bool trans = trans_in;
  T half_bins = (T)(n_bins / 2); // floor
  T res = (T)1.0 / ((T)half_bins);

  RPU_CUDA_1D_KERNEL_LOOP(idx, total_size) {

    T stoch_value = curand_uniform(&local_state);
    T value = values[idx];

    int sidx = trans ? (idx % m_batch) : (idx / size);
    T amax = nm_values[sidx]; // amax from noise management
    value = amax > (T)0.0 ? value / amax : value;
    value = RPU_ROUNDFUN(value / res + stoch_value - (T)0.5);
    value = MIN(MAX(value, -half_bins), half_bins) * amax * res;

    quantized_values[idx] = value;
  }

  random_states[tid] = local_state;
}

template <typename T>
const T *MixedPrecRPUDeviceCuda<T>::quantize(
    T *buffer_values,
    const T *values,
    RPU::NoiseManager<T> *nm,
    int n_bins,
    int size,
    int m_batch,
    bool trans,
    bool stochastic_rounding) {

  if (n_bins <= 0) {
    return values;
  }

  nm->compute(values, NoiseManagementType::AbsMax, this->io_, m_batch, trans, false);
  int nthreads = this->context_->getNThreads();
  int nblocks = this->context_->getNBlocks(m_batch * size, nthreads);
  nblocks = MIN(this->nblocks_batch_max_, nblocks);

  cudaStream_t s = this->context_->getStream();
  if (stochastic_rounding) {
    kernelQuantizeBatchStochasticRounding<<<nblocks, nthreads, 0, s>>>(
        buffer_values, values, nm->getScaleValues(), n_bins, size, m_batch, trans,
        this->context_->getRandomStates(nthreads * nblocks));
  } else {
    kernelQuantizeBatch<<<nblocks, nthreads, 0, s>>>(
        buffer_values, values, nm->getScaleValues(), n_bins, size, m_batch, trans);
  }

  return buffer_values;
}
template <typename T>
__global__ void kernelNSPoly5UpdateX(T *X, const T *B, const T *C, int n, T a, T b, T c) {
  RPU_CUDA_1D_KERNEL_LOOP(i, n) {
    X[i] = a * X[i] + b * B[i] + c * C[i];
  }
}

template <typename T>
void MixedPrecRPUDeviceCuda<T>::computeMuonO_(
    T *dev_O, const T *dev_M, int x_size, int d_size) {

  const auto &par = this->getPar();
  const int DX = x_size * d_size;
  const int DD = d_size * d_size;

  // X <- M
  {
    int nthreads = this->context_->getNThreads();
    int nblocks = this->context_->getNBlocks(DX, nthreads);
    kernelCopy<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
        dev_ns_X_->getData(), dev_M, DX);
  }

  // Optional: Frobenius normalization for stability
  if (par.ns_fro_norm) {
    CudaArray<float> dev_ss(this->context_, 1);
    dev_ss.setConst(0.0f);

    int nthreads = this->context_->getNThreads();
    int nblocks = this->context_->getNBlocks(DX, nthreads);
    kernelSumSquares<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
        dev_ss.getData(), dev_ns_X_->getDataConst(), DX);

    float h_ss = 0.0f;
    dev_ss.copyTo(&h_ss);
    this->context_->synchronizeDevice();

    float inv_norm = 1.0f / std::sqrt(h_ss + (float)par.ns_eps);

    nblocks = this->context_->getNBlocks(DX, nthreads);
    kernelScaleInPlace<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
        dev_ns_X_->getData(), DX, (T)inv_norm);
  }

for (int it = 0; it < par.ns_iters; ++it) {

  // G = X^T X   (d x d)
  RPU::math::gemm<T>(
      this->context_,
      true, false,
      d_size, d_size, x_size,
      (T)1.0,
      dev_ns_X_->getDataConst(), x_size,
      dev_ns_X_->getDataConst(), x_size,
      (T)0.0,
      dev_ns_A_->getData(), d_size);

  // B = X G     (x x d)
  RPU::math::gemm<T>(
      this->context_,
      false, false,
      x_size, d_size, d_size,
      (T)1.0,
      dev_ns_X_->getDataConst(), x_size,
      dev_ns_A_->getDataConst(), d_size,
      (T)0.0,
      dev_ns_B_->getData(), x_size);

  // G2 = G G    (d x d)
  RPU::math::gemm<T>(
      this->context_,
      false, false,
      d_size, d_size, d_size,
      (T)1.0,
      dev_ns_A_->getDataConst(), d_size,
      dev_ns_A_->getDataConst(), d_size,
      (T)0.0,
      dev_ns_G2_->getData(), d_size);

  // C = X G2    (x x d)
  RPU::math::gemm<T>(
      this->context_,
      false, false,
      x_size, d_size, d_size,
      (T)1.0,
      dev_ns_X_->getDataConst(), x_size,
      dev_ns_G2_->getDataConst(), d_size,
      (T)0.0,
      dev_ns_C_->getData(), x_size);

  // X = aX + bB + cC   (elementwise)
  int nthreads = this->context_->getNThreads();
  int nblocks  = this->context_->getNBlocks(DX, nthreads);
  kernelNSPoly5UpdateX<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      dev_ns_X_->getData(),
      dev_ns_B_->getDataConst(),
      dev_ns_C_->getDataConst(),
      DX,
      par.ns_a, par.ns_b, par.ns_c);
}
// ===== SVD debug check (only first few calls) =====
// {
//   static int dbg_count = 0;
//   const int DBG_MAX = 5;  // 只检查前 5 次
//   if (dbg_count < DBG_MAX) {

//     // copy M and X(O) from device to host
//     const int DX = x_size * d_size;
//     std::vector<T> hM(DX);
//     std::vector<T> hO(DX);

//     cudaStream_t s = this->context_->getStream();
//     cudaMemcpyAsync(hM.data(), dev_M, (size_t)DX * sizeof(T), cudaMemcpyDeviceToHost, s);
//     cudaMemcpyAsync(hO.data(), dev_ns_X_->getDataConst(), (size_t)DX * sizeof(T), cudaMemcpyDeviceToHost, s);
//     cudaStreamSynchronize(s);

//     debug_check_uvt_svd<T>(hM.data(), hO.data(), x_size, d_size, dbg_count);
//     dbg_count++;
//   }
// }

  // Output O <- X  (same layout as chi, x*d with lda=x_size)
  {
    int nthreads = this->context_->getNThreads();
    int nblocks = this->context_->getNBlocks(DX, nthreads);
    kernelCopy<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
        dev_O, dev_ns_X_->getDataConst(), DX);
  }
}

template <typename T>
void MixedPrecRPUDeviceCuda<T>::doDirectUpdate(
    const T *x_input,
    const T *d_input,
    T *dev_weights,
    const T lr,
    const int m_batch,
    const bool x_trans,
    const bool d_trans,
    const T beta,
    const PulsedUpdateMetaParameter<T> &up,
    T *x_buffer,
    T *d_buffer) {

  if (beta != (T)1.0) {
    RPU_FATAL("beta not equal 1 is not supported.")
  }
  static bool printed = false;
  if (!printed) {
    printed = true;
    std::cout
      << "[HIT] MixedPrecRPUDeviceCuda::doDirectUpdate"
      << std::endl;
  }

  this->setUpPar(up);
  const auto &par = getPar();

  const T *d_val = quantize(
      d_buffer, d_input, &*this->noise_manager_d_, par.n_d_bins, this->d_size_, m_batch, d_trans,
      par.stoc_round_d);

  // % Quantize x
  const T *x_val = quantize(
      x_buffer, x_input, &*this->noise_manager_x_, par.n_x_bins, this->x_size_, m_batch, x_trans,
      par.stoc_round_x);

  // dev_chi is x-size (row) major !! (to facilitate the readout below)

  // if (m_batch == 1) {
  //   RPU::math::ger<T>(
  //       this->context_, this->x_size_, this->d_size_, lr, x_val, 1, d_val, 1, dev_chi_->getData(),
  //       this->x_size_);
  // } else {
  //   RPU::math::gemm<T>(
  //       this->context_, x_trans, !d_trans, this->x_size_, this->d_size_, m_batch, lr, x_val,
  //       x_trans ? m_batch : this->x_size_, d_val, d_trans ? m_batch : this->d_size_,
  //       1.0, // set beta to 1.0. We want to add to Chi
  //       dev_chi_->getData(), this->x_size_);
  // }
    // dev_chi is stored column-major with leading dim = x_size:
  // shape (x_size x d_size), so column j (d-index) is contiguous with length x_size.
  static bool printed_muon = false;
if (!printed_muon) {
  printed_muon = true;
  std::cout << "[MUON] use_muon=" << par.use_muon
            << " ns_iters=" << par.ns_iters
            << " muon_momentum=" << par.muon_momentum
            << " ns_fro_norm=" << par.ns_fro_norm
            << std::endl;
}

  if (par.use_muon) {
    T alphaG = (T)1.0 - (T)par.muon_momentum;

    // lazy alloc muon buffers
    allocateMuonContainers_();

    // M *= mu
    {
      int nthreads = this->context_->getNThreads();
      int nblocks = this->context_->getNBlocks(this->size_, nthreads);
      kernelScaleInPlace<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
          dev_m_->getData(), this->size_, (T)par.muon_momentum);
    }

    // M += G   (G = x * d^T aggregated over batch)
    if (m_batch == 1) {
      RPU::math::ger<T>(
          this->context_, this->x_size_, this->d_size_,
          // alphaG, 
          1, 
          x_val, 1,
          d_val, 1,
          dev_m_->getData(), this->x_size_);
    } else {
      RPU::math::gemm<T>(
          this->context_,
          x_trans, !d_trans,
          this->x_size_, this->d_size_, m_batch,
          // alphaG,  // alpha=alphaG (multiply lr here)
          1,
          x_val,
          x_trans ? m_batch : this->x_size_,
          d_val,
          d_trans ? m_batch : this->d_size_,
          (T)1.0,  // beta=1, accumulate into M
          dev_m_->getData(),
          this->x_size_);
    }

    // O = NewtonSchulz(M)
    computeMuonO_(dev_ns_O_->getData(), dev_m_->getDataConst(), this->x_size_, this->d_size_);

    // chi += -lr * O * sqrt{max(A,B)}
    T dim_scale = (T)std::sqrt((double)MAX(this->x_size_, this->d_size_));
    T step = (T)(lr) * dim_scale; 
    // T step = (T)(lr);
    {
      int nthreads = this->context_->getNThreads();
      int nblocks = this->context_->getNBlocks(this->size_, nthreads);
      kernelAxpyInPlace<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
          dev_chi_->getData(), dev_ns_O_->getDataConst(), this->size_, (T)(step*0.01));
    }

  } else {

    // ===== original MixedPrec behavior: directly accumulate into Chi =====
    if (m_batch == 1) {
      RPU::math::ger<T>(
          this->context_, this->x_size_, this->d_size_, lr, x_val, 1, d_val, 1,
          dev_chi_->getData(), this->x_size_);
    } else {
      RPU::math::gemm<T>(
          this->context_, x_trans, !d_trans, this->x_size_, this->d_size_, m_batch, lr, x_val,
          x_trans ? m_batch : this->x_size_, d_val, d_trans ? m_batch : this->d_size_,
          (T)1.0, // add to Chi
          dev_chi_->getData(), this->x_size_);
    }
  }
if (par.use_muon && par.muon_wd > (T)0) {
  T etaW = (T)lr * (T)0.01;

  T decay_mul = (T)1.0 - etaW * (T)par.muon_wd;

  int nthreads = this->context_->getNThreads();
  int nblocks  = this->context_->getNBlocks(this->size_, nthreads);
  kernelScaleInPlace<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      dev_weights, this->size_, decay_mul);
}

  this->doTransfer(dev_weights,1, m_batch);
  // after doTransfer(...)
{
  static int wdbg = 0;
  const int WDBG_MAX = 5;
  if (wdbg % 100 < WDBG_MAX) {
    std::vector<T> hw(this->size_);
    cudaStream_t s = this->context_->getStream();
    cudaMemcpyAsync(hw.data(), dev_weights, (size_t)this->size_ * sizeof(T),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    double wmin = 1e100, wmax = -1e100;
    for (int i = 0; i < this->size_; ++i) {
      double v = (double)hw[i];
      if (v < wmin) wmin = v;
      if (v > wmax) wmax = v;
    }
    std::cout << "[W-RANGE " << wdbg << "] min=" << wmin << " max=" << wmax << std::endl;
    wdbg++;
  }
}

  this->computeSparsity(x_buffer, d_buffer, m_batch);
  this->advanceUpdateCounter(m_batch);
}

template <typename T>
__global__ void
kernelMixedPrecTransfer(T *transfer_out, T *chi, const int size, const T granularity) {
  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < size) {
    T value = chi[tid];
    T dw = trunc(value / granularity);
    transfer_out[tid] = dw;

    chi[tid] = value - granularity * dw;
  }
}

template <typename T>
void MixedPrecRPUDeviceCuda<T>::forwardUpdate(
    T *dev_weights,
    const T lr,
    int i_row_start,
    const T *transfer_vec,
    const int n_vec,
    const bool trans) {

  if (!lr) {
    return;
  }
  int t_size = n_vec * this->x_size_;
  if ((this->dev_transfer_tmp_ == nullptr) || this->dev_transfer_tmp_->getSize() < t_size) {
    this->dev_transfer_tmp_ = RPU::make_unique<CudaArray<T>>(this->context_, t_size);
  }

  const auto &par = this->getPar();

  int nthreads = this->context_->getNThreads();
  int nblocks = this->context_->getNBlocks(t_size, nthreads);
  kernelMixedPrecTransfer<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      this->dev_transfer_tmp_->getData(), dev_chi_->getData() + i_row_start * this->x_size_, t_size,
      this->granularity_);

  // requires to turn on update_managment / bl managment as well
  this->transfer_pwu_->update(
      this->dev_transfer_tmp_->getDataConst(), // this is the transfer vector (x_size)
      transfer_vec,                            // this should be d_size, non-trans
      dev_weights, &*this->rpucuda_device_, this->up_, lr * this->granularity_, n_vec, trans,
      false);
}

template <typename T> std::vector<T> MixedPrecRPUDeviceCuda<T>::getHiddenWeights() const {

  auto data = MixedPrecRPUDeviceBaseCuda<T>::getHiddenWeights();

  int offset = data.size();
  data.resize(offset + this->size_);
  dev_chi_->copyTo(data.data() + offset);

  return data;
}

template class MixedPrecRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class MixedPrecRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class MixedPrecRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
