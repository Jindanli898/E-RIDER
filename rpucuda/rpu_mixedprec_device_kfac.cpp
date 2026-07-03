/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "rpu_mixedprec_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <memory>
#include <sstream>
#include <cmath>

namespace RPU {

/******************************************************************************************/
/* MixedPrecRPUDeviceMetaParameter*/

template <typename T>
void MixedPrecRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {
  ss << "\t\bUpdate using digital outer product + transfer to analog: " << std::endl;

  ss << "\t n_x_bins: \t\t";
  ss << n_x_bins << std::endl;
  ss << "\t n_d_bins: \t\t";
  ss << n_d_bins << std::endl;
  ss << "\t transfer_lr: \t\t";
  ss << transfer_lr << std::endl;
  if (stoc_round_x || stoc_round_d) {
    ss << "\t stoc_round (x / d): \t";
    ss << std::boolalpha << stoc_round_x << " / " << stoc_round_d << std::endl;
  }

  MixedPrecRPUDeviceBaseMetaParameter<T>::printToStream(ss);
}

template <typename T> void MixedPrecRPUDeviceMetaParameter<T>::initialize() {
  if (!this->_par_initialized) {
    MixedPrecRPUDeviceBaseMetaParameter<T>::initialize();

    if (n_d_bins <= 0 || n_x_bins <= 0) {
      this->compute_sparsity = false;
    }
  }
}

template struct MixedPrecRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct MixedPrecRPUDeviceMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct MixedPrecRPUDeviceMetaParameter<half_t>;
#endif

/******************************************************************************************/

template <typename T> void MixedPrecRPUDevice<T>::initialize(int x_sz, int d_sz) {
  freeContainers();
  chi_ = Array_2D_Get<T>(d_sz, x_sz);
  m_   = Array_2D_Get<T>(d_sz, x_sz);

  PRAGMA_SIMD
  for (int i = 0; i < x_sz * d_sz; i++) {
    chi_[0][i] = 0;
    m_[0][i]   = (T)0;
  }
}

template <typename T> void MixedPrecRPUDevice<T>::freeContainers() {
  if (chi_ != nullptr) {
    Array_2D_Free<T>(chi_);
    chi_ = nullptr;
  }
  if (m_   != nullptr) { Array_2D_Free<T>(m_);   m_   = nullptr; }
}

// dtor
template <typename T> MixedPrecRPUDevice<T>::~MixedPrecRPUDevice() { freeContainers(); }

// ctor
template <typename T>
MixedPrecRPUDevice<T>::MixedPrecRPUDevice(int x_sz, int d_sz)
    : MixedPrecRPUDeviceBase<T>(x_sz, d_sz) {}

template <typename T>
MixedPrecRPUDevice<T>::MixedPrecRPUDevice(
    int x_sz, int d_sz, const MixedPrecRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : MixedPrecRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy construcutor
// template <typename T>
// MixedPrecRPUDevice<T>::MixedPrecRPUDevice(const MixedPrecRPUDevice<T> &other)
//     : MixedPrecRPUDeviceBase<T>(other) {

//   initialize(other.x_size_, other.d_size_);

//   if (other.chi_ != nullptr) {
//     setChi(other.chi_[0]);
//   }
//   qx_ = other.qx_;
//   qd_ = other.qd_;
//   qx_index_ = other.qx_index_;
// }
template <typename T>
MixedPrecRPUDevice<T>::MixedPrecRPUDevice(const MixedPrecRPUDevice<T> &other)
    : MixedPrecRPUDeviceBase<T>(other) {

  initialize(other.x_size_, other.d_size_);

  if (other.chi_ != nullptr) {
    setChi(other.chi_[0]);
  }

  if (other.m_ != nullptr) {
    for (int k = 0; k < other.d_size_ * other.x_size_; ++k) {
      m_[0][k] = other.m_[0][k];
    }
  }

  qx_ = other.qx_;
  qd_ = other.qd_;
  qx_index_ = other.qx_index_;

  ns_X_ = other.ns_X_;
  ns_A_ = other.ns_A_;
  ns_B_ = other.ns_B_;
  ns_O_ = other.ns_O_;
}

// copy assignment
template <typename T>
MixedPrecRPUDevice<T> &MixedPrecRPUDevice<T>::operator=(const MixedPrecRPUDevice<T> &other) {

  MixedPrecRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
MixedPrecRPUDevice<T>::MixedPrecRPUDevice(MixedPrecRPUDevice<T> &&other) noexcept {
  *this = std::move(other);
}

// // move assignment
// template <typename T>
// MixedPrecRPUDevice<T> &MixedPrecRPUDevice<T>::operator=(MixedPrecRPUDevice<T> &&other) noexcept {
//   MixedPrecRPUDeviceBase<T>::operator=(std::move(other));

//   chi_ = std::move(other.chi_);
//   qx_ = other.qx_;
//   qd_ = other.qd_;
//   qx_index_ = other.qx_index_;

//   return *this;
// }
// move assignment
template <typename T>
MixedPrecRPUDevice<T> &MixedPrecRPUDevice<T>::operator=(MixedPrecRPUDevice<T> &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  // 释放自己原来的 chi_/m_，避免泄漏
  freeContainers();  // 你需要让它同时 free chi_ 和 m_

  // move 基类
  MixedPrecRPUDeviceBase<T>::operator=(std::move(other));

  // ===== 照着 chi 的做法接管指针，但必须置空源对象 =====
  chi_ = other.chi_;
  other.chi_ = nullptr;

  m_ = other.m_;
  other.m_ = nullptr;

  // vectors 用 move（否则变成拷贝）
  qx_ = std::move(other.qx_);
  qd_ = std::move(other.qd_);
  qx_index_ = std::move(other.qx_index_);

  // 如果你加了 NS buffers，也 move
  ns_X_ = std::move(other.ns_X_);
  ns_A_ = std::move(other.ns_A_);
  ns_B_ = std::move(other.ns_B_);
  ns_O_ = std::move(other.ns_O_);

  return *this;
}

/*********************************************************************************/
/* populate */

template <typename T>
void MixedPrecRPUDevice<T>::populate(
    const MixedPrecRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  MixedPrecRPUDeviceBase<T>::populate(p, rng);
  initialize(this->x_size_, this->d_size_);
}

/*********************************************************************************/
/* forwardUpdate */

template <typename T>
void MixedPrecRPUDevice<T>::forwardUpdate(
    T **weights,
    const T lr,
    int j_row_start,
    const T *transfer_d_vec,
    const int n_vec,
    const bool trans) {

  if (!lr) { // not used actually
    return;
  }
  if (trans) { // not needed really
    RPU_NOT_IMPLEMENTED;
  }

  if (this->transfer_tmp_.size() < (size_t)this->x_size_) {
    this->transfer_tmp_.resize(this->x_size_);
  }

  if (this->granularity_ <= (T)0.0) {
    RPU_FATAL("Granularity cannot be zero!");
  }

  // forward / update
  for (size_t j = 0; j < (size_t)n_vec; j++) {
    T *chi_row = chi_[j_row_start + j];

    PRAGMA_SIMD
    for (size_t i = 0; i < (size_t)this->x_size_; i++) {
      T value = chi_row[i];
      T dw = (T)truncf(value / this->granularity_);
      this->transfer_tmp_[i] = dw;
      chi_row[i] = value - dw * this->granularity_;
    }

    this->transfer_pwu_->updateVectorWithDevice(
        weights, this->transfer_tmp_.data(), 1, transfer_d_vec + (size_t)this->d_size_ * j, 1,
        this->granularity_ * lr, n_vec, &*this->rpu_device_);
  }
}

template <typename T>
void MixedPrecRPUDevice<T>::computeMuonO(T *O, const T *M, int d_size, int x_size) const {
  const auto &par = getPar();
  const int DX = d_size * x_size;
  const int DD = d_size * d_size;

  // resize buffers
  ns_X_.resize(DX);
  ns_A_.resize(DD);
  ns_B_.resize(DX);

  // X = M
  PRAGMA_SIMD
  for (int i = 0; i < DX; ++i) ns_X_[i] = M[i];

  // normalize X0 for stability
  if (par.ns_fro_norm) {
    double ss = 0.0;
    for (int i = 0; i < DX; ++i) {
      double v = (double)ns_X_[i];
      ss += v * v;
    }
    T norm = (T)std::sqrt(ss) + par.ns_eps;
    PRAGMA_SIMD
    for (int i = 0; i < DX; ++i) ns_X_[i] = ns_X_[i] / norm;
  }

  // Newton–Schulz iterations: X <- 1.5 X - 0.5 (X X^T) X
for (int it = 0; it < par.ns_iters; ++it) {

  // A = X X^T  (d x d)
  RPU::math::gemm<T>(
      CblasRowMajor,
      CblasNoTrans, CblasTrans,
      d_size, d_size, x_size,
      (T)1.0,
      ns_X_.data(), x_size,
      ns_X_.data(), x_size,
      (T)0.0,
      ns_A_.data(), d_size);

  // B = A X  (d x x)
  RPU::math::gemm<T>(
      CblasRowMajor,
      CblasNoTrans, CblasNoTrans,
      d_size, x_size, d_size,
      (T)1.0,
      ns_A_.data(), d_size,
      ns_X_.data(), x_size,
      (T)0.0,
      ns_B_.data(), x_size);

  // X = 1.5 X - 0.5 B
  PRAGMA_SIMD
  for (int i = 0; i < DX; ++i) {
    ns_X_[i] = (T)1.5 * ns_X_[i] - (T)0.5 * ns_B_[i];
  }
}


  // output
  PRAGMA_SIMD
  for (int i = 0; i < DX; ++i) O[i] = ns_X_[i];
}


/*********************************************************************************/
/* update */

template <typename T>
void MixedPrecRPUDevice<T>::doDirectVectorUpdate(
    T **weights,
    const T *x_input,
    const int x_inc,
    const T *d_input,
    const int d_inc,
    const T learning_rate,
    const int m_batch_info,
    const PulsedUpdateMetaParameter<T> &up) {

  this->setUpPar(up);
  const auto &par = getPar();

  // NEED TO CHECK LEARNING RATE DIRECTION CORRECT ...
  T x_width = (T)1.0;
  T half_x_bins = (T)(par.n_x_bins / 2); // floor
  if (par.n_x_bins > 0) {
    int max_index = RPU::math::iamax<T>(this->x_size_, x_input, x_inc);
    T x_amax = (T)fabsf(x_input[max_index * x_inc]);
    x_width = x_amax / ((T)half_x_bins);
    x_width = x_width == (T)0.0 ? (T)1.0 : x_width;
  }

  T d_width = (T)1.0;
  T half_d_bins = (T)(par.n_d_bins / 2); // floor
  if (par.n_d_bins > 0) {

    int max_index = RPU::math::iamax<T>(this->d_size_, d_input, d_inc);
    T d_amax = (T)fabsf(d_input[max_index * d_inc]);
    d_width = d_amax / ((T)half_d_bins);
    d_width = d_width == (T)0.0 ? (T)1.0 : d_width;
  }

  int i_stop = this->x_size_ * x_inc;
  int j_stop = this->d_size_ * d_inc;
  int i = -1;
  int j = -1;
  int kx = 0;
  int kd = 0;
  if (par.use_muon) {
    printf("Muon active\n");
    PRAGMA_SIMD
    for (int t = 0; t < this->size_; ++t) {
      m_[0][t] *= par.muon_momentum;
    }
  }


  if (par.n_d_bins <= 0 && par.n_x_bins <= 0) {

    // RPU::math::ger<T>(
    //     CblasRowMajor, this->d_size_, this->x_size_, learning_rate, d_input, d_inc, x_input, x_inc,
    //     chi_[0], this->x_size_);
    T *accum = par.use_muon ? m_[0] : chi_[0];
    T alpha = par.use_muon ? (T)1.0 : learning_rate;
       RPU::math::ger<T>(
        CblasRowMajor, this->d_size_, this->x_size_, alpha, d_input, d_inc, x_input, x_inc,
        accum, this->x_size_);
        
  } else if (par.n_d_bins <= 0 || par.n_x_bins <= 0) {

    qx_.resize(this->x_size_);
    qd_.resize(this->d_size_);

    const bool stochastic_rounding_x = par.stoc_round_x;
    T stoch_value = 0.0;

    PRAGMA_SIMD
    for (int i_x = 0; i_x < i_stop; i_x += x_inc) {
      i++;
      T x = x_input[i_x];

      // quantize (by abs max, thus already clipped)
      if (par.n_x_bins > 0) {

        if (stochastic_rounding_x) {
          stoch_value = this->rng_.sampleUniform() - (T)0.5;
        }

        x = RPU_ROUNDFUNF(x / x_width + stoch_value);
        x = MIN(MAX(x, -half_x_bins), half_x_bins) * x_width;
      }
      qx_[i] = x;
    }

    const bool stochastic_rounding_d = par.stoc_round_d;
    stoch_value = 0.0;

    PRAGMA_SIMD
    for (int j_d = 0; j_d < j_stop; j_d += d_inc) {
      j++;
      T d = d_input[j_d];

      // quantize (by abs max, thus already clipped)
      if (par.n_d_bins > 0) {

        if (stochastic_rounding_d) {
          stoch_value = this->rng_.sampleUniform() - (T)0.5;
        }

        d = RPU_ROUNDFUNF(d / d_width + stoch_value);
        d = MIN(MAX(d, -half_d_bins), half_d_bins) * d_width;
      }
      // qd_[i] = d;
      qd_[j] = d;
    }

    // RPU::math::ger<T>(
    //     CblasRowMajor, this->d_size_, this->x_size_, learning_rate, qd_.data(), 1, qx_.data(), 1,
    //     chi_[0], this->x_size_);
    T *accum = par.use_muon ? m_[0] : chi_[0];
    T alpha = par.use_muon ? (T)1.0 : learning_rate;
     RPU::math::ger<T>(
        CblasRowMajor, this->d_size_, this->x_size_, alpha, qd_.data(), 1, qx_.data(), 1,
        accum, this->x_size_);

  } else {
    // sparse outer product

    qx_.resize(this->x_size_);
    qx_index_.resize(this->x_size_);
    const bool stochastic_rounding_x = par.stoc_round_x;
    T stoch_value = 0.0;

    PRAGMA_SIMD
    for (int i_x = 0; i_x < i_stop; i_x += x_inc) {
      T x = x_input[i_x];
      i++;
      // quantize (by abs max, thus already clipped)

      if (stochastic_rounding_x) {
        stoch_value = this->rng_.sampleUniform() - (T)0.5;
      }

      T qx = RPU_ROUNDFUNF(x / x_width + stoch_value);

      if (qx == (T)0.0) {
        continue;
      }
      qx_index_[kx] = i;
      qx = MIN(MAX(qx, -half_x_bins), half_x_bins);
      qx_[kx++] = qx * x_width;
    }

    const bool stochastic_rounding_d = par.stoc_round_d;
    stoch_value = 0.0;

    // for (int j_d = 0; j_d < j_stop; j_d += d_inc) {
    //   T d = d_input[j_d];
    //   j++;

    //   // quantize
    //   if (stochastic_rounding_d) {
    //     stoch_value = this->rng_.sampleUniform() - (T)0.5;
    //   }

    //   T qd = RPU_ROUNDFUNF(d / d_width + stoch_value);
    //   if (qd == (T)0) {
    //     continue;
    //   }
    //   qd = MIN(MAX(qd, -half_d_bins), half_d_bins);
    //   qd *= d_width;

    //   T *chi_row = chi_[j];
    //   kd++;
    //   PRAGMA_SIMD
    //   for (int ii = 0; ii < kx; ii++) {
    //     int idx = qx_index_[ii];
    //     T qx = qx_[ii];
    //     // chi_row[idx] += learning_rate * qd * qx;
    //     m_row[idx] += qd * qx;

    //   }
    // }
    for (int j_d = 0; j_d < j_stop; j_d += d_inc) {
  T d = d_input[j_d];
  j++;

  // quantize
  if (stochastic_rounding_d) {
    stoch_value = this->rng_.sampleUniform() - (T)0.5;
  }

  T qd = RPU_ROUNDFUNF(d / d_width + stoch_value);
  if (qd == (T)0) {
    continue;
  }
  qd = MIN(MAX(qd, -half_d_bins), half_d_bins);
  qd *= d_width;

  // choose accumulator row
  T *row = par.use_muon ? m_[j] : chi_[j];

  kd++;
  PRAGMA_SIMD
  for (int ii = 0; ii < kx; ii++) {
    int idx = qx_index_[ii];
    T qx = qx_[ii];

    if (par.use_muon) {
      // accumulate raw outer-product into momentum buffer
      row[idx] += qd * qx;
    } else {
      // original SGD-style accumulation into chi
      row[idx] += learning_rate * qd * qx;
    }
  }
}

  }
    if (par.use_muon) {
    ns_O_.resize(this->size_);
    computeMuonO(ns_O_.data(), m_[0], this->d_size_, this->x_size_);

    PRAGMA_SIMD
    for (int t = 0; t < this->size_; ++t) {
      chi_[0][t] += (T)(-learning_rate) * ns_O_[t];
    }
  }

  this->doTransfer(weights, par.transfer_lr, m_batch_info);
  this->computeSparsity(kx, kd); // will only compute if both are quantized
  this->advanceUpdateCounter();
}

template <typename T> bool MixedPrecRPUDevice<T>::onSetWeights(T **weights) {

  // reset chi
  initialize(this->x_size_, this->d_size_);
  MixedPrecRPUDeviceBase<T>::onSetWeights(weights);

  return true; // modified device thus true
}

template <typename T> void MixedPrecRPUDevice<T>::getChi(T *data) const {
  for (int i = 0; i < this->size_; ++i) {
    data[i] = chi_[0][i];
  }
}

template <typename T> void MixedPrecRPUDevice<T>::setChi(const T *data) {
  for (int i = 0; i < this->size_; ++i) {
    chi_[0][i] = data[i];
  }
}

template class MixedPrecRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class MixedPrecRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class MixedPrecRPUDevice<half_t>;
#endif

} // namespace RPU
