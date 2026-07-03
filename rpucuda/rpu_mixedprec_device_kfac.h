// /**
//  * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
//  *
//  * Licensed under the MIT license. See LICENSE file in the project root for details.
//  */

// #pragma once
// #include "rpu_mixedprec_device_base.h"
// #include "rpu_pulsed_device.h"
// #include "rpu_simple_device.h"
// #include "rpu_weight_updater.h"
// #include <sstream>
// #include <stdio.h>

// namespace RPU {

// template <typename T> class MixedPrecRPUDevice;

// /* Defines the mixed prec device.

//    outer-product update is computed in digital in reduced precision (e.g. 3 bins
//    for d only) and thus highly sparse. The update is stored in a Chi matrix in
//    digital.

//    Each mini-batch, the full Chi matrix transferred to the analog
//    weights, depending on the device granularity, as suggested by
//    Nandakumar et al. Front. in Neurosci. (2020).

//  */

// template <typename T>
// struct MixedPrecRPUDeviceMetaParameter : MixedPrecRPUDeviceBaseMetaParameter<T> {

//   int n_x_bins = 0;
//   int n_d_bins = 0;
//   bool stoc_round_d = false;
//   bool stoc_round_x = false;
//   T transfer_lr = 1.0;

//   MixedPrecRPUDeviceMetaParameter() = default;
//   ~MixedPrecRPUDeviceMetaParameter() = default;

//   friend void
//   swap(MixedPrecRPUDeviceMetaParameter<T> &a, MixedPrecRPUDeviceMetaParameter<T> &b) noexcept {
//     using std::swap;
//     swap(
//         static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(a),
//         static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(b));

//     swap(a.n_d_bins, b.n_d_bins);
//     swap(a.n_x_bins, b.n_x_bins);
//     swap(a.transfer_lr, b.transfer_lr);
//     swap(a.stoc_round_x, b.stoc_round_x);
//     swap(a.stoc_round_d, b.stoc_round_d);
//   }

//   std::string getName() const override {
//     std::ostringstream ss;
//     if (!this->device_par) {
//       ss << "MixedPrec[UNDEFINED]";
//     } else {
//       ss << "MixedPrec[" << this->device_par->getName() << "]";
//     }
//     return ss.str();
//   };

//   MixedPrecRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
//     return new MixedPrecRPUDevice<T>(x_size, d_size, *this, rng);
//   };

//   MixedPrecRPUDeviceMetaParameter<T> *clone() const override {
//     return new MixedPrecRPUDeviceMetaParameter<T>(*this);
//   };
//   DeviceUpdateType implements() const override { return DeviceUpdateType::MixedPrec; };
//   void printToStream(std::stringstream &ss) const override;
//   void initialize() override;
// };

// template <typename T> class MixedPrecRPUDevice : public MixedPrecRPUDeviceBase<T> {

// public:
//   // constructor / destructor
//   MixedPrecRPUDevice(int x_size, int d_size);
//   MixedPrecRPUDevice(
//       int x_size, int d_size, const MixedPrecRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
//   ~MixedPrecRPUDevice();

//   MixedPrecRPUDevice(const MixedPrecRPUDevice<T> &);
//   MixedPrecRPUDevice<T> &operator=(const MixedPrecRPUDevice<T> &);
//   MixedPrecRPUDevice(MixedPrecRPUDevice<T> &&) noexcept;
//   MixedPrecRPUDevice<T> &operator=(MixedPrecRPUDevice<T> &&) noexcept;

//   friend void swap(MixedPrecRPUDevice<T> &a, MixedPrecRPUDevice<T> &b) noexcept {
//     using std::swap;
//     swap(static_cast<MixedPrecRPUDeviceBase<T> &>(a), static_cast<MixedPrecRPUDeviceBase<T> &>(b));

//     swap(a.chi_, b.chi_);
//     swap(a.qx_, b.qx_);
//     swap(a.qd_, b.qd_);
//     swap(a.qx_index_, b.qx_index_);
//   }

//   MixedPrecRPUDeviceMetaParameter<T> &getPar() const override {
//     return static_cast<MixedPrecRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
//   };

//   MixedPrecRPUDevice<T> *clone() const override { return new MixedPrecRPUDevice<T>(*this); };

//   bool onSetWeights(T **weights) override;
//   void getChi(T *data) const override;
//   void setChi(const T *data) override;

//   void forwardUpdate(
//       T **weights,
//       const T lr,
//       int i_row_start,
//       const T *transfer_vec,
//       const int n_vec,
//       const bool trans) override;

//   void doDirectVectorUpdate(
//       T **weights,
//       const T *x_input,
//       const int x_inc,
//       const T *d_input,
//       const int d_inc,
//       const T learning_rate,
//       const int m_batch_info,
//       const PulsedUpdateMetaParameter<T> &up) override;

// protected:
//   void populate(const MixedPrecRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

// private:
//   void initialize(int x_size, int d_size);
//   void freeContainers();

//   // handled in base
//   T **chi_ = nullptr;

//   // temporary
//   std::vector<T> qx_;
//   std::vector<T> qd_;
//   std::vector<int> qx_index_;
// };

// } // namespace RPU
/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once
#include "rpu_mixedprec_device_base.h"
#include "rpu_pulsed_device.h"
#include "rpu_simple_device.h"
#include "rpu_weight_updater.h"
#include <sstream>
#include <stdio.h>
#include <vector>

namespace RPU {

template <typename T> class MixedPrecRPUDevice;

/* Defines the mixed prec device.

   outer-product update is computed in digital in reduced precision (e.g. 3 bins
   for d only) and thus highly sparse. The update is stored in a Chi matrix in
   digital.

   Each mini-batch, the full Chi matrix transferred to the analog
   weights, depending on the device granularity, as suggested by
   Nandakumar et al. Front. in Neurosci. (2020).

 */

template <typename T>
struct MixedPrecRPUDeviceMetaParameter : MixedPrecRPUDeviceBaseMetaParameter<T> {

  int n_x_bins = 0;
  int n_d_bins = 0;
  bool stoc_round_d = false;
  bool stoc_round_x = false;
  T transfer_lr = (T)0.02;

  // ----------------------------
  // Muon-like (optional) settings
  // ----------------------------
  bool use_muon = true;                 // if true: use M <- mu*M + G, O <- NewtonSchulz(M)
  T muon_momentum = (T)0.8; 
  T muon_wd = (T)1e-4;             // μ in M_t = μ M_{t-1} + G_t
  int ns_iters = 10;                      // Newton–Schulz iterations
  T ns_eps = (T)1e-7;                    // epsilon for normalization / stability
  bool ns_fro_norm = true;               // normalize M by Frobenius norm before NS
T ns_a = (T)3.4445;
T ns_b = (T)-4.7750;
T ns_c = (T)2.0315;

  MixedPrecRPUDeviceMetaParameter() = default;
  ~MixedPrecRPUDeviceMetaParameter() = default;

  friend void
  swap(MixedPrecRPUDeviceMetaParameter<T> &a, MixedPrecRPUDeviceMetaParameter<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(a),
        static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(b));

    swap(a.n_d_bins, b.n_d_bins);
    swap(a.n_x_bins, b.n_x_bins);
    swap(a.transfer_lr, b.transfer_lr);
    swap(a.stoc_round_x, b.stoc_round_x);
    swap(a.stoc_round_d, b.stoc_round_d);

    // Muon fields
    swap(a.use_muon, b.use_muon);
    swap(a.muon_momentum, b.muon_momentum);
    swap(a.ns_iters, b.ns_iters);
    swap(a.ns_eps, b.ns_eps);
    swap(a.ns_fro_norm, b.ns_fro_norm);
  }

  std::string getName() const override {
    std::ostringstream ss;
    if (!this->device_par) {
      ss << "MixedPrec[UNDEFINED]";
    } else {
      ss << "MixedPrec[" << this->device_par->getName() << "]";
    }
    if (use_muon) {
      ss << "+Muon";
    }
    return ss.str();
  };

  MixedPrecRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new MixedPrecRPUDevice<T>(x_size, d_size, *this, rng);
  };

  MixedPrecRPUDeviceMetaParameter<T> *clone() const override {
    return new MixedPrecRPUDeviceMetaParameter<T>(*this);
  };

  DeviceUpdateType implements() const override { return DeviceUpdateType::MixedPrec; };
  void printToStream(std::stringstream &ss) const override;
  void initialize() override;
};

template <typename T> class MixedPrecRPUDevice : public MixedPrecRPUDeviceBase<T> {

public:
  // constructor / destructor
  MixedPrecRPUDevice(int x_size, int d_size);
  MixedPrecRPUDevice(
      int x_size, int d_size, const MixedPrecRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  ~MixedPrecRPUDevice();

  MixedPrecRPUDevice(const MixedPrecRPUDevice<T> &);
  MixedPrecRPUDevice<T> &operator=(const MixedPrecRPUDevice<T> &);
  MixedPrecRPUDevice(MixedPrecRPUDevice<T> &&) noexcept;
  MixedPrecRPUDevice<T> &operator=(MixedPrecRPUDevice<T> &&) noexcept;

  friend void swap(MixedPrecRPUDevice<T> &a, MixedPrecRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<MixedPrecRPUDeviceBase<T> &>(a), static_cast<MixedPrecRPUDeviceBase<T> &>(b));

    swap(a.chi_, b.chi_);

    // Muon momentum buffer
    swap(a.m_, b.m_);

    // temporaries
    swap(a.qx_, b.qx_);
    swap(a.qd_, b.qd_);
    swap(a.qx_index_, b.qx_index_);

    // Newton–Schulz workspaces
    swap(a.ns_X_, b.ns_X_);
    swap(a.ns_A_, b.ns_A_);
    swap(a.ns_B_, b.ns_B_);
    swap(a.ns_O_, b.ns_O_);
  }

  MixedPrecRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<MixedPrecRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  MixedPrecRPUDevice<T> *clone() const override { return new MixedPrecRPUDevice<T>(*this); };

 
  bool onSetWeights(T **weights) override;

  // Chi access (already exists)
  void getChi(T *data) const override;
  void setChi(const T *data) override;

  void forwardUpdate(
      T **weights,
      const T lr,
      int i_row_start,
      const T *transfer_vec,
      const int n_vec,
      const bool trans) override;

  void doDirectVectorUpdate(
      T **weights,
      const T *x_input,
      const int x_inc,
      const T *d_input,
      const int d_inc,
      const T learning_rate,
      const int m_batch_info,
      const PulsedUpdateMetaParameter<T> &up) override;

protected:
  void populate(const MixedPrecRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

private:
  void initialize(int x_size, int d_size);
  void freeContainers();

  // Muon helper: compute O ≈ (M M^T)^(-1/2) M using Newton–Schulz iterations
  void computeMuonO(T *O, const T *M, int d_size, int x_size) const;

  // handled in base / this device
  T **chi_ = nullptr;

  // Muon momentum buffer (digital, full precision)
  T **m_ = nullptr;

  // temporary (quantization)
  std::vector<T> qx_;
  std::vector<T> qd_;
  std::vector<int> qx_index_;

  mutable std::vector<T> ns_X_;  // d*x
  mutable std::vector<T> ns_A_;  // d*d
  mutable std::vector<T> ns_B_;  // d*x
  mutable std::vector<T> ns_O_;  // d*x

};

} // namespace RPU
