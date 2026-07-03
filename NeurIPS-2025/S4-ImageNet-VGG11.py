# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Selective analog finetuning for pretrained VGG11-BN on ImageNet.

The default setup keeps all batch-norm layers digital and converts only:
1. VGG block 5 convolutions: ``features[22]`` and ``features[25]``
2. The final classifier layer ``fc3``: ``classifier[6]``

An ``fc3``-only analog mode is also supported for faster, stabler finetuning.
"""

import argparse
import json
import os
import sys
from enum import Enum
from pathlib import Path
from time import time

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

import aihwkit
from utils.logger import Logger

print("aihwkit path: ", aihwkit.__file__)
print("[Running]", " ".join(sys.argv))

from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    build_config,
    FloatingPointRPUConfig,
    SingleRPUConfig,
    UnitCellRPUConfig,
)
from aihwkit.simulator.configs.devices import (
    BufferedTransferCompound,
    ConstantStepDevice,
    DynamicTransferCompound,
    LinearStepDevice,
    SoftBoundsReferenceDevice,
)
from aihwkit.simulator.parameters.enums import (
    BoundManagementType,
    NoiseManagementType,
)
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.training import UpdateParameters


BLOCK5_ANALOG_MODULES = ("features.22", "features.25")
BLOCK5_ANALOG_CONVS = (22, 25)
SECOND_LAST_ANALOG_FC = 3
TARGET_ANALOG_FC = 6
BLOCK5_DIGITAL_BN = ("features.23", "features.26")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


parser = argparse.ArgumentParser(description="VGG11-BN ImageNet selective analog finetuning")

parser.add_argument("-SETTING", "--SETTING", type=str, default="Analog SGD")
parser.add_argument("-CUDA", "--CUDA", type=int, default=0)
parser.add_argument("-tau", "--tau", type=float, default=1.0)
parser.add_argument("-RPU", "--RPU", type=str, default="HfO2")
parser.add_argument("-SB-d2d", "--SB-d2d", type=float, default=-1.0)
parser.add_argument("-SB-dw_min_std", "--SB-dw_min_std", type=float, default=-1.0)
parser.add_argument("-res-state", "--res-state", type=float, default=None)
parser.add_argument("-res-gamma", "--res-gamma", type=float, default=-1.0)
parser.add_argument("--pref", action="store_true")
parser.add_argument("--reference-mean", type=float, default=0.4)
parser.add_argument("--reference-std", type=float, default=1.0)
parser.add_argument("-save", "--save-checkpoint", action="store_true")

parser.add_argument("--LR", "--lr", dest="LR", type=float, default=1e-3)
parser.add_argument("--omega", type=float, default=0.3)
parser.add_argument("--thres_scale", type=float, default=1.0)
parser.add_argument("--fast_lr_tt", type=float, default=0.01)
parser.add_argument("--transfer_lr_tt", type=float, default=0.5)
parser.add_argument("--scale_transfer_lr_tt", type=int, default=1)
parser.add_argument("--CONSTRUCTION_SEED", type=int, default=123)

parser.add_argument("--data-dir", type=str, default="data/ImageNet")
parser.add_argument("--train-dir", type=str, default=None)
parser.add_argument("--val-dir", type=str, default=None)
parser.add_argument("--weights-path", type=str, default=None)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--eval-batch-size", type=int, default=128)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--num-classes", type=int, default=-1)
parser.add_argument("--train-frac", type=float, default=1.0)
parser.add_argument("--val-frac", type=float, default=1.0)
parser.add_argument("--train-per-class", type=int, default=-1)
parser.add_argument("--val-per-class", type=int, default=-1)
parser.add_argument("--step-size", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--image-size", type=int, default=224)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--freeze-digital", action="store_true")
parser.add_argument("--no-pretrained", action="store_true")
parser.add_argument("--digital-only", action="store_true")
parser.add_argument(
    "--analog-target",
    type=str,
    default="block5-fc3",
    choices=["block5-fc3", "fc3", "fc2-fc3"],
)
parser.add_argument("--digital-optimizer", type=str, default="SGD", choices=["SGD", "AdamW"])
parser.add_argument("--weight-decay", type=float, default=0.0)

args = parser.parse_args()

setting = args.SETTING
tau = args.tau
RPU_NAME = args.RPU
EPOCHS = args.epochs

USE_CUDA = torch.cuda.is_available() and args.CUDA >= 0
DEVICE = torch.device(f"cuda:{args.CUDA}" if USE_CUDA else "cpu")
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
print("Using Device: ", DEVICE)


class opt_T(Enum):
    KIT_FP = 1
    KIT_ANALOG = 2


def get_opt_type(optimizer_str):
    fp_list = ["FP SGD", "FP SGDM"]
    analog_list = ["Analog SGD", "TT-v1", "TT-v2", "TT-v3", "TT-v4", "mp", "RL-v2", "RL-v3"]
    if optimizer_str in fp_list:
        return opt_T.KIT_FP
    if optimizer_str in analog_list:
        return opt_T.KIT_ANALOG
    raise ValueError(f"unknown algorithm type: {optimizer_str}")


opt_type = None if args.digital_only else get_opt_type(setting)


def get_active_analog_conv_indices():
    if args.analog_target == "block5-fc3":
        return BLOCK5_ANALOG_CONVS
    if args.analog_target in ("fc3", "fc2-fc3"):
        return ()
    raise ValueError(f"unknown analog target: {args.analog_target}")


def get_active_analog_fc_indices():
    if args.analog_target == "block5-fc3":
        return (TARGET_ANALOG_FC,)
    if args.analog_target == "fc3":
        return (TARGET_ANALOG_FC,)
    if args.analog_target == "fc2-fc3":
        return (SECOND_LAST_ANALOG_FC, TARGET_ANALOG_FC)
    raise ValueError(f"unknown analog target: {args.analog_target}")


def get_active_analog_modules():
    modules = []
    for idx in get_active_analog_conv_indices():
        modules.append(f"features.{idx}")
    for idx in get_active_analog_fc_indices():
        modules.append(f"classifier.{idx}")
    return tuple(modules)


def get_active_digital_bn_modules():
    if args.analog_target == "block5-fc3":
        return BLOCK5_DIGITAL_BN
    return ()

DEFAULT_NUMBER_OF_STATES = 4
if args.res_state is not None:
    num_of_states = args.res_state
else:
    num_of_states = DEFAULT_NUMBER_OF_STATES


def _to_serializable(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Enum):
        return x.name
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]

    try:
        import dataclasses

        if dataclasses.is_dataclass(x):
            return _to_serializable(dataclasses.asdict(x))
    except Exception:
        pass

    if hasattr(x, "__dict__"):
        dct = {k: v for k, v in x.__dict__.items() if not k.startswith("_")}
        return _to_serializable(dct)

    return str(x)


def save_run_config_json(log_dir, config, extra=None, filename="config.json"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": config.get("name", None),
        "grad_per_iter": config.get("grad_per_iter", None),
        "batch_size": config.get("batch_size", None),
        "rpu_config": config.get("rpu_config", None),
    }
    if extra:
        payload.update(extra)

    payload = _to_serializable(payload)
    (log_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True))


def get_RPU_device(device_name):
    dw_min = 2 * tau / num_of_states

    if device_name == "CS":
        return ConstantStepDevice()
    if device_name == "Softbounds":
        device = SoftBoundsReferenceDevice(
            dw_min=1,
            dw_min_dtod=0.02,
            dw_min_dtod_log_normal=False,
            dw_min_std=0.0,
            up_down=0.4,
            up_down_dtod=0.08,
            w_min=-1.0,
            w_max=1.0,
            w_min_dtod=0.0,
            w_max_dtod=0.0,
            reset=0.0,
            reset_dtod=0.0,
            reset_std=0.0,
            diffusion=0.0,
            diffusion_dtod=0.0,
            lifetime=0.0,
            lifetime_dtod=0.0,
            slope_up_dtod=0.0,
            slope_down_dtod=0.0,
            reference_mean=0.0,
            reference_std=0.0,
            subtract_symmetry_point=True,
        )
        if args.SB_dw_min_std > 0:
            device.dw_min_std = args.SB_dw_min_std
        if args.SB_d2d > 0:
            device.dw_min_dtod = args.SB_d2d
            device.w_max_dtod = args.SB_d2d
            device.w_min_dtod = args.SB_d2d
        return device
    if device_name == "Exp":
        from aihwkit.simulator.configs.devices import ExpStepDevice

        device = ExpStepDevice(dw_min=dw_min, w_max=tau, w_min=-tau, w_max_dtod=0, w_min_dtod=0)
        if args.res_gamma > 0:
            device.gamma_up = args.res_gamma
            device.gamma_down = args.res_gamma
        return device
    if device_name == "Pow":
        from aihwkit.simulator.configs.devices import PowStepDevice

        device = PowStepDevice(
            dw_min=dw_min,
            pow_gamma_dtod=0,
            w_max=tau,
            w_min=-tau,
            w_max_dtod=0,
            w_min_dtod=0,
        )
        if args.res_gamma > 0:
            device.pow_gamma = args.res_gamma
        return device
    if device_name == "LS":
        return LinearStepDevice(w_max_dtod=0.4)
    if device_name == "ReRamSB":
        from aihwkit.simulator.presets.devices import ReRamSBPresetDevice

        return ReRamSBPresetDevice()
    if device_name == "ReRamES":
        from aihwkit.simulator.presets.devices import ReRamESPresetDevice

        return ReRamESPresetDevice()
    if device_name == "EcRam":
        from aihwkit.simulator.presets.devices import EcRamPresetDevice

        return EcRamPresetDevice()
    if device_name == "EcRamMO":
        from aihwkit.simulator.presets.devices import EcRamMOPresetDevice

        return EcRamMOPresetDevice()
    if device_name == "HfO2":
        return SoftBoundsReferenceDevice(
            enforce_consistency=True,
            dw_min_dtod_log_normal=True,
            dw_min=0.4622,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            mult_noise=False,
            dw_min_dtod=0.7125,
            up_down_dtod=0.01,
            w_max_dtod=0.4295,
            w_min_dtod=0.5990,
            dw_min_std=0.2174,
            write_noise_std=0.5841,
            corrupt_devices_range=0.0100,
            corrupt_devices_prob=0.0,
            subtract_symmetry_point=True,
            reference_std=args.reference_std,
            reference_mean=args.reference_mean,
        )
    if device_name == "OM":
        # Mirror ReRamArrayOMPresetDevice, but keep the symmetry-point
        # parameters user-controllable from the CLI as in the HfO2 branch.
        return SoftBoundsReferenceDevice(
            enforce_consistency=True,
            dw_min_dtod_log_normal=True,
            dw_min=0.0949,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            mult_noise=False,
            dw_min_dtod=0.7829,
            up_down_dtod=0.01,
            w_max_dtod=0.3499,
            w_min_dtod=0.5695,
            dw_min_std=0.4158,
            write_noise_std=1.4113,
            corrupt_devices_range=0.0100,
            corrupt_devices_prob=0.0,
            subtract_symmetry_point=True,
            reference_std=args.reference_std,
            reference_mean=args.reference_mean,
        )
    if device_name == "PCM":
        from aihwkit.simulator.presets.devices import PCMPresetDevice

        return PCMPresetDevice()
    if device_name == "test":
        from aihwkit.simulator.configs.devices import PowStepDevice

        device = PowStepDevice(dw_min=0.001 * 2 ** args.res_gamma, w_max=tau, w_min=-tau)
        return device

    raise NotImplementedError(f"unknown RPU device: {device_name}")


def get_AnalogSGD_optimizer_generator(default_lr, *opt_args, **opt_kwargs):
    def _generator(params):
        return AnalogSGD(params, lr=default_lr, *opt_args, **opt_kwargs)

    return _generator


def get_config(config_name):
    if config_name == "FP SGD":
        rpu_config = FloatingPointRPUConfig()
        return {
            "name": "FP SGD",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "FP SGDM":
        rpu_config = FloatingPointRPUConfig()
        return {
            "name": "FPSGDM",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR, momentum=0.99),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "Analog SGD":
        rpu_config = SingleRPUConfig(device=get_RPU_device(RPU_NAME))
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        return {
            "name": "Analog SGD",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "TT-v1":
        rpu_config = build_config("ttv1", device=get_RPU_device(RPU_NAME), construction_seed=args.CONSTRUCTION_SEED)
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.n_reads_per_transfer = 1
        name = "TT-v1"
        if rpu_config.device.n_reads_per_transfer > 1:
            name += f"-st={rpu_config.device.n_reads_per_transfer}"
        return {
            "name": name,
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "TT-v2":
        rpu_config = UnitCellRPUConfig(
            device=BufferedTransferCompound(
                unit_cell_devices=[get_RPU_device(RPU_NAME), get_RPU_device(RPU_NAME)],
                transfer_forward=IOParameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=UpdateParameters(
                    desired_bl=1,
                    update_bl_management=False,
                    update_management=False,
                ),
                units_in_mbatch=True,
                construction_seed=args.CONSTRUCTION_SEED,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
            update=UpdateParameters(desired_bl=5),
        )
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = args.omega
        rpu_config.device.thres_scale = args.thres_scale
        rpu_config.device.fast_lr = args.fast_lr_tt
        rpu_config.device.transfer_lr = args.transfer_lr_tt
        rpu_config.device.scale_transfer_lr = bool(args.scale_transfer_lr_tt)
        return {
            "name": "TT-v2",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "TT-v4":
        rpu_config = UnitCellRPUConfig(
            device=DynamicTransferCompound(
                unit_cell_devices=[get_RPU_device(RPU_NAME), get_RPU_device(RPU_NAME)],
                transfer_forward=IOParameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=UpdateParameters(
                    desired_bl=1,
                    update_bl_management=False,
                    update_management=False,
                ),
                in_chop_prob=0.1,
                units_in_mbatch=True,
                auto_scale=False,
                construction_seed=args.CONSTRUCTION_SEED,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
            update=UpdateParameters(desired_bl=5),
        )
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.scale_fast_lr = False
        rpu_config.device.transfer_lr = 0.5
        rpu_config.device.scale_transfer_lr = True
        rpu_config.device.auto_granularity = 1000
        return {
            "name": "TT-v4",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "RL-v2":
        rpu_config = UnitCellRPUConfig(
            device=DynamicTransferCompound(
                unit_cell_devices=[get_RPU_device(RPU_NAME), get_RPU_device(RPU_NAME)],
                transfer_forward=IOParameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=UpdateParameters(
                    desired_bl=1,
                    update_bl_management=False,
                    update_management=False,
                ),
                in_chop_prob=0.1,
                units_in_mbatch=True,
                auto_scale=False,
                construction_seed=args.CONSTRUCTION_SEED,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
            update=UpdateParameters(desired_bl=5),
        )
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        rpu_config.device.fast_lr = 0.1
        rpu_config.device.scale_fast_lr = False
        if num_of_states == 64:
            rpu_config.device.transfer_lr = 0.1 + 1 / num_of_states
        else:
            rpu_config.device.transfer_lr = 0.2
        rpu_config.device.scale_transfer_lr = False
        rpu_config.device.auto_granularity = 1000
        return {
            "name": "RL-v2",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "RL-v3":
        rpu_config = UnitCellRPUConfig(
            device=DynamicTransferCompound(
                unit_cell_devices=[get_RPU_device(RPU_NAME), get_RPU_device(RPU_NAME)],
                transfer_forward=IOParameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=UpdateParameters(
                    desired_bl=1,
                    update_bl_management=False,
                    update_management=False,
                ),
                in_chop_prob=0.1,
                units_in_mbatch=True,
                auto_scale=False,
                construction_seed=args.CONSTRUCTION_SEED,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
            update=UpdateParameters(desired_bl=5),
        )
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        rpu_config.device.buffer_as_momentum = True
        rpu_config.device.momentum = 0.9
        rpu_config.device.fast_lr = 1.0
        rpu_config.device.scale_fast_lr = True
        if num_of_states == 64:
            rpu_config.device.transfer_lr = 0.1 + 1 / num_of_states
        else:
            rpu_config.device.transfer_lr = 0.099 + 1 / num_of_states
        rpu_config.device.scale_transfer_lr = False
        rpu_config.device.auto_granularity = 10000
        return {
            "name": "RL-v3",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "TT-v3":
        rpu_config = build_config("ttv3", device=get_RPU_device(RPU_NAME), construction_seed=args.CONSTRUCTION_SEED)
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.scale_fast_lr = False
        rpu_config.device.transfer_lr = 1.0
        rpu_config.device.scale_transfer_lr = True
        rpu_config.device.auto_granularity = 1000
        rpu_config.device.in_chop_prob = 0.0
        rpu_config.device.out_chop_prob = 0.0
        rpu_config.device.auto_scale = False
        return {
            "name": "TT-v3",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    if config_name == "mp":
        rpu_config = build_config("mp", device=get_RPU_device(RPU_NAME), construction_seed=args.CONSTRUCTION_SEED)
        return {
            "name": "mp",
            "rpu_config": rpu_config,
            "optimizer_cls": get_AnalogSGD_optimizer_generator(args.LR),
            "grad_per_iter": 1,
            "batch_size": args.batch_size,
        }

    raise NotImplementedError(f"unknown config: {config_name}")


def get_digital_baseline_config():
    if args.digital_optimizer == "SGD":
        optimizer_cls = lambda params: optim.SGD(
            params,
            lr=args.LR,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.digital_optimizer == "AdamW":
        optimizer_cls = lambda params: optim.AdamW(
            params,
            lr=args.LR,
            weight_decay=args.weight_decay,
            amsgrad=True,
        )
    else:
        raise ValueError(f"unknown digital optimizer: {args.digital_optimizer}")

    return {
        "name": f"Digital-{args.digital_optimizer}",
        "rpu_config": None,
        "optimizer_cls": optimizer_cls,
        "grad_per_iter": 1,
        "batch_size": args.batch_size,
    }


def set_pref_from_config(rpu_config, enable):
    if enable:
        os.environ["AIHWKIT_PREF_ON"] = "1"
        os.environ["AIHWKIT_PREF_GAMMA"] = str(float(0.1))
        print(f"[PY] Pref ON, gamma={os.environ['AIHWKIT_PREF_GAMMA']}")
        return

    os.environ.pop("AIHWKIT_PREF_ON", None)
    os.environ.pop("AIHWKIT_PREF_GAMMA", None)
    print("[PY] Pref OFF")


def maybe_subset_dataset(dataset, frac, seed):
    if frac >= 1.0:
        return dataset
    if frac <= 0.0:
        raise ValueError(f"dataset fraction must be in (0, 1], got {frac}")

    num_items = max(1, int(len(dataset) * frac))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_items].tolist()
    return Subset(dataset, indices)


def balanced_subset_dataset(dataset, samples_per_class, seed, split_name):
    if samples_per_class <= 0:
        return dataset

    if not hasattr(dataset, "targets"):
        raise TypeError(f"{split_name} dataset must expose .targets for class-balanced sampling")

    class_to_indices = {}
    for sample_idx, class_idx in enumerate(dataset.targets):
        class_to_indices.setdefault(int(class_idx), []).append(sample_idx)

    if len(class_to_indices) == 0:
        raise ValueError(f"{split_name} dataset has no classes")

    min_count = min(len(indices) for indices in class_to_indices.values())
    if samples_per_class > min_count:
        raise ValueError(
            f"Requested --{split_name}-per-class={samples_per_class}, "
            f"but the smallest class in {split_name} has only {min_count} samples"
        )

    generator = torch.Generator().manual_seed(seed)
    selected_indices = []
    for class_idx in sorted(class_to_indices):
        indices = class_to_indices[class_idx]
        chosen = torch.randperm(len(indices), generator=generator)[:samples_per_class].tolist()
        selected_indices.extend(indices[idx] for idx in chosen)

    shuffle_order = torch.randperm(len(selected_indices), generator=generator).tolist()
    selected_indices = [selected_indices[idx] for idx in shuffle_order]
    return Subset(dataset, selected_indices)


def apply_subset_policy(dataset, frac, per_class, seed, split_name):
    if per_class > 0:
        if frac < 1.0:
            raise ValueError(
                f"Use either --{split_name}-frac or --{split_name}-per-class, not both"
            )
        subset = balanced_subset_dataset(dataset, per_class, seed, split_name)
        desc = f"balanced-{per_class}/class"
        return subset, desc

    subset = maybe_subset_dataset(dataset, frac, seed)
    if frac < 1.0:
        desc = f"random-frac={frac:g}"
    else:
        desc = "full"
    return subset, desc


def resolve_imagenet_dirs():
    train_dir = args.train_dir or os.path.join(args.data_dir, "train")
    val_dir = args.val_dir or os.path.join(args.data_dir, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val directory not found: {val_dir}")
    return train_dir, val_dir


def create_dataloaders():
    train_dir, val_dir = resolve_imagenet_dirs()

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_set_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set_full = datasets.ImageFolder(val_dir, transform=val_transform)
    train_set, train_subset_desc = apply_subset_policy(
        train_set_full,
        args.train_frac,
        args.train_per_class,
        args.seed,
        "train",
    )
    val_set, val_subset_desc = apply_subset_policy(
        val_set_full,
        args.val_frac,
        args.val_per_class,
        args.seed + 1,
        "val",
    )

    pin_memory = USE_CUDA
    persistent_workers = args.workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    subset_info = {
        "train_desc": train_subset_desc,
        "val_desc": val_subset_desc,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "train_full_size": len(train_set_full),
        "val_full_size": len(val_set_full),
    }
    return train_loader, val_loader, len(train_set_full.classes), train_dir, val_dir, subset_info


def load_vgg11_bn(use_pretrained):
    if hasattr(models, "VGG11_BN_Weights"):
        weights = models.VGG11_BN_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.vgg11_bn(weights=weights)
    else:
        model = models.vgg11_bn(pretrained=use_pretrained)

    if args.weights_path:
        checkpoint = torch.load(args.weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith("module.") else key
            cleaned_state_dict[new_key] = value
        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"[Weights] Loaded local checkpoint: {args.weights_path}")
        if missing:
            print(f"[Weights] Missing keys: {missing}")
        if unexpected:
            print(f"[Weights] Unexpected keys: {unexpected}")

    return model


def initialize_classifier(linear_layer):
    nn.init.normal_(linear_layer.weight, 0.0, 0.01)
    if linear_layer.bias is not None:
        nn.init.constant_(linear_layer.bias, 0.0)


def prepare_digital_vgg11_bn(model, num_classes):
    if not isinstance(model.classifier[TARGET_ANALOG_FC], nn.Linear):
        raise TypeError("Unexpected VGG11-BN classifier layout: fc3 is not nn.Linear")

    if model.classifier[TARGET_ANALOG_FC].out_features != num_classes:
        model.classifier[TARGET_ANALOG_FC] = nn.Linear(
            model.classifier[TARGET_ANALOG_FC].in_features,
            num_classes,
        )
        initialize_classifier(model.classifier[TARGET_ANALOG_FC])

    return model


def convert_vgg11_selected_modules_to_analog(model, rpu_config, num_classes):
    analog_conv_indices = get_active_analog_conv_indices()
    analog_fc_indices = get_active_analog_fc_indices()
    if args.analog_target == "block5-fc3":
        if not isinstance(model.features[22], nn.Conv2d) or not isinstance(model.features[25], nn.Conv2d):
            raise TypeError("Unexpected VGG11-BN feature layout: block 5 conv indices are not Conv2d")
        if not isinstance(model.features[23], nn.BatchNorm2d) or not isinstance(model.features[26], nn.BatchNorm2d):
            raise TypeError("Unexpected VGG11-BN feature layout: block 5 BN indices are not BatchNorm2d")
    for idx in analog_fc_indices:
        if not isinstance(model.classifier[idx], nn.Linear):
            raise TypeError(f"Unexpected VGG11-BN classifier layout: classifier[{idx}] is not nn.Linear")

    if model.classifier[TARGET_ANALOG_FC].out_features != num_classes:
        model.classifier[TARGET_ANALOG_FC] = nn.Linear(
            model.classifier[TARGET_ANALOG_FC].in_features,
            num_classes,
        )
        initialize_classifier(model.classifier[TARGET_ANALOG_FC])

    for idx in analog_conv_indices:
        model.features[idx] = AnalogConv2d.from_digital(model.features[idx], rpu_config)

    for idx in analog_fc_indices:
        model.classifier[idx] = AnalogLinear.from_digital(
            model.classifier[idx],
            rpu_config,
        )

    return model


def freeze_all_but_analog_targets(model):
    target_analog_modules = get_active_analog_modules()
    for name, param in model.named_parameters():
        param.requires_grad = any(
            name == prefix or name.startswith(prefix + ".") for prefix in target_analog_modules
        )


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE, non_blocking=USE_CUDA)
            labels = labels.to(DEVICE, non_blocking=USE_CUDA)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def build_paths(config):
    dataset_name = "ImageNet-VGG11-BN"
    if args.digital_only:
        model_scope = "pretrained-digital-baseline"
    elif args.analog_target == "fc3":
        model_scope = "pretrained-fc3-analog"
    elif args.analog_target == "fc2-fc3":
        model_scope = "pretrained-fc2-fc3-analog"
    else:
        model_scope = "pretrained-block5-fc3-analog"

    alg_name = config["name"]
    alg_name += f"-lr={args.LR:g}"
    alg_name += f"-step={args.step_size}"
    alg_name += f"-lrgamma={args.gamma:g}"
    alg_name += f"-ep={args.epochs}"
    alg_name += f"-bs={args.batch_size}"
    if args.freeze_digital:
        alg_name += "-freeze-digital"
    if args.train_per_class > 0:
        alg_name += f"-tpc={args.train_per_class}"
    elif args.train_frac < 1.0:
        alg_name += f"-tfrac={args.train_frac:g}"
    if args.val_per_class > 0:
        alg_name += f"-vpc={args.val_per_class}"
    elif args.val_frac < 1.0:
        alg_name += f"-vfrac={args.val_frac:g}"
    if args.no_pretrained:
        alg_name += "-scratch"
    if not args.digital_only:
        alg_name += "-prefon" if args.pref else "-prefoff"
    if not args.digital_only and args.analog_target != "block5-fc3":
        alg_name += f"-target={args.analog_target}"
    if not args.digital_only and args.RPU in ("HfO2", "OM"):
        alg_name += f"-rmean={args.reference_mean:g}"
        alg_name += f"-rstd={args.reference_std:g}"

    if args.digital_only:
        rpu_suffix = "Digital"
    elif opt_type is opt_T.KIT_ANALOG:
        rpu_suffix = args.RPU
        if args.RPU in ("Pow", "Exp") and args.res_gamma > 0:
            rpu_suffix += f"-gamma={args.res_gamma:.2f}"
        if args.res_state is not None:
            rpu_suffix += f"-state={int(args.res_state)}"
    else:
        rpu_suffix = "FloatingPoint"

    path_name = os.path.join(dataset_name, model_scope, rpu_suffix)
    checkpoint_dir = f"checkpoints/{path_name}"
    checkpoint_path = f"{checkpoint_dir}/{alg_name}.pth"
    log_path = f"runs/{path_name}/{alg_name}"
    return dataset_name, path_name, alg_name, checkpoint_dir, checkpoint_path, log_path


def main():
    target_analog_modules = get_active_analog_modules()
    target_digital_bn_modules = get_active_digital_bn_modules()

    if args.digital_only:
        config = get_digital_baseline_config()
        rpu_config = None
        print("[Mode] Pure digital baseline enabled. Analog conversion is skipped.")
    else:
        config = get_config(setting)
        set_pref_from_config(config["rpu_config"], enable=args.pref)
        rpu_config = config["rpu_config"]

    train_loader, val_loader, detected_num_classes, train_dir, val_dir, subset_info = create_dataloaders()
    if args.num_classes > 0:
        num_classes = args.num_classes
        if num_classes != detected_num_classes:
            print(
                f"[Warning] --num-classes={num_classes} but ImageFolder detected "
                f"{detected_num_classes} classes."
            )
    else:
        num_classes = detected_num_classes

    model = load_vgg11_bn(use_pretrained=not args.no_pretrained)
    if args.digital_only:
        model = prepare_digital_vgg11_bn(model, num_classes=num_classes)
    else:
        model = convert_vgg11_selected_modules_to_analog(model, rpu_config, num_classes=num_classes)

    if args.freeze_digital:
        freeze_all_but_analog_targets(model)

    if USE_CUDA:
        model = model.to(DEVICE)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = config["optimizer_cls"](params)
    if isinstance(optimizer, AnalogSGD):
        optimizer.regroup_param_groups(model)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    total_params, trainable_params = count_parameters(model)
    print(f"[Dataset] train_dir={train_dir}")
    print(f"[Dataset] val_dir={val_dir}")
    print(f"[Dataset] num_classes={num_classes}")
    print(
        f"[Subset] train={subset_info['train_desc']} "
        f"({subset_info['train_size']}/{subset_info['train_full_size']})"
    )
    print(
        f"[Subset] val={subset_info['val_desc']} "
        f"({subset_info['val_size']}/{subset_info['val_full_size']})"
    )
    print(f"[Model] pretrained={not args.no_pretrained}")
    if args.digital_only:
        print("[Analog] converted modules=[]")
        print("[Digital] model stays fully digital")
    else:
        print(f"[Pref] enabled={args.pref}")
        print(f"[Analog] target={args.analog_target}")
        print(f"[Analog] converted modules={list(target_analog_modules)}")
        print(f"[Digital] kept BN modules={list(target_digital_bn_modules)}")
    print(f"[Params] total={total_params} trainable={trainable_params}")

    dataset_name, path_name, alg_name, checkpoint_dir, checkpoint_path, log_path = build_paths(config)
    logger = Logger(log_path)
    text_log_path = logger.get_text_log_path()
    json_path = text_log_path[:-4] + ".json"
    save_run_config_json(
        log_path,
        config,
        extra={
            "dataset_name": dataset_name,
            "path_name": path_name,
            "alg_name": alg_name,
            "checkpoint_path": checkpoint_path,
            "train_dir": train_dir,
            "val_dir": val_dir,
            "num_classes": num_classes,
            "subset_info": subset_info,
            "target_analog_modules": [] if args.digital_only else list(target_analog_modules),
            "target_digital_bn_modules": [] if args.digital_only else list(target_digital_bn_modules),
            "analog_target": None if args.digital_only else args.analog_target,
            "digital_only": args.digital_only,
            "pref_enabled": False if args.digital_only else args.pref,
            "RPU_NAME": RPU_NAME,
            "num_of_states": num_of_states,
            "tau": tau,
            "args": vars(args),
        },
        filename=Path(json_path).name,
    )

    if not args.digital_only:
        logger.write(0, f"[Config] pref={'on' if args.pref else 'off'}", {})

    start_time = time()
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    logger.write(
        0,
        f"Epoch 0 - Training loss: --------   Val Accuracy: {val_accuracy:.4f}",
        {
            "Loss/val": val_loss,
            "Accuracy/val": val_accuracy,
        },
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=USE_CUDA)
            labels = labels.to(DEVICE, non_blocking=USE_CUDA)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

        scheduler.step()

        train_loss = running_loss / max(running_samples, 1)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        logger.write(
            epoch,
            f"Epoch {epoch} - Training loss: {train_loss:.6f}   Val Accuracy: {val_accuracy:.4f}",
            {
                "Loss/train": train_loss,
                "Loss/val": val_loss,
                "Accuracy/val": val_accuracy,
                "State/lr": scheduler.get_last_lr()[0],
            },
        )

    print(f"Finished Training: {path_name}/{alg_name}")
    print("Training Time (s) =", time() - start_time)

    if args.save_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_checkpoint(model, optimizer, scheduler, EPOCHS, checkpoint_path)
        print(f"Save checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
