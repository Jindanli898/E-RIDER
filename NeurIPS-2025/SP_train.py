# import os
# from time import time
# from dataclasses import dataclass, field
# # Imports from PyTorch.
# import torch
# torch.autograd.set_detect_anomaly(True)
# import random
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset
# from torch import nn
# from torch.optim.lr_scheduler import StepLR
# from torchvision import datasets, transforms
# import numpy as np
# import sys
# from torch.optim.lr_scheduler import LambdaLR
# from utils.logger import Logger
# from collections import deque
# # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# # AIHWKIT_SRC = os.path.join(CURRENT_DIR, 'aihwkit', 'src')
# # sys.path.insert(0, AIHWKIT_SRC)
# # # For warm start
# import aihwkit
# print('aihwkit path: ', aihwkit.__file__)
# from aihwkit.nn import AnalogLinear, AnalogSequential, AnalogConv2d
# from aihwkit.optim import AnalogSGD
# from aihwkit.simulator.rpu_base import cuda
# import aihwkit.simulator.rpu_base.devices as dev
# from aihwkit.simulator.parameters.io import IOParameters
# from aihwkit.nn.conversion import convert_to_analog, convert_to_digital
# from aihwkit.simulator.configs import (
#     build_config,
#     UnitCellRPUConfig,
#     DigitalRankUpdateRPUConfig,
#     FloatingPointRPUConfig,
#     SingleRPUConfig,
#     UpdateParameters,
# )
# from aihwkit.simulator.configs.devices import (
#     FloatingPointDevice,
#     ConstantStepDevice,
#     VectorUnitCell,
#     LinearStepDevice,
#     SoftBoundsDevice,
#     SoftBoundsReferenceDevice,
#     TransferCompound,
#     MixedPrecisionCompound,
#     BufferedTransferCompound,
#     ChoppedTransferCompound,
#     DynamicTransferCompound,
# )
# import argparse
# from enum import Enum
# from aihwkit.simulator.parameters.inference import DriftParameter
# parser = argparse.ArgumentParser(description="A simple command-line argument example")

# # Add command line arguments
# parser.add_argument('-SETTING', '--SETTING', type=str, help="", default='FP SGD')
# parser.add_argument('-BATCH_SIZE', '--BATCH_SIZE', type=int, help="", default='8')
# parser.add_argument('-CUDA', '--CUDA', type=int, help="", default=-1)
# parser.add_argument('-tau', '--tau', type=float, help="", default=1)
# parser.add_argument('-TTAWDC', '--TTv1-active-weight-decay-count', type=int, help="", default=0)
# parser.add_argument('-TTAWDP', '--TTv1-active_weight_decay_probability', type=float, help="", default=0)
# parser.add_argument('-save', '--save-checkpoint', action='store_true')
# parser.add_argument('-Tcolumn', '--Tcolumn', type=int, help="", default='1')
# parser.add_argument('-ns', '--ns', type=float, help="", default='1')
# parser.add_argument('-sigma', '--sigma', type=float, help="", default='0.3')
# parser.add_argument('-gamma', '--gamma', type=float, help="", default='0')
# parser.add_argument('-Wmax', '--Wmax', type=float, help="", default='1')
# parser.add_argument('-dwmin', '--dwmin', type=float, help="", default='0.1')
# # IO precision and noise parameters
# parser.add_argument('--io_inp_res_bit', type=float, default='7')
# parser.add_argument('--io_out_res_bit', type=float, default='9')
# parser.add_argument('--io_inp_noise', type=float, default='0.0')
# parser.add_argument('--io_out_noise', type=float, default='0.0')
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# parser.add_argument('--io_perfect_forward', type=str2bool, default=True)
# parser.add_argument('--io_perfect_backward', type=str2bool, default=True)
# checkpoint_path = "/home/jindan/Desktop/analog/checkpoints/MNIST-CNN/Softbounds/TT-v1-tile=6-alg2--6-state4-dataset-tau0.5.pth"
# import os

# print("Checking path:", checkpoint_path)
# print("Exists?", os.path.exists(checkpoint_path))

# args = parser.parse_args()
# setting = args.SETTING

# # Check device
# USE_CUDA = 0
# if cuda.is_compiled() and args.CUDA >= 0:
#     USE_CUDA = 1
# DEVICE = torch.device(f"cuda:{args.CUDA}" if USE_CUDA else "cpu")
# print('Using Device: ', DEVICE)


# # Path where the datasets will be stored.
# PATH_DATASET = os.path.join("data")

# # Training parameters.
# EPOCHS = 200
# N_CLASSES = 10

# tau = args.tau
# # DEVICE_NAME = 'PCM'
# # DEVICE_NAME = 'HfO2'
# # DEVICE_NAME = 'OM'
# DEVICE_NAME = 'Softbounds'
# # DEVICE_NAME = 'RRAM-offset'

# lr = 0.05


# def get_model_size(model, input):
#     from thop import profile
#     from thop import clever_format
#     macs, params = profile(model, inputs=(input, ))
#     macs, params = clever_format([macs, params], "%.3f")
#     print(macs, params)
#     raise SystemExit
#     return macs, params

# def get_device(device_name='CS',
#                reference_mean: float = 0.0,
#                reference_std: float = 0.0,
#                subtract_sp: bool = True):
#     """Return a device meta config. Softbounds uses SoftBoundsReferenceDevice."""
#     if device_name == 'CS':
#         return ConstantStepDevice()
#     elif device_name == 'Softbounds':
#         return SoftBoundsReferenceDevice(
#             dw_min=0.001,
#             dw_min_dtod=0.02,
#             dw_min_dtod_log_normal=False,
#             dw_min_std=0.0,
#             up_down=0.4,
#             up_down_dtod=0.08,
#             w_min=-1.0,
#             w_max=1.0,
#             w_min_dtod=0.0,
#             w_max_dtod=0.0,
#             reset=0.0,
#             reset_dtod=0.0,
#             reset_std=0.0,
#             diffusion=0.0,
#             diffusion_dtod=0.0,
#             lifetime=0.0,
#             lifetime_dtod=0.0,
#             slope_up_dtod=0.0,
#             slope_down_dtod=0.0,
#             reference_mean=reference_mean,
#             reference_std=reference_std,
#             subtract_symmetry_point=subtract_sp
#         )
#     else:
#         raise NotImplementedError

# # --------- Helper: read  w_reference from tile ----------
# def get_w_reference_from_tile(tile) -> np.ndarray:
#     """Read w_reference (true SP) from hidden parameters and flatten to 1D."""
#     hidden = tile.get_hidden_parameters()
#     ref_key = None
#     for name in hidden.keys():
#         if "reference" in name.lower():
#             ref_key = name
#             break
#     if ref_key is None:
#         raise RuntimeError("Could not find a hidden parameter containing 'reference' in tile.")
#     w_ref_tensor = hidden[ref_key]  # shape [d_size, x_size]
#     w_ref = w_ref_tensor.detach().cpu().numpy().ravel()
#     return w_ref


# from aihwkit.simulator.tiles import AnalogTile
# import math
# def calibrate_sp_offset_unitcell(
#     n_pulses: int = 2000,
#     n_rows: int = 512,
#     n_cols: int = 512,
# ) -> tuple[float, float]:
#     """
#     Calibrate SP mismatch using two UnitCell SoftBounds tiles:

#       - tile_true: subtract_symmetry_point=True, reference_mean=0, reference_std=0
#           --> w_reference = a*_{ij}  (theoretical / mathematical “true” SP samples)
#       - tile_pulse: subtract_symmetry_point=False, reference_mean=0, reference_std=0
#           --> approximate SP samples found via a finite number of pulses

#     Then define μ_r and σ_r using the difference between the two distributions:
#         μ_r = mean(SP_pulse) - mean(SP_true)
#         σ_r = std(SP_pulse)  - std(SP_true)    (or use sqrt(σ_pulse^2 - σ_true^2))

#     Returns:
#         mu_r, sigma_r
#     """

#     # ------------- 1) True SP distribution: tile_true (subtract_sp = True) -------------
#     softbounds_true = get_device(
#         "Softbounds",
#         reference_mean=0.0,
#         reference_std=0.0,
#         subtract_sp=True       # include sp in w_reference
#     )
#     rpu_config_true = UnitCellRPUConfig(
#         device=softbounds_true,
#         update=UpdateParameters(
#             desired_bl=1,
#             update_bl_management=False,
#             update_management=False
#         )
#     )
#     tile_true = AnalogTile(n_rows, n_cols, rpu_config_true)

#     # Read out w_reference = a*_{ij} from tile_true
#     hidden_true = tile_true.get_hidden_parameters()
#     if isinstance(hidden_true, dict):
#         ref_key = None
#         for name in hidden_true.keys():
#             if "reference" in name.lower():
#                 ref_key = name
#                 break
#         if ref_key is None:
#             raise RuntimeError("Could not find a hidden parameter containing 'reference'.")
#         w_ref_tensor = hidden_true[ref_key]
#     else:
#         raise RuntimeError(
#             f"Expected hidden parameters to be a dict for UnitCell tile, got {type(hidden_true)}."
#         )

#     sp_true = w_ref_tensor.detach().cpu().numpy().ravel()
#     mu_true = float(sp_true.mean())
#     sigma_true = float(sp_true.std())

#     print("---- True SP distribution (from w_reference, subtract_sp=True) ----")
#     print(f"mu_true     = {mu_true:.6f}")
#     print(f"sigma_true  = {sigma_true:.6f}")

#     # ------------- 2) Approximate SP distribution: tile_pulse (subtract_sp = False) -------------
#     softbounds_pulse = get_device(
#         "Softbounds",
#         reference_mean=0.0,
#         reference_std=0.0,
#         subtract_sp=False      # do not shift by symmetry point when estimating SP via pulses
#     )
#     rpu_config_pulse = UnitCellRPUConfig(
#         device=softbounds_pulse,
#         update=UpdateParameters(
#             desired_bl=1,
#             update_bl_management=False,
#             update_management=False
#         )
#     )
#     tile_pulse = AnalogTile(n_rows, n_cols, rpu_config_pulse)

#     # Initialize weights to -0.5
#     w_init = torch.full((n_rows, n_cols), -0.5, dtype=torch.float32)
#     tile_pulse.set_weights(w_init, torch.empty(0))

#     # Apply alternating pulses over the full matrix
#     x = torch.ones(n_cols, dtype=torch.float32)
#     d_pos = torch.ones(n_rows, dtype=torch.float32)
#     d_neg = -torch.ones(n_rows, dtype=torch.float32)

#     for t in range(n_pulses):
#         d = d_pos if (t % 2 == 0) else d_neg
#         tile_pulse.update(x, d)

#     w_pulse, _ = tile_pulse.get_weights()
#     sp_approx = w_pulse.detach().cpu().numpy().ravel()

#     mu_pulse = float(sp_approx.mean())
#     sigma_pulse = float(sp_approx.std())

#     print("---- Approx SP distribution (from pulses, subtract_sp=False) ----")
#     print(f"mu_pulse    = {mu_pulse:.6f}")
#     print(f"sigma_pulse = {sigma_pulse:.6f}")

#     # ------------- 3) Define μ_r and σ_r using the “difference of distributions” -------------
#     # Mean difference: offset of the approximate distribution relative to the true distribution
#     mu_r = mu_pulse - mu_true

#     # Std difference: here we follow your paper intuition and define it as "std difference"
#     # If you want strict physical correspondence to variance composition, use:
#     #   sqrt(max(0, σ_pulse^2 - σ_true^2))
#     sigma_r = sigma_pulse - sigma_true

#     print("\n========== SP offset calibration (UnitCell, distribution-wise) ==========")
#     print(f"n_pulses          = {n_pulses}")
#     print(f"mu_r   (mean diff)     = {mu_r:.6f}")
#     print(f"sigma_r(std diff)      = {sigma_r:.6f}")
#     print("=====================================================================\n")

#     return mu_r, sigma_r


# def load_images(fraction=0.1, seed=None):
#     """Load a random fraction of MNIST images for training and validation."""
#     transform = transforms.Compose([transforms.ToTensor()])

#     # Load full datasets
#     train_set_full = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
#     val_set_full = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)

#     # Compute subset sizes
#     num_train = int(len(train_set_full) * fraction)
#     num_val = int(len(val_set_full) * fraction)

#     # Optionally fix random seed for reproducibility
#     if seed is not None:
#         random.seed(seed)

#     # Randomly sample subset indices
#     train_indices = random.sample(range(len(train_set_full)), num_train)
#     val_indices = random.sample(range(len(val_set_full)), num_val)

#     # Create subsets
#     train_subset = Subset(train_set_full, train_indices)
#     val_subset = Subset(val_set_full, val_indices)

#     # Wrap into DataLoaders
#     train_data = DataLoader(train_subset, batch_size=args.BATCH_SIZE, shuffle=True)
#     validation_data = DataLoader(val_subset, batch_size=args.BATCH_SIZE, shuffle=True)

#     return train_data, validation_data

# def create_analog_network(rpu_config):
#     """Return a LeNet5 inspired analog model."""
#     channel = [16, 32, 512, 128]
#     model = AnalogSequential(
#         AnalogConv2d(
#             in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=rpu_config
#         ),
#         nn.Tanh(),
#         nn.MaxPool2d(kernel_size=2),
#         AnalogConv2d(
#             in_channels=channel[0],
#             out_channels=channel[1],
#             kernel_size=5,
#             stride=1,
#             rpu_config=rpu_config,
#         ),
#         nn.Tanh(),
#         nn.MaxPool2d(kernel_size=2),
#         nn.Tanh(),
#         nn.Flatten(),
#         AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
#         nn.Tanh(),
#         AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=rpu_config),
#         nn.LogSoftmax(dim=1),
#     )

#     if USE_CUDA:
#         model.cuda(DEVICE)
#     return model
# def save_checkpoint(model, optimizer, scheduler, epoch, filename):
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#     }
#     torch.save(checkpoint, filename)

# triggered_count = 0
# loss_history = []

# def train(model, train_set, config, logger, checkpoint_path):
#     """Train the network."""
#     optimizer_cls = config['optimizer_cls']
#     classifier = nn.NLLLoss()
#     optimizer = optimizer_cls(model.parameters())

#     def lr_lambda(epoch):
#         return 0.1 ** (epoch // 35)
#     scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

   
#     # aggresive
#     def aggressive_plateau(history, threshold=0.0001):
#         if len(history) < 2:
#             return False
#         delta = history[-2] - history[-1]
#         plateau = -delta > threshold
#         print(f"[Aggressive] Δ={delta:.6f}, plateau={plateau}")
#         return plateau

#     # smooth
#     def smooth_plateau(history, threshold=-0.01, window=5, max_violations=2):
#         if len(history) < window + 1:
#             return False
#         recent = history[-(window + 1):]
#         violations = 0

#         for i in range(window):  
#             delta = recent[i+1] - recent[i]
#             if delta >= threshold:
#                 violations += 1
#             print(f"Δ = {delta:.6f}, upward violation = {delta > threshold}")
#         return violations >= max_violations


#     def trigger_tile_switch_by_plateau(model, loss_history, aggressive_tile_count=4): 
#         print("\n[Tile switch check]")
#         global triggered_count
#         any_triggered = False  # track if any tile was actually triggered this round

#         for i, (name, module) in enumerate(model.named_modules()):
#             if hasattr(module, "analog_module"):
#                 tile = module.analog_module.tile
#                 if triggered_count < aggressive_tile_count:
#                     plateau = aggressive_plateau(loss_history)
#                 else:
#                     plateau = smooth_plateau(loss_history)
#                 tile.set_flags(plateau)
#                 print(f"[Tile {i}] {name}: trigger_tile_switch_flag = {plateau}")
#                 if plateau:
#                     any_triggered = True

#         if any_triggered:
#             triggered_count += 1
#             loss_history.clear()  # Reset history to wait for new tile to accumulate fresh stats
#             print("Tile switch triggered → Resetting loss history.")
#             print("Current loss history:", loss_history)  
#         print(f"\n=> Total triggered tiles = {triggered_count}")
#         return triggered_count

#     # --- Load checkpoint if exists ---
#     if checkpoint_path and os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint.get('epoch', 0) + 1  # resume from next epoch
#         # logger.info(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
#     else:
#         print("No checkpoint found. Training from scratch.")
#     for epoch in range(1, EPOCHS + 1):
#         total_loss = 0
#         for images, labels in train_set:
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             optimizer.zero_grad()
#             output = model(images)
#             loss = classifier(output, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         scheduler.step()
#         train_loss = total_loss / len(train_set)
#         loss_history.append(train_loss)
#         # trigger_tile_switch_by_plateau(model, loss_history)
#         # For warm start
#         test_loss, test_accuracy = test_evaluation(model, validation_dataset)

#         log_str = f"Epoch {epoch} - Training loss: {train_loss:.6f}   Test Accuracy: {test_accuracy:.4f}"
        

#         logger.write(epoch, log_str, {
#             "Loss/train": train_loss,
#             "Loss/test": test_loss,
#             "Accuracy/test": test_accuracy,
#             "State/lr": scheduler.get_last_lr()[0],
#             # "Tile/switch": float(plateau_triggered),  # optional: also track in tensorboard
#         })

#         if args.save_checkpoint:
#             alg_name = config['name']
#             alg_name += f'-tile={num_tile}'
#             # alg_name += f'-scale_lr={True}'
#             alg_name += f'-alg2--6-state10-dataset-tau0.3'
#             path_name = f'{dataset_name}/{DEVICE_NAME}'
#             check_point_folder = f'checkpoints/{path_name}'
#             check_point_path = f'{check_point_folder}/{alg_name}.pth'
#             if not os.path.isdir(check_point_folder):
#                 os.makedirs(check_point_folder)
#             print(f'Saving model checkpoint to {check_point_path}')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }, check_point_path)

#     print("\nTraining Time (s) = {}".format(time() - time_init))


# @torch.no_grad()
# def test_evaluation(model, val_set):
#     """Test trained network

#     Args:
#         model (nn.Model): Trained model to be evaluated
#         val_set (DataLoader): Validation set to perform the evaluation
#     """
#     # Setup counter of images predicted to 0.
#     predicted_ok = 0
#     total_images = 0

#     model.eval()
#     classifier = nn.NLLLoss()

#     total_loss = 0
#     for images, labels in val_set:
#         # Predict image.
#         images = images.to(DEVICE)
#         labels = labels.to(DEVICE)

#         # images = images.view(images.shape[0], -1)
#         pred = model(images)

#         _, predicted = torch.max(pred.data, 1)
#         total_images += labels.size(0)
#         predicted_ok += (predicted == labels).sum().item()
#         loss = classifier(pred, labels)
#         total_loss += loss.item()

#     # print("\nNumber Of Images Tested = {}".format(total_images))
#     # print("Model Accuracy = {}".format(predicted_ok / total_images))
#     loss = total_loss / total_images
#     accuracy = predicted_ok / total_images
#     return loss, accuracy
# def get_AnalogSGD_optimizer_generator(lr=lr, *args, **kargs):
#     def _generator(params):
#         return AnalogSGD(params, lr=lr, *args, **kargs)
#     return _generator

# construction_seed = 23

# def config_IO(io_param: IOParameters, config): 
#     if config["io_perfect"]:
#         io_param.is_perfect = True
#     else:
#         if config["io_inp_res_bit"] != -1:
#             io_param.inp_res = config["io_inp_res_bit"]
#         if config["io_out_res_bit"] != -1:
#             io_param.out_res = config["io_out_res_bit"]
#         if config["io_inp_noise"] != -1:
#             io_param.inp_noise = config["io_inp_noise"]
#         if config["io_out_noise"] != -1:
#             io_param.out_noise = config["io_out_noise"]
#     print(io_param.out_res)
#     print(args.io_perfect_forward)
# def get_config(config_name, mu_r: float = 0.0, sigma_r: float = 0.0):
#     if config_name == 'TT-v1':
#         active_weight_decay_count = args.TTv1_active_weight_decay_count
#         active_weight_decay_probability = args.TTv1_active_weight_decay_probability
#         algorithm = 'ttv1'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad

#         # 这里把 mu_r, sigma_r 传给 Softbounds 设备
#         device_config_fit = get_device(
#             "Softbounds",
#             reference_mean=mu_r,
#             reference_std=sigma_r,
#             subtract_sp=True
#         )

#         rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
#         # update onto A matrix needs to be increased somewhat
#         rpu_config.mapping.learn_out_scaling = True
#         rpu_config.mapping.weight_scaling_columnwise = True
#         # rpu_config.mapping.weight_scaling_omega = 0.6
#         rpu_config.device.fast_lr =  0.05
#         rpu_config.device.n_reads_per_transfer = 1
#         # rpu_config.device.no_self_transfer = (not args.TTv1_self_transfer)

#         if active_weight_decay_count != 0:
#             rpu_config.device.active_weight_decay_count = active_weight_decay_count
#         if active_weight_decay_probability != 0:
#             rpu_config.device.active_weight_decay_probability = active_weight_decay_probability
        
#         config = {
#             'name': f'TT-v1',
#             'rpu_config': rpu_config,
#             'optimizer_cls': get_AnalogSGD_optimizer_generator(),
#             'grad_per_iter': 1,
#         }
#         if active_weight_decay_count != 0:
#             config['name'] += f'-T={active_weight_decay_count}'
#         elif active_weight_decay_probability > 0:
#             config['name'] += f'-T={active_weight_decay_probability}'
            
#         if rpu_config.device.n_reads_per_transfer > 1:
#             config['name'] += f'-st={rpu_config.device.n_reads_per_transfer}'
#         if not rpu_config.device.no_self_transfer:
#             config['name'] += f'-stran'
#         return config

#     elif config_name == 'TT-v2':
    
#         rpu_config = UnitCellRPUConfig(
#             # device=ChoppedTransferCompound(
#             # device=BufferedTransferCompound(
#             device=DynamicTransferCompound(
#                 unit_cell_devices=[get_device(
#                     "Softbounds",
#                     reference_mean=mu_r,
#                     reference_std=sigma_r,
#                     subtract_sp=True     
#                 ), get_device(
#                     "Softbounds",
#                     reference_mean=mu_r,
#                     reference_std=sigma_r,
#                     subtract_sp=True  
#                 )],
#                 transfer_update=UpdateParameters(
#                     desired_bl=1, update_bl_management=False, update_management=False
#                 ),
#                 in_chop_prob=0.1,
#                 units_in_mbatch=True,
#                 # auto_scale=False,
#                 construction_seed=123,
#             ),
#             forward=IOParameters(),
#             backward=IOParameters(),
#             update=UpdateParameters(desired_bl=5),
#             # **kwargs,
#         )
#         # update onto A matrix needs to be increased somewhat
#         rpu_config.mapping.learn_out_scaling = True
#         rpu_config.mapping.weight_scaling_columnwise = True
#         # rpu_config.mapping.weight_scaling_omega = 0.1
#         # rpu_config.mapping.weight_scaling_omega = 0.3
        
#         rpu_config.device.fast_lr = 0.05        # rpu_config.device.scale_fast_lr = False
#         rpu_config.device.transfer_lr = 1
#         rpu_config.device.scale_transfer_lr = True
#         # rpu_config.device.auto_granularity = 1000
        
#         config = {
#             'name': f'TT-v2',
#             # 'name': f'TT-v2-omega={rpu_config.mapping.weight_scaling_omega}',
#             # 'name': f'TT-v2-flr={rpu_config.device.fast_lr}',
#             # 'name': f'granularity={rpu_config.device.auto_granularity}',
#             'rpu_config': rpu_config,
#             'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
#             'grad_per_iter': 1,
#             # 'batch_size': BATCH_SIZE,
#         }
#         return config
# #  μ_r=0, σ_r=0 for calibration SP offset 
# config_calib = get_config(setting, mu_r=0.0, sigma_r=0.0)
# rpu_config_calib = config_calib['rpu_config']

# def create_analog_network_cpu(rpu_config):
#     channel = [16, 32, 512, 128]
#     model = AnalogSequential(
#         AnalogConv2d(
#             in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=rpu_config
#         ),
#         nn.Tanh(),
#         nn.MaxPool2d(kernel_size=2),
#         AnalogConv2d(
#             in_channels=channel[0],
#             out_channels=channel[1],
#             kernel_size=5,
#             stride=1,
#             rpu_config=rpu_config,
#         ),
#         nn.Tanh(),
#         nn.MaxPool2d(kernel_size=2),
#         nn.Tanh(),
#         nn.Flatten(),
#         AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
#         nn.Tanh(),
#         AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=rpu_config),
#         nn.LogSoftmax(dim=1),
#     )
#     return model

# print("\n=== Calibrating SP offset using a standalone UnitCell SoftBounds tile ===")
# N_CAL_PULSES = 6000   # N_pulse
# mu_r, sigma_r = calibrate_sp_offset_unitcell(
#     n_pulses=N_CAL_PULSES,
#     n_rows=512,
#     n_cols=512
# )


# # use  μ_r, σ_r form calibration to build real training config / model
# config = get_config(setting, mu_r=mu_r, sigma_r=sigma_r)
# no_tau_list = ['FP SGD']
# dataset_name = 'MNIST-CNN'
# name = config['name']
# if config['name'] not in no_tau_list:
#     name += f'-tau={tau}'
# path_name = f'{dataset_name}/TT-AW-no-fit-state'

# rpu_config = config['rpu_config']

# check_point_folder = f'checkpoints/{path_name}'
# check_point_path = f'{check_point_folder}/{name}.pth'
# log_path = f'runs/{path_name}/{name}'
# logger = Logger(log_path)
# if args.save_checkpoint and not os.path.isdir(check_point_folder):
#     os.makedirs(check_point_folder)

# # Load datasets.
# train_dataset, validation_dataset = load_images(fraction=0.1)

# # Prepare the model for training
# model = create_analog_network(rpu_config=rpu_config)

# # Train
# train(model, train_dataset, config, logger, check_point_path)

# # Evaluate
# test_evaluation(model, validation_dataset)



# # # --------- 对整个 analog model 做 SP offset 标定 ----------
# # def calibrate_sp_offset_on_model(model, n_pulses: int = 2000, init_w_value: float = -0.5):
# #     """
# #     在当前 analog model 的每一个 analog tile 上：
# #       1) 读取 true SP: a*_{ij} = w_reference
# #       2) 初始化权重为 init_w_value
# #       3) 施加 n_pulses 个交替 ± 脉冲，得到 r_{ij}(N)
# #       4) 计算 offset = r - a*，汇总所有层，得到全局 μ_r, σ_r

# #     返回:
# #         mu_r, sigma_r
# #     """
# #     all_offsets = []

# #     for name, module in model.named_modules():
# #         if hasattr(module, "analog_module"):
# #             tile = module.analog_module.tile

# #             # 1) true SP
# #             w_ref = get_w_reference_from_tile(tile)  # a*_{ij}

# #             # 2) 初始化权重
# #             w, _ = tile.get_weights()
# #             w_init = torch.full_like(w, init_w_value, dtype=torch.float32)
# #             tile.set_weights(w_init, torch.empty(0))

# #             # 3) 施加交替脉冲（在 tile 的矩阵维度上）
# #             d_size, x_size = w_init.shape  # rows, cols
# #             x = torch.ones(x_size, dtype=torch.float32)
# #             d_pos = torch.ones(d_size, dtype=torch.float32)
# #             d_neg = -torch.ones(d_size, dtype=torch.float32)

# #             for t in range(n_pulses):
# #                 d = d_pos if (t % 2 == 0) else d_neg
# #                 tile.update(x, d)

# #             # 4) 读出 r_{ij}(N)，计算 offset
# #             w_pulse, _ = tile.get_weights()
# #             w_pulse_flat = w_pulse.detach().cpu().numpy().ravel()
# #             offset = w_pulse_flat - w_ref
# #             all_offsets.append(offset)

# #             print(f"[Calib] Layer {name}: collected {offset.size} offsets.")

# #     if not all_offsets:
# #         raise RuntimeError("No analog tiles found in model during SP calibration.")

# #     all_offsets = np.concatenate(all_offsets, axis=0)
# #     mu_r = float(all_offsets.mean())
# #     sigma_r = float(all_offsets.std())

# #     print("\n========== SP offset calibration ==========")
# #     print(f"mu_r   (mean offset)      = {mu_r:.6f}")
# #     print(f"sigma_r(std of offset)   = {sigma_r:.6f}")
# #     print("===========================================\n")
# #     return mu_r, sigma_r
import os
from time import time
from dataclasses import dataclass, field
# Imports from PyTorch.
import torch
torch.autograd.set_detect_anomaly(True)
import random
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import numpy as np
import sys
from torch.optim.lr_scheduler import LambdaLR
from utils.logger import Logger
from collections import deque
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AIHWKIT_SRC = os.path.join(CURRENT_DIR, 'aihwkit', 'src')
sys.path.insert(0, AIHWKIT_SRC)
# # For warm start
import aihwkit
print('aihwkit path: ', aihwkit.__file__)
from aihwkit.nn import AnalogLinear, AnalogSequential, AnalogConv2d
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda
import aihwkit.simulator.rpu_base.devices as dev
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.nn.conversion import convert_to_analog, convert_to_digital
from aihwkit.simulator.configs import (
    build_config,
    UnitCellRPUConfig,
    DigitalRankUpdateRPUConfig,
    FloatingPointRPUConfig,
    SingleRPUConfig,
    UpdateParameters,
)
from aihwkit.simulator.configs.devices import (
    FloatingPointDevice,
    ConstantStepDevice,
    VectorUnitCell,
    LinearStepDevice,
    SoftBoundsDevice,
    SoftBoundsReferenceDevice,
    TransferCompound,
    MixedPrecisionCompound,
    BufferedTransferCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
)
import argparse
from enum import Enum
from aihwkit.simulator.parameters.inference import DriftParameter
parser = argparse.ArgumentParser(description="A simple command-line argument example")

# Add command line arguments
parser.add_argument('-SETTING', '--SETTING', type=str, help="", default='FP SGD')
parser.add_argument('-BATCH_SIZE', '--BATCH_SIZE', type=int, help="", default='8')
parser.add_argument('-CUDA', '--CUDA', type=int, help="", default=-1)
parser.add_argument('-tau', '--tau', type=float, help="", default=1)
parser.add_argument('-TTAWDC', '--TTv1-active-weight-decay-count', type=int, help="", default=0)
parser.add_argument('-TTAWDP', '--TTv1-active_weight_decay_probability', type=float, help="", default=0)
parser.add_argument('-save', '--save-checkpoint', action='store_true')
parser.add_argument('-Tcolumn', '--Tcolumn', type=int, help="", default='1')
parser.add_argument('-ns', '--ns', type=float, help="", default='1')
parser.add_argument('-sigma', '--sigma', type=float, help="", default='0.3')
parser.add_argument('-gamma', '--gamma', type=float, help="", default='0')
parser.add_argument('-Wmax', '--Wmax', type=float, help="", default='1')
parser.add_argument('-dwmin', '--dwmin', type=float, help="", default='0.1')
# IO precision and noise parameters
parser.add_argument('--io_inp_res_bit', type=float, default='7')
parser.add_argument('--io_out_res_bit', type=float, default='9')
parser.add_argument('--io_inp_noise', type=float, default='0.0')
parser.add_argument('--io_out_noise', type=float, default='0.0')
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--io_perfect_forward', type=str2bool, default=True)
parser.add_argument('--io_perfect_backward', type=str2bool, default=True)
checkpoint_path = "/home/jindan/Desktop/analog/checkpoints/MNIST-CNN/Softbounds/TT-v1-tile=6-alg2--6-state4-dataset-tau0.5.pth"
import os

print("Checking path:", checkpoint_path)
print("Exists?", os.path.exists(checkpoint_path))

args = parser.parse_args()
setting = args.SETTING

# Check device
USE_CUDA = 0
if cuda.is_compiled() and args.CUDA >= 0:
    USE_CUDA = 1
DEVICE = torch.device(f"cuda:{args.CUDA}" if USE_CUDA else "cpu")
print('Using Device: ', DEVICE)


# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data")

# Training parameters.
EPOCHS = 200
N_CLASSES = 10

tau = args.tau
# DEVICE_NAME = 'PCM'
# DEVICE_NAME = 'HfO2'
# DEVICE_NAME = 'OM'
DEVICE_NAME = 'Softbounds'
# DEVICE_NAME = 'RRAM-offset'

lr = 0.05


def get_model_size(model, input):
    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    raise SystemExit
    return macs, params

def get_device(device_name='CS',
               reference_mean: float = 0.0,
               reference_std: float = 0.0,
               subtract_sp: bool = True):
    """Return a device meta config. Softbounds uses SoftBoundsReferenceDevice."""
    if device_name == 'CS':
        return ConstantStepDevice()
    elif device_name == 'Softbounds':
        return SoftBoundsReferenceDevice(
            dw_min=0.001,
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
            reference_mean=reference_mean,
            reference_std=reference_std,
            count_pulses = True,
            subtract_symmetry_point=subtract_sp
        )
    else:
        raise NotImplementedError

# --------- Helper: 从 tile 读 w_reference ----------
def get_w_reference_from_tile(tile) -> np.ndarray:
    """Read w_reference (true SP) from hidden parameters and flatten to 1D."""
    hidden = tile.get_hidden_parameters()
    ref_key = None
    for name in hidden.keys():
        if "reference" in name.lower():
            ref_key = name
            break
    if ref_key is None:
        raise RuntimeError("Could not find a hidden parameter containing 'reference' in tile.")
    w_ref_tensor = hidden[ref_key]  # shape [d_size, x_size]
    w_ref = w_ref_tensor.detach().cpu().numpy().ravel()
    return w_ref

from aihwkit.simulator.tiles import AnalogTile
import math

def calibrate_sp_offset_unitcell(
    n_pulses: int = 2000,
    n_rows: int = 512,
    n_cols: int = 512,
) -> tuple[float, float]:
    """
    使用两块 UnitCell SoftBounds tile 来标定 SP mismatch:

      - tile_true: subtract_symmetry_point=True, reference_mean=0, reference_std=0
          --> w_reference = a*_{ij}  (理论 / 数学上的真 SP 样本)
      - tile_pulse: subtract_symmetry_point=False, reference_mean=0, reference_std=0
          --> 通过有限脉冲找到的近似 SP 样本

    然后用「两个分布的差」来定义 μ_r, σ_r：
        μ_r = mean(SP_pulse) - mean(SP_true)
        σ_r = std(SP_pulse)  - std(SP_true)    (或者用 sqrt(σ_pulse^2 - σ_true^2))

    返回:
        mu_r, sigma_r
    """

    # ------------- 1) 真 SP 分布: tile_true (subtract_sp = True) -------------
    softbounds_true = get_device(
        "Softbounds",
        reference_mean=0.0,
        reference_std=0.0,
        subtract_sp=True       # 让 w_reference 里加上 sp
    )
    rpu_config_true = UnitCellRPUConfig(
        device=softbounds_true,
        update=UpdateParameters(
            desired_bl=1,
            update_bl_management=False,
            update_management=False
        )
    )
    tile_true = AnalogTile(n_rows, n_cols, rpu_config_true)

    # 从 tile_true 读出 w_reference = a*_{ij}
    hidden_true = tile_true.get_hidden_parameters()
    if isinstance(hidden_true, dict):
        ref_key = None
        for name in hidden_true.keys():
            if "reference" in name.lower():
                ref_key = name
                break
        if ref_key is None:
            raise RuntimeError("Could not find a hidden parameter containing 'reference'.")
        w_ref_tensor = hidden_true[ref_key]
    else:
        raise RuntimeError(
            f"Expected hidden parameters to be a dict for UnitCell tile, got {type(hidden_true)}."
        )

    sp_true = w_ref_tensor.detach().cpu().numpy().ravel()
    mu_true = float(sp_true.mean())
    sigma_true = float(sp_true.std())

    print("---- True SP distribution (from w_reference, subtract_sp=True) ----")
    print(f"mu_true     = {mu_true:.6f}")
    print(f"sigma_true  = {sigma_true:.6f}")

    # ------------- 2) 近似 SP 分布: tile_pulse (subtract_sp = False) -------------
    softbounds_pulse = get_device(
        "Softbounds",
        reference_mean=0.0,
        reference_std=0.0,
        subtract_sp=False      # 手动脉冲找 SP 时不做对称点平移
    )
    rpu_config_pulse = UnitCellRPUConfig(
        device=softbounds_pulse,
        update=UpdateParameters(
            desired_bl=1,
            update_bl_management=False,
            update_management=False
        )
    )
    tile_pulse = AnalogTile(n_rows, n_cols, rpu_config_pulse)

    # 初始化权重为 -0.5
    w_init = torch.full((n_rows, n_cols), -0.5, dtype=torch.float32)
    tile_pulse.set_weights(w_init, torch.empty(0))

    # 在整个矩阵上打交替脉冲
    x = torch.ones(n_cols, dtype=torch.float32)
    d_pos = torch.ones(n_rows, dtype=torch.float32)
    d_neg = -torch.ones(n_rows, dtype=torch.float32)

    for t in range(n_pulses):
        d = d_pos if (t % 2 == 0) else d_neg
        tile_pulse.update(x, d)

    w_pulse, _ = tile_pulse.get_weights()
    sp_approx = w_pulse.detach().cpu().numpy().ravel()

    mu_pulse = float(sp_approx.mean())
    sigma_pulse = float(sp_approx.std())

    print("---- Approx SP distribution (from pulses, subtract_sp=False) ----")
    print(f"mu_pulse    = {mu_pulse:.6f}")
    print(f"sigma_pulse = {sigma_pulse:.6f}")

    # ------------- 3) 用“两个分布的差”来定义 μ_r, σ_r -------------
    # 均值差：近似分布相对真分布的偏移
    mu_r = mu_pulse - mu_true

    # 标准差差：这里先按你论文里的直觉定义成「std 差」
    # 如果你希望物理上严格对应 var 叠加，也可以改成 sqrt(max(0, σ_pulse^2 - σ_true^2))
    sigma_r = sigma_pulse - sigma_true

    print("\n========== SP offset calibration (UnitCell, distribution-wise) ==========")
    print(f"n_pulses          = {n_pulses}")
    print(f"mu_r   (mean diff)     = {mu_r:.6f}")
    print(f"sigma_r(std diff)      = {sigma_r:.6f}")
    print("=====================================================================\n")

    return mu_r, sigma_r


def load_images(fraction=1, seed=None):
    """Load a random fraction of MNIST images for training and validation."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load full datasets
    train_set_full = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set_full = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)

    # Compute subset sizes
    num_train = int(len(train_set_full) * fraction)
    num_val = int(len(val_set_full) * fraction)

    # Optionally fix random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Randomly sample subset indices
    train_indices = random.sample(range(len(train_set_full)), num_train)
    val_indices = random.sample(range(len(val_set_full)), num_val)

    # Create subsets
    train_subset = Subset(train_set_full, train_indices)
    val_subset = Subset(val_set_full, val_indices)

    # Wrap into DataLoaders
    train_data = DataLoader(train_subset, batch_size=args.BATCH_SIZE, shuffle=True)
    validation_data = DataLoader(val_subset, batch_size=args.BATCH_SIZE, shuffle=True)

    return train_data, validation_data

def create_analog_network(rpu_config):
    """Return a LeNet5 inspired analog model."""
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(
            in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=rpu_config
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=5,
            stride=1,
            rpu_config=rpu_config,
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        nn.Tanh(),
        AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=rpu_config),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda(DEVICE)
    return model
def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)

triggered_count = 0
loss_history = []
import numpy as np
import torch

def _sum_pulse_counters(pc: torch.Tensor) -> tuple[int, int, int]:
    """
    pc: torch.Tensor from get_pulse_counters()
    Returns: (pos, neg, total) as python ints
    Works for common shapes:
      - [2, rows, cols]
      - [n_dev, 2, rows, cols]
      - [2, n_dev, rows, cols]
    """
    pc_cpu = pc.detach().cpu()

    if pc_cpu.ndim == 3:
        # [2, rows, cols]
        pos = int(pc_cpu[0].sum().item())
        neg = int(pc_cpu[1].sum().item())
        return pos, neg, pos + neg

    if pc_cpu.ndim == 4:
        # could be [n_dev, 2, r, c] or [2, n_dev, r, c]
        if pc_cpu.shape[0] == 2:
            # [2, n_dev, r, c]
            pos = int(pc_cpu[0].sum().item())
            neg = int(pc_cpu[1].sum().item())
            return pos, neg, pos + neg
        elif pc_cpu.shape[1] == 2:
            # [n_dev, 2, r, c]
            pos = int(pc_cpu[:, 0].sum().item())
            neg = int(pc_cpu[:, 1].sum().item())
            return pos, neg, pos + neg

    raise ValueError(f"Unexpected pulse counter shape: {tuple(pc_cpu.shape)}")


def get_model_pulse_counters(model) -> dict:
    """
    Iterate all analog layers, read pulse counters, and sum them.
    Returns dict with layer-wise and total counts.
    """
    out = {"layers": {}, "total_pos": 0, "total_neg": 0, "total": 0}

    for name, module in model.named_modules():
        if hasattr(module, "analog_module"):
            # bottom C++ tile is usually at ...tile.tile
            base_tile = module.analog_module.tile
            if not hasattr(base_tile, "tile"):
                continue

            cpp_tile = base_tile.tile
            if not hasattr(cpp_tile, "get_pulse_counters"):
                continue

            pc = cpp_tile.get_pulse_counters()
            pos, neg, total = _sum_pulse_counters(pc)

            out["layers"][name] = {"pos": pos, "neg": neg, "total": total}
            out["total_pos"] += pos
            out["total_neg"] += neg
            out["total"] += total

    return out


def train(model, train_set, config, logger, checkpoint_path):
    """Train the network."""
    optimizer_cls = config['optimizer_cls']
    classifier = nn.NLLLoss()
    optimizer = optimizer_cls(model.parameters())

    def lr_lambda(epoch):
        return 0.1 ** (epoch // 35)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- Load checkpoint if exists ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1  # resume from next epoch
        # logger.info(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Training from scratch.")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            output = model(images)
            loss = classifier(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_set)
        loss_history.append(train_loss)
        # trigger_tile_switch_by_plateau(model, loss_history)
        # For warm start
        test_loss, test_accuracy = test_evaluation(model, validation_dataset)

        log_str = f"Epoch {epoch} - Training loss: {train_loss:.6f}   Test Accuracy: {test_accuracy:.4f}"
        

        logger.write(epoch, log_str, {
            "Loss/train": train_loss,
            "Loss/test": test_loss,
            "Accuracy/test": test_accuracy,
            "State/lr": scheduler.get_last_lr()[0],
            # "Tile/switch": float(plateau_triggered),  # optional: also track in tensorboard
        })

        if args.save_checkpoint:
            alg_name = config['name']
            alg_name += f'-tile={num_tile}'
            # alg_name += f'-scale_lr={True}'
            alg_name += f'-alg2--6-state10-dataset-tau0.3'
            path_name = f'{dataset_name}/{DEVICE_NAME}'
            check_point_folder = f'checkpoints/{path_name}'
            check_point_path = f'{check_point_folder}/{alg_name}.pth'
            if not os.path.isdir(check_point_folder):
                os.makedirs(check_point_folder)
            print(f'Saving model checkpoint to {check_point_path}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, check_point_path)
        if train_loss <= 0.1:
            pulse_stat = get_model_pulse_counters(model)
            print(f"[REACH 95%] epoch={epoch}, pulses={pulse_stat['total']}")
            break


    print("\nTraining Time (s) = {}".format(time() - time_init))


@torch.no_grad()
def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    model.eval()
    classifier = nn.NLLLoss()

    total_loss = 0
    for images, labels in val_set:
        # Predict image.
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # images = images.view(images.shape[0], -1)
        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        loss = classifier(pred, labels)
        total_loss += loss.item()

    # print("\nNumber Of Images Tested = {}".format(total_images))
    # print("Model Accuracy = {}".format(predicted_ok / total_images))
    loss = total_loss / total_images
    accuracy = predicted_ok / total_images
    return loss, accuracy
def get_AnalogSGD_optimizer_generator(lr=lr, *args, **kargs):
    def _generator(params):
        return AnalogSGD(params, lr=lr, *args, **kargs)
    return _generator

construction_seed = 23

def config_IO(io_param: IOParameters, config): 
    if config["io_perfect"]:
        io_param.is_perfect = True
    else:
        if config["io_inp_res_bit"] != -1:
            io_param.inp_res = config["io_inp_res_bit"]
        if config["io_out_res_bit"] != -1:
            io_param.out_res = config["io_out_res_bit"]
        if config["io_inp_noise"] != -1:
            io_param.inp_noise = config["io_inp_noise"]
        if config["io_out_noise"] != -1:
            io_param.out_noise = config["io_out_noise"]
    print(io_param.out_res)
    print(args.io_perfect_forward)
def get_config(config_name, mu_r: float = 0.0, sigma_r: float = 0.0):
    if config_name == 'TT-v1':
        active_weight_decay_count = args.TTv1_active_weight_decay_count
        active_weight_decay_probability = args.TTv1_active_weight_decay_probability
        algorithm = 'ttv1'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad

        # 这里把 mu_r, sigma_r 传给 Softbounds 设备
        device_config_fit = get_device(
            "Softbounds",
            reference_mean=mu_r,
            reference_std=sigma_r,
            subtract_sp=True
        )

        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr =  0.1
        rpu_config.device.n_reads_per_transfer = 1
        # rpu_config.device.no_self_transfer = (not args.TTv1_self_transfer)

        if active_weight_decay_count != 0:
            rpu_config.device.active_weight_decay_count = active_weight_decay_count
        if active_weight_decay_probability != 0:
            rpu_config.device.active_weight_decay_probability = active_weight_decay_probability
        
        config = {
            'name': f'TT-v1',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
        if active_weight_decay_count != 0:
            config['name'] += f'-T={active_weight_decay_count}'
        elif active_weight_decay_probability > 0:
            config['name'] += f'-T={active_weight_decay_probability}'
            
        if rpu_config.device.n_reads_per_transfer > 1:
            config['name'] += f'-st={rpu_config.device.n_reads_per_transfer}'
        if not rpu_config.device.no_self_transfer:
            config['name'] += f'-stran'
        return config

    # 其他 config_name 如果有的话，按需扩展
    elif config_name == 'TT-v2':
    
        rpu_config = UnitCellRPUConfig(
            # device=ChoppedTransferCompound(
            device=BufferedTransferCompound(
            # device=DynamicTransferCompound(
                unit_cell_devices=[get_device(
            "Softbounds",
            reference_mean=mu_r,
            reference_std=sigma_r,
            subtract_sp=True
        ), get_device(
            "Softbounds",
            reference_mean=mu_r,
            reference_std=sigma_r,
            subtract_sp=True
        )],
                transfer_update=UpdateParameters(
                    desired_bl=1, update_bl_management=False, update_management=False
                ),
                # in_chop_prob=0.1,
                units_in_mbatch=True,
                # auto_scale=False,
                construction_seed=123,
                
            ),
            forward=IOParameters(),
            backward=IOParameters(),
            update=UpdateParameters(desired_bl=5),
            # **kwargs,
        )
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.1
        rpu_config.mapping.weight_scaling_omega = 0.3
        
        rpu_config.device.fast_lr = 0.01
        # rpu_config.device.scale_fast_lr = False
        rpu_config.device.transfer_lr = 1
        rpu_config.device.scale_transfer_lr = True
        # rpu_config.device.auto_granularity = 1000
        
        config = {
            'name': f'TT-v2',
            # 'name': f'TT-v2-omega={rpu_config.mapping.weight_scaling_omega}',
            # 'name': f'TT-v2-flr={rpu_config.device.fast_lr}',
            # 'name': f'granularity={rpu_config.device.auto_granularity}',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': 64,
        }
        return config
    elif config_name == 'RL-v2':
        rpu_config = UnitCellRPUConfig(
            # device=ChoppedTransferCompound(
            device=DynamicTransferCompound(
                unit_cell_devices=[get_RPU_device(RPU_NAME), get_RPU_device(RPU_NAME)],
                transfer_forward=IOParameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=UpdateParameters(
                    desired_bl=1, update_bl_management=False, update_management=False
                ),
                in_chop_prob=0.05,
                units_in_mbatch=True,
                auto_scale=False,
                construction_seed=123,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
            update=UpdateParameters(desired_bl=5),
        )
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        
        # rpu_config.device.buffer_as_momentum = True
        # rpu_config.device.momentum = 0.9
        # rpu_config.device.fast_lr = 1
        rpu_config.device.fast_lr = 0.5
        # rpu_config.device.scale_fast_lr = True
        rpu_config.device.scale_fast_lr = False
        # rpu_config.device.transfer_lr = 0.11 + 1 / num_of_states
        rpu_config.device.transfer_lr = 0.05
    
        rpu_config.device.scale_transfer_lr = False
        rpu_config.device.auto_granularity = 1000
        
        config = {
            'name': f'RL-v2',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
        return config
# 先用 μ_r=0, σ_r=0 的配置构建一个 model，用来做 SP offset 标定
config_calib = get_config(setting, mu_r=0.0, sigma_r=0.0)
rpu_config_calib = config_calib['rpu_config']
def set_pref_from_config(rpu_config, enable: bool):
    if enable:
        os.environ["AIHWKIT_PREF_ON"] = "1"
        os.environ["AIHWKIT_PREF_GAMMA"] = str(float(0.1))
        print(f"[PY] Pref ON, gamma={os.environ['AIHWKIT_PREF_GAMMA']}")
    else:
        os.environ.pop("AIHWKIT_PREF_ON", None)
        os.environ.pop("AIHWKIT_PREF_GAMMA", None)
        print("[PY] Pref OFF")


config = get_config(setting)
# set_pref_from_config(config["rpu_config"], enable="True")
set_pref_from_config(config["rpu_config"], enable=False)

# 标定用的模型可以不放到 GPU 上（减少干扰）
def create_analog_network_cpu(rpu_config):
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(
            in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=rpu_config
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=5,
            stride=1,
            rpu_config=rpu_config,
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        nn.Tanh(),
        AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=rpu_config),
        nn.LogSoftmax(dim=1),
    )
    return model

print("\n=== Calibrating SP offset using a standalone UnitCell SoftBounds tile ===")
N_CAL_PULSES =8000   # 这里就是“输入的脉冲数”，你可以改或做成命令行参数
mu_r, sigma_r = calibrate_sp_offset_unitcell(
    n_pulses=N_CAL_PULSES,
    n_rows=512,
    n_cols=512
)

def debug_first_tile_reference(model, tag=""):
    for name, module in model.named_modules():
        if hasattr(module, "analog_module"):
            tile = module.analog_module.tile
            # 取 hidden parameter 里的 w_reference
            hidden = tile.get_hidden_parameters()
            ref_key = None
            for k in hidden.keys():
                if "reference" in k.lower():
                    ref_key = k
                    break
            if ref_key is None:
                print(f"[{tag}] {name}: no reference key")
                return
            wref = hidden[ref_key].detach().cpu().numpy().ravel()
            print(f"[{tag}] first analog layer = {name}")
            print(f"[{tag}] w_reference mean={wref.mean():.6f}, std={wref.std():.6f}, min={wref.min():.3f}, max={wref.max():.3f}")
            return

# after model creation


# 用标定得到的 μ_r, σ_r 构建真正训练用的 config / model
config = get_config(setting, mu_r=mu_r, sigma_r=sigma_r)
no_tau_list = ['FP SGD']
dataset_name = 'MNIST-CNN'
name = config['name']
if config['name'] not in no_tau_list:
    name += f'-tau={tau}'
path_name = f'{dataset_name}/TT-AW-no-fit-state'

rpu_config = config['rpu_config']

check_point_folder = f'checkpoints/{path_name}'
check_point_path = f'{check_point_folder}/{name}.pth'
log_path = f'runs/{path_name}/{name}'
logger = Logger(log_path)
if args.save_checkpoint and not os.path.isdir(check_point_folder):
    os.makedirs(check_point_folder)

# Load datasets.
train_dataset, validation_dataset = load_images(fraction=1)

# Prepare the model for training（这次用带 offset 的 rpu_config）
model = create_analog_network(rpu_config=rpu_config)
# debug_first_tile_reference(model, tag="TRAIN_MODEL")
# Train
train(model, train_dataset, config, logger, check_point_path)

# Evaluate
test_evaluation(model, validation_dataset)
