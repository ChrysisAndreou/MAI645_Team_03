import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# To import from sibling scripts
import sys
sys.path.append(os.path.dirname(__file__))

# Imports from training scripts
from pytorch_train_pos_aclstm import acLSTM as PosAcLSTM
from pytorch_train_pos_aclstm import Hip_index as PosHipIndex
from pytorch_train_pos_aclstm import load_dances as load_pos_dances
# Globals from pytorch_train_pos_aclstm that might be needed for model init or data prep
# POS_SEQ_LEN = 100 # Default model sequence length
POS_CONDITION_NUM = 5
POS_GROUNDTRUTH_NUM = 5
POS_JOINTS_NUM = 57
POS_IN_FRAME_SIZE = POS_JOINTS_NUM * 3
POS_HIDDEN_SIZE = 1024


from pytorch_train_quad_aclstm import acLSTM as QuadAcLSTM
from pytorch_train_quad_aclstm import HIP_X_CHANNEL, HIP_Y_CHANNEL, HIP_Z_CHANNEL
from pytorch_train_quad_aclstm import load_dances_and_metadata as load_quad_dances_and_metadata
# Globals from pytorch_train_quad_aclstm
QUAD_GROUNDTRUTH_NUM = 5
QUAD_DEFAULT_HIDDEN_SIZE = 1024
import torch.nn.functional as F # For Quad loss calculation
import torch.nn as nn # For Quad loss calculation (MSE)


# --- Helper function to extract iteration from filename ---
def extract_iteration(filename):
    match = re.search(r'(\d{7,})\.weight', filename)
    if match:
        return int(match.group(1))
    return -1

# --- POS Model Helper Functions ---

def get_validation_batch_pos(data_folder, model_seq_len, batch_size=1):
    """Loads the first dance and takes the first possible segment for validation."""
    dances = load_pos_dances(data_folder)
    if not dances:
        raise ValueError(f"No dance data found in {data_folder}")
    
    dance = dances[0] # Use the first dance
    sampling_seq_len = model_seq_len + 2 # As per training script logic

    if dance.shape[0] < sampling_seq_len:
        raise ValueError(f"First dance in {data_folder} is too short ({dance.shape[0]} frames) for "
                         f"sampling_seq_len {sampling_seq_len}")
    
    # Take the first segment
    # No speed resampling or augmentation for validation batch
    raw_dance_segment_np = dance[0:sampling_seq_len, :]
    return np.expand_dims(raw_dance_segment_np, axis=0) # Add batch dimension


def prepare_pos_data_for_model(real_seq_np_batch, model_seq_len):
    """Prepares POS data batch for model input and ground truth."""
    # real_seq_np_batch has shape (batch, sampling_seq_len, features)
    # sampling_seq_len = model_seq_len + 2
    
    # Hip difference calculation (as in train_one_iteraton)
    dif = real_seq_np_batch[:, 1:real_seq_np_batch.shape[1]] - real_seq_np_batch[:, 0: real_seq_np_batch.shape[1]-1]
    real_seq_dif_hip_x_z_np = real_seq_np_batch[:, 0:real_seq_np_batch.shape[1]-1].copy() # Shape: (batch, model_seq_len+1, feat)

    real_seq_dif_hip_x_z_np[:,:,PosHipIndex*3] = dif[:,:,PosHipIndex*3]
    real_seq_dif_hip_x_z_np[:,:,PosHipIndex*3+2] = dif[:,:,PosHipIndex*3+2]

    # Prepare model input and ground truth
    # model_input_seq_len for forward pass is model_seq_len
    in_real_seq_np = real_seq_dif_hip_x_z_np[:, 0:model_seq_len] 
    # gt sequence for loss is also model_seq_len
    predict_groundtruth_seq_np = real_seq_dif_hip_x_z_np[:, 1:model_seq_len+1]
    
    in_real_seq_torch = torch.FloatTensor(in_real_seq_np)
    # Ground truth for loss is reshaped in the original script
    predict_groundtruth_seq_torch = torch.FloatTensor(predict_groundtruth_seq_np).view(real_seq_np_batch.shape[0], -1)
    
    return in_real_seq_torch, predict_groundtruth_seq_torch

# --- QUAD Model Helper Functions ---

def get_validation_batch_quad(data_folder, metadata_path, model_seq_len, batch_size=1):
    """Loads the first dance and takes the first possible segment for validation."""
    dances, metadata = load_quad_dances_and_metadata(data_folder, metadata_path)
    if not dances:
        raise ValueError(f"No dance data found in {data_folder}")
    
    dance = dances[0] # Use the first dance
    sampling_seq_len = model_seq_len + 2 # As per training script logic

    if dance.shape[0] < sampling_seq_len:
        raise ValueError(f"First dance in {data_folder} is too short ({dance.shape[0]} frames) for "
                         f"sampling_seq_len {sampling_seq_len}")

    raw_dance_segment_np = dance[0:sampling_seq_len, :]
    return np.expand_dims(raw_dance_segment_np, axis=0), metadata # Add batch dimension


def prepare_quad_data_for_model(real_seq_np_batch, model_seq_len):
    """Prepares QUAD data batch for model input and ground truth."""
    # real_seq_np_batch: (batch, sampling_seq_len, num_data_channels)
    # sampling_seq_len = model_seq_len + 2

    # Hip X,Z difference calculation (as in train_one_iteration)
    dif_hip_x_scaled = real_seq_np_batch[:, 1:, HIP_X_CHANNEL] - real_seq_np_batch[:, :-1, HIP_X_CHANNEL]
    dif_hip_z_scaled = real_seq_np_batch[:, 1:, HIP_Z_CHANNEL] - real_seq_np_batch[:, :-1, HIP_Z_CHANNEL]
    
    real_seq_processed_for_lstm_np = real_seq_np_batch[:, :-1].copy() # Shape: (batch, model_seq_len+1, channels)
    real_seq_processed_for_lstm_np[:, :, HIP_X_CHANNEL] = dif_hip_x_scaled
    real_seq_processed_for_lstm_np[:, :, HIP_Z_CHANNEL] = dif_hip_z_scaled

    # Prepare model input and ground truth
    # Model input seq length is model_seq_len
    in_lstm_seq_np = real_seq_processed_for_lstm_np[:, :-1]      # Shape: (batch, model_seq_len, channels)
    target_lstm_seq_np = real_seq_processed_for_lstm_np[:, 1:]   # Shape: (batch, model_seq_len, channels)
    
    in_lstm_seq_torch = torch.FloatTensor(in_lstm_seq_np)
    target_lstm_seq_torch = torch.FloatTensor(target_lstm_seq_np)
    
    return in_lstm_seq_torch, target_lstm_seq_torch


def calculate_quad_loss_components(pred_seq, groundtruth_seq):
    """Calculates individual loss components for the QUAD model."""
    # pred_seq and groundtruth_seq are [batch_size, seq_len, out_frame_size]
    hip_pos_pred = pred_seq[..., :3]
    hip_pos_gt = groundtruth_seq[..., :3]
    
    quat_pred = pred_seq[..., 3:]
    quat_gt = groundtruth_seq[..., 3:]

    loss_fn_mse = nn.MSELoss()
    loss_hip = loss_fn_mse(hip_pos_pred, hip_pos_gt)

    quat_pred_reshaped = quat_pred.reshape(-1, 4)
    quat_gt_reshaped = quat_gt.reshape(-1, 4)

    quat_pred_norm = F.normalize(quat_pred_reshaped, p=2, dim=-1)
    quat_gt_norm = F.normalize(quat_gt_reshaped, p=2, dim=-1)

    dot_product = torch.sum(quat_pred_norm * quat_gt_norm, dim=-1)
    abs_dot_product_clamped = torch.clamp(torch.abs(dot_product), min=0.0, max=1.0 - 1e-7)
    
    loss_quat_values = 2.0 * torch.acos(abs_dot_product_clamped)
    loss_rot_ql = torch.mean(loss_quat_values) # Mean over all quaternions in batch and sequence
    
    # Ensure losses are scalar if they aren't already (though mean should make them scalar)
    loss_hip_item = loss_hip.item() if torch.is_tensor(loss_hip) else loss_hip
    loss_rot_ql_item = loss_rot_ql.item() if torch.is_tensor(loss_rot_ql) else loss_rot_ql
    
    total_loss = loss_hip_item + loss_rot_ql_item
    return loss_hip_item, loss_rot_ql_item, total_loss


# --- Plotting Functions ---

def plot_pos_loss_curve(pos_data_folder, pos_weights_folder, output_plot_path, 
                        model_seq_len, device_str, 
                        in_frame_size=POS_IN_FRAME_SIZE, hidden_size=POS_HIDDEN_SIZE):
    print(f"Plotting POS model loss from weights in: {pos_weights_folder}")
    device = torch.device(device_str)

    # 1. Load and prepare validation data (once)
    try:
        val_real_seq_np = get_validation_batch_pos(pos_data_folder, model_seq_len)
        val_in_seq, val_gt_seq = prepare_pos_data_for_model(val_real_seq_np, model_seq_len)
        val_in_seq, val_gt_seq = val_in_seq.to(device), val_gt_seq.to(device)
    except Exception as e:
        print(f"Error loading or preparing POS validation data: {e}")
        return

    # 2. Initialize model
    model = PosAcLSTM(in_frame_size=in_frame_size, hidden_size=hidden_size, out_frame_size=in_frame_size)
    model.to(device)
    model.eval()

    # 3. Iterate through weights, calculate loss
    iterations = []
    losses = []
    
    weight_files = sorted(glob.glob(os.path.join(pos_weights_folder, "*.weight")))
    if not weight_files:
        print(f"No .weight files found in {pos_weights_folder}")
        return

    for weight_file in weight_files:
        iteration = extract_iteration(os.path.basename(weight_file))
        if iteration == -1:
            print(f"Could not parse iteration from {weight_file}, skipping.")
            continue
        
        if iteration % 1000 != 0 and iteration != 0: # Process every 1000th and the first one if exists
             if not any(wf.endswith(f'{0:07d}.weight') for wf in weight_files) and iteration != min(extract_iteration(os.path.basename(w)) for w in weight_files if extract_iteration(os.path.basename(w)) != -1):
                 # If 0000000.weight is not present, ensure we process the very first available weight file.
                 # And then only every 1000th.
                 pass 
             elif iteration != 0:
                # print(f"Skipping POS weight: {os.path.basename(weight_file)} (Iteration: {iteration}) - not a multiple of 1000")
                continue

        print(f"Processing POS weight: {os.path.basename(weight_file)} (Iteration: {iteration})")
        try:
            model.load_state_dict(torch.load(weight_file, map_location=device))
        except Exception as e:
            print(f"Error loading weights from {weight_file}: {e}")
            continue
            
        with torch.no_grad():
            # The forward pass in PosAcLSTM uses Condition_num, Groundtruth_num from its scope
            # For PosAcLSTM, these are POS_CONDITION_NUM, POS_GROUNDTRUTH_NUM
            # predict_seq has shape (batch, model_seq_len * features)
            predict_seq = model.forward(val_in_seq, condition_num=POS_CONDITION_NUM, groundtruth_num=POS_GROUNDTRUTH_NUM)
            loss = model.calculate_loss(predict_seq, val_gt_seq)
            losses.append(loss.item())
            iterations.append(iteration)

    if not iterations:
        print("No losses calculated for POS model.")
        return

    # 4. Plot
    plt.figure(figsize=(10, 6))
    # Sort by iteration for correct plotting order if glob didn't guarantee it
    sorted_pairs = sorted(zip(iterations, losses))
    iterations_sorted, losses_sorted = zip(*sorted_pairs) if sorted_pairs else ([],[])

    plt.plot(iterations_sorted, losses_sorted, marker='o', linestyle='-')
    plt.title(f'POS Model MSE Loss vs. Iterations\nData: {os.path.basename(pos_data_folder)}, Weights: {os.path.basename(pos_weights_folder)}')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"POS loss plot saved to {output_plot_path}")
    plt.close()


def plot_quad_loss_curves(quad_data_folder, quad_metadata_path, 
                         quad_weights_config, # Dict: {'con5': path, 'con30': path, ...}
                         output_plot_prefix, model_seq_len, device_str,
                         hidden_size=QUAD_DEFAULT_HIDDEN_SIZE):
    device = torch.device(device_str)

    # 1. Load and prepare validation data (once)
    try:
        val_real_seq_np, metadata = get_validation_batch_quad(quad_data_folder, quad_metadata_path, model_seq_len)
        val_in_seq, val_gt_seq = prepare_quad_data_for_model(val_real_seq_np, model_seq_len)
        val_in_seq, val_gt_seq = val_in_seq.to(device), val_gt_seq.to(device)
        num_data_channels = metadata.get("num_channels")
        if num_data_channels is None:
            raise ValueError("'num_channels' not found in metadata.")
    except Exception as e:
        print(f"Error loading or preparing QUAD validation data: {e}")
        return

    # 2. Initialize model (frame sizes depend on metadata)
    model = QuadAcLSTM(in_frame_size=num_data_channels, hidden_size=hidden_size, out_frame_size=num_data_channels)
    model.to(device)
    model.eval()

    for config_label, weights_folder in quad_weights_config.items():
        print(f"\nProcessing QUAD model configuration: {config_label} from weights in: {weights_folder}")
        
        current_condition_num = int(config_label.replace('con', ''))
        
        iterations = []
        hip_losses = []
        rot_losses = []
        total_losses = []

        weight_files = sorted(glob.glob(os.path.join(weights_folder, "*.weight")))
        if not weight_files:
            print(f"No .weight files found in {weights_folder} for config {config_label}")
            continue

        all_iterations_in_folder = sorted([extract_iteration(os.path.basename(wf)) for wf in weight_files if extract_iteration(os.path.basename(wf)) != -1])
        min_iter_in_folder = all_iterations_in_folder[0] if all_iterations_in_folder else -1

        for weight_file in weight_files:
            iteration = extract_iteration(os.path.basename(weight_file))
            if iteration == -1:
                print(f"Could not parse iteration from {weight_file}, skipping.")
                continue
            
            # Process only every 1000th iteration, but always process iteration 0 if present, or the first available one.
            if iteration % 1000 != 0 and iteration != 0 and iteration != min_iter_in_folder:
                # print(f"Skipping QUAD weight: {os.path.basename(weight_file)} (Iter: {iteration}) for {config_label} - not a multiple of 1000 or first iter")
                continue

            print(f"Processing QUAD weight: {os.path.basename(weight_file)} (Iteration: {iteration}) for {config_label}")
            try:
                model.load_state_dict(torch.load(weight_file, map_location=device))
            except Exception as e:
                print(f"Error loading weights from {weight_file}: {e}")
                continue

            with torch.no_grad():
                # pred_seq has shape (batch, model_seq_len, channels)
                pred_seq = model.forward(val_in_seq, condition_num=current_condition_num, groundtruth_num=QUAD_GROUNDTRUTH_NUM)
                # calculate_quad_loss_components expects pred_seq and target_lstm_seq (val_gt_seq here)
                loss_h, loss_r, loss_t = calculate_quad_loss_components(pred_seq, val_gt_seq)
                
                iterations.append(iteration)
                hip_losses.append(loss_h)
                rot_losses.append(loss_r)
                total_losses.append(loss_t)
        
        if not iterations:
            print(f"No losses calculated for QUAD model config {config_label}.")
            continue

        # Sort by iteration for correct plotting order
        sorted_plot_data = sorted(zip(iterations, hip_losses, rot_losses, total_losses))
        if not sorted_plot_data:
            print(f"No data to plot for {config_label}")
            continue
        iterations_sorted, hip_losses_sorted, rot_losses_sorted, total_losses_sorted = zip(*sorted_plot_data)

        # Plot for current configuration
        plt.figure(figsize=(12, 7))
        plt.plot(iterations_sorted, hip_losses_sorted, marker='o', linestyle='-', label='Hip Position Loss (MSE)')
        plt.plot(iterations_sorted, rot_losses_sorted, marker='s', linestyle='--', label='Rotation Loss (Quaternion)')
        plt.plot(iterations_sorted, total_losses_sorted, marker='^', linestyle=':', label='Total Loss')
        
        plt.title(f'QUAD Model Losses vs. Iterations ({config_label})\nData: {os.path.basename(quad_data_folder)}, Weights: {os.path.basename(weights_folder)}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_plot_path = f"{output_plot_prefix}_{config_label}.png"
        plt.savefig(output_plot_path)
        print(f"QUAD loss plot for {config_label} saved to {output_plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves for POS and QUAD ACLSTM models.")
    
    # Common arguments
    parser.add_argument('--model_seq_len', type=int, default=100, help='Sequence length used by the LSTM model.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu).')

    # POS model arguments
    parser.add_argument('--pos_data_folder', type=str, help='Path to training data for POS model (e.g., ../train_data_pos/salsa).')
    parser.add_argument('--pos_weights_folder', type=str, help='Path to POS model weights folder.')
    parser.add_argument('--pos_output_plot', type=str, default='pos_loss_plot.png', help='Path to save POS loss plot.')
    parser.add_argument('--pos_in_frame_size', type=int, default=POS_IN_FRAME_SIZE, help='Input frame size for POS model.')
    parser.add_argument('--pos_hidden_size', type=int, default=POS_HIDDEN_SIZE, help='Hidden size for POS model.')

    # QUAD model arguments
    parser.add_argument('--quad_data_folder', type=str, help='Path to training data for QUAD model (e.g., ../train_data_quad/salsa).')
    parser.add_argument('--quad_metadata_path', type=str, help='Path to metadata_quad.json for QUAD data.')
    parser.add_argument('--quad_weights_folder_con5', type=str, help='Path to QUAD model weights for con5.')
    parser.add_argument('--quad_weights_folder_con30', type=str, help='Path to QUAD model weights for con30.')
    parser.add_argument('--quad_weights_folder_con45', type=str, help='Path to QUAD model weights for con45.')
    parser.add_argument('--quad_output_plot_prefix', type=str, default='quad_loss_plot', help='Prefix for saving QUAD loss plots.')
    parser.add_argument('--quad_hidden_size', type=int, default=QUAD_DEFAULT_HIDDEN_SIZE, help='Hidden size for QUAD model.')

    args = parser.parse_args()

    # Plot POS loss if paths are provided
    if args.pos_data_folder and args.pos_weights_folder:
        plot_pos_loss_curve(args.pos_data_folder, args.pos_weights_folder, args.pos_output_plot,
                            args.model_seq_len, args.device, 
                            args.pos_in_frame_size, args.pos_hidden_size)
    else:
        print("POS data or weights folder not provided, skipping POS loss plotting.")

    # Plot QUAD loss if paths are provided
    quad_configs = {}
    if args.quad_weights_folder_con5:
        quad_configs['con5'] = args.quad_weights_folder_con5
    if args.quad_weights_folder_con30:
        quad_configs['con30'] = args.quad_weights_folder_con30
    if args.quad_weights_folder_con45:
        quad_configs['con45'] = args.quad_weights_folder_con45
    
    if args.quad_data_folder and args.quad_metadata_path and quad_configs:
        plot_quad_loss_curves(args.quad_data_folder, args.quad_metadata_path,
                              quad_configs, args.quad_output_plot_prefix,
                              args.model_seq_len, args.device, args.quad_hidden_size)
    else:
        print("QUAD data, metadata, or at least one weights folder (con5/con30/con45) not provided, skipping QUAD loss plotting.")

if __name__ == '__main__':
    main() 