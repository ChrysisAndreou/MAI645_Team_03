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

# Imports from Euler training script
from gcvProject.project645.code.pytorch_train_euler_aclstm_for_calculating_loss_to_device import acLSTM as EulerAcLSTM
from gcvProject.project645.code.pytorch_train_euler_aclstm_for_calculating_loss_to_device import load_dances as load_euler_dances
from gcvProject.project645.code.pytorch_train_euler_aclstm_for_calculating_loss_to_device import HIP_X_CHANNEL, HIP_Z_CHANNEL
# Globals from pytorch_train_euler_aclstm that might be needed
EULER_CONDITION_NUM = 5 # Default from pytorch_train_euler_aclstm.py
EULER_GROUNDTRUTH_NUM = 5 # Default from pytorch_train_euler_aclstm.py
DEFAULT_EULER_HIDDEN_SIZE = 1024 # Common default

# --- Helper function to extract iteration from filename (reused) ---
def extract_iteration(filename):
    match = re.search(r'(\d{7,})\.weight', filename)
    if match:
        return int(match.group(1))
    return -1

# --- EULER Model Helper Functions ---

def get_validation_batch_euler(data_folder, model_seq_len, batch_size=1):
    """Loads the first dance and takes the first possible segment for validation for Euler model."""
    dances = load_euler_dances(data_folder)
    if not dances:
        raise ValueError(f"No Euler dance data found in {data_folder}")
    
    dance = dances[0] # Use the first dance
    # The Euler training script uses model_seq_len + 2 for sampling_seq_len
    # to prepare input and target of model_seq_len
    sampling_seq_len = model_seq_len + 2 

    if dance.shape[0] < sampling_seq_len:
        raise ValueError(f"First Euler dance in {data_folder} is too short ({dance.shape[0]} frames) for "
                         f"sampling_seq_len {sampling_seq_len}")
    
    # Take the first segment. No speed resampling or augmentation for validation.
    raw_dance_segment_np = dance[0:sampling_seq_len, :]
    return np.expand_dims(raw_dance_segment_np, axis=0) # Add batch dimension


def prepare_euler_data_for_model(real_seq_np_batch, model_seq_len):
    """Prepares Euler data batch for model input and ground truth, similar to train_one_iteration."""
    # real_seq_np_batch has shape (batch, sampling_seq_len, features)
    # sampling_seq_len = model_seq_len + 2
    
    # Hip X,Z difference calculation (as in train_one_iteration from pytorch_train_euler_aclstm.py)
    dif_hip_x_scaled = real_seq_np_batch[:, 1:, HIP_X_CHANNEL] - real_seq_np_batch[:, :-1, HIP_X_CHANNEL]
    dif_hip_z_scaled = real_seq_np_batch[:, 1:, HIP_Z_CHANNEL] - real_seq_np_batch[:, :-1, HIP_Z_CHANNEL]
    
    # real_seq_processed_for_lstm_np has shape (batch, model_seq_len + 1, features)
    real_seq_processed_for_lstm_np = real_seq_np_batch[:, :-1].copy() 
    real_seq_processed_for_lstm_np[:, :, HIP_X_CHANNEL] = dif_hip_x_scaled
    real_seq_processed_for_lstm_np[:, :, HIP_Z_CHANNEL] = dif_hip_z_scaled
    
    # Input to model: first `model_seq_len` frames of the processed sequence.
    # Target for model: next `model_seq_len` frames (offset by 1) of the processed sequence.
    in_lstm_seq_np = real_seq_processed_for_lstm_np[:, :-1]      # Shape: (batch, model_seq_len, channels)
    target_lstm_seq_np = real_seq_processed_for_lstm_np[:, 1:]   # Shape: (batch, model_seq_len, channels)
        
    in_lstm_seq_torch = torch.FloatTensor(in_lstm_seq_np)
    target_lstm_seq_torch = torch.FloatTensor(target_lstm_seq_np)
    
    return in_lstm_seq_torch, target_lstm_seq_torch

# --- Plotting Function for Euler Model ---

def plot_euler_loss_curve(euler_data_folder, euler_weights_folder, output_plot_path, 
                          model_seq_len, device_str, 
                          hidden_size=DEFAULT_EULER_HIDDEN_SIZE, in_frame_size_arg=None):
    print(f"Plotting Euler model loss from weights in: {euler_weights_folder}")
    device = torch.device(device_str)

    # 1. Load and prepare validation data (once)
    try:
        val_real_seq_np = get_validation_batch_euler(euler_data_folder, model_seq_len)
        # Determine in_frame_size from data if not provided
        if in_frame_size_arg is None:
            current_in_frame_size = val_real_seq_np.shape[2]
            print(f"Auto-detected Euler in_frame_size: {current_in_frame_size}")
        else:
            current_in_frame_size = in_frame_size_arg
            print(f"Using provided Euler in_frame_size: {current_in_frame_size}")

        val_in_seq, val_gt_seq = prepare_euler_data_for_model(val_real_seq_np, model_seq_len)
        val_in_seq, val_gt_seq = val_in_seq.to(device), val_gt_seq.to(device)
    except Exception as e:
        print(f"Error loading or preparing Euler validation data: {e}")
        return

    # 2. Initialize model
    # The EulerAcLSTM takes in_frame_size, hidden_size, out_frame_size. out_frame_size = in_frame_size
    model = EulerAcLSTM(in_frame_size=current_in_frame_size, hidden_size=hidden_size, out_frame_size=current_in_frame_size)
    model.to(device)
    model.eval()

    # 3. Iterate through weights, calculate loss
    iterations = []
    losses = []
    
    weight_files = sorted(glob.glob(os.path.join(euler_weights_folder, "*.weight")))
    if not weight_files:
        print(f"No .weight files found in {euler_weights_folder}")
        return

    all_iterations_in_folder = sorted([extract_iteration(os.path.basename(wf)) for wf in weight_files if extract_iteration(os.path.basename(wf)) != -1])
    min_iter_in_folder = all_iterations_in_folder[0] if all_iterations_in_folder else -1

    for weight_file in weight_files:
        iteration = extract_iteration(os.path.basename(weight_file))
        if iteration == -1:
            print(f"Could not parse iteration from {weight_file}, skipping.")
            continue
        
        # Process only every 1000th iteration, but always process iteration 0 if present, or the first available one.
        if iteration % 1000 != 0 and iteration != 0 and iteration != min_iter_in_folder :
            # print(f"Skipping Euler weight: {os.path.basename(weight_file)} (Iteration: {iteration}) - not a multiple of 1000 or first")
            continue

        print(f"Processing Euler weight: {os.path.basename(weight_file)} (Iteration: {iteration})")
        try:
            model.load_state_dict(torch.load(weight_file, map_location=device))
        except Exception as e:
            print(f"Error loading weights from {weight_file}: {e}")
            continue
            
        with torch.no_grad():
            # pred_seq has shape (batch, model_seq_len, features)
            predict_seq = model.forward(val_in_seq, condition_num=EULER_CONDITION_NUM, groundtruth_num=EULER_GROUNDTRUTH_NUM)
            loss = model.calculate_loss(predict_seq, val_gt_seq) # calculate_loss is defined in EulerAcLSTM
            losses.append(loss.item())
            iterations.append(iteration)

    if not iterations:
        print("No losses calculated for Euler model.")
        return

    # 4. Plot
    plt.figure(figsize=(10, 6))
    # Sort by iteration for correct plotting order
    sorted_pairs = sorted(zip(iterations, losses))
    iterations_sorted, losses_sorted = zip(*sorted_pairs) if sorted_pairs else ([],[])

    plt.plot(iterations_sorted, losses_sorted, marker='o', linestyle='-')
    plt.title(f'Euler Model Loss vs. Iterations\nData: {os.path.basename(euler_data_folder)}, Weights: {os.path.basename(euler_weights_folder)}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Hip MSE + Rotation AD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Euler loss plot saved to {output_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves for Euler ACLSTM models.")
    
    # Common arguments
    parser.add_argument('--model_seq_len', type=int, default=100, help='Sequence length used by the LSTM model.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu).')

    # Euler model arguments
    parser.add_argument('--euler_data_folder', type=str, required=True, help='Path to training data for Euler model (e.g., ../train_data_euler/salsa).')
    parser.add_argument('--euler_weights_folder', type=str, required=True, help='Path to Euler model weights folder.')
    parser.add_argument('--euler_output_plot', type=str, default='euler_loss_plot.png', help='Path to save Euler loss plot.')
    parser.add_argument('--euler_in_frame_size', type=int, default=None, help='Input frame size for Euler model. Auto-detect if None. (e.g. 132)')
    parser.add_argument('--euler_hidden_size', type=int, default=DEFAULT_EULER_HIDDEN_SIZE, help='Hidden size for Euler model.')

    args = parser.parse_args()

    if args.euler_data_folder and args.euler_weights_folder:
        plot_euler_loss_curve(args.euler_data_folder, args.euler_weights_folder, args.euler_output_plot,
                              args.model_seq_len, args.device,
                              args.euler_hidden_size, args.euler_in_frame_size)
    else:
        print("Euler data or weights folder not provided, skipping Euler loss plotting.")

if __name__ == '__main__':
    main() 