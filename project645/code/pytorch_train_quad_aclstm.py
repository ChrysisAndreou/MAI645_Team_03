import os
import torch
import torch.nn as nn
import torch.nn.functional as F # Added for F.normalize in loss
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse
import json # For loading metadata
import read_bvh_hierarchy # For loading skeleton hierarchy for BVH conversion
import rotation_conversions # For quaternion to euler conversions

# Default values, will be overridden by metadata or args
DEFAULT_HIDDEN_SIZE = 1024

# Constants for hip channel processing
HIP_X_CHANNEL, HIP_Y_CHANNEL, HIP_Z_CHANNEL = 0, 1, 2

# Global parameters for acLSTM training (can be overridden by args if needed, but often fixed)
# four experiments 
# condition num
# Experiment Notes on Condition_num and Groundtruth_num:
# Three experiments were conducted to observe the effect on motion synthesis in synthesize_quad_motion.py:
# 1. Condition_num = 5, Groundtruth_num = 5: 
# 2. Condition_num = 30, Groundtruth_num = 5: 
# 3. Condition_num = 45, Groundtruth_num = 5: 
# These parameters are critical for how the model learns to transition between conditioned (ground truth) and generated sequences.
Condition_num = 5 # changed to 30, 45 depending on experiment
Groundtruth_num = 5 

# Global variable for non_end_bones, loaded once
NON_END_BONES_LIST = []
ORIGINAL_FRAME_TIME = 0.016667 # Default (60 FPS), will be updated from metadata

class acLSTM(nn.Module):
    def __init__(self, in_frame_size, hidden_size, out_frame_size):
        super(acLSTM, self).__init__()
        
        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size
        
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        c0 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).to(device))
        c1 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).to(device))
        c2 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).to(device))
        h0 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).to(device))
        h1 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).to(device))
        h2 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).to(device))
        return ([h0,h1,h2], [c0,c1,c2])
    
    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h0, (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h1, (vec_h[2], vec_c[2]))
        out_frame = self.decoder(vec_h2)
        return (out_frame, [vec_h0, vec_h1, vec_h2], [vec_c0, vec_c1, vec_c2])
        
    def get_condition_lst(self, condition_num, groundtruth_num, seq_len):
        num_cycles = seq_len // (condition_num + groundtruth_num) + 2 # Ensure enough elements
        gt_lst = np.ones((num_cycles, groundtruth_num))
        con_lst = np.zeros((num_cycles, condition_num))
        
        lst_parts = []
        for i in range(num_cycles):
            lst_parts.append(gt_lst[i])
            lst_parts.append(con_lst[i])
        
        return np.concatenate(lst_parts).reshape(-1)[:seq_len]

    def forward(self, real_seq, condition_num, groundtruth_num):
        batch_size = real_seq.size()[0]
        seq_len = real_seq.size()[1]
        
        condition_lst = self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        (vec_h, vec_c) = self.init_hidden(batch_size)
        
        out_seq_list = []
        # Initialize out_frame for the first conditioned step if seq starts with it
        device = next(self.parameters()).device
        out_frame = Variable(torch.FloatTensor(np.zeros((batch_size, self.out_frame_size))).to(device))
        
        for i in range(seq_len):
            if condition_lst[i] == 1:
                in_frame = real_seq[:,i]
            else:
                in_frame = out_frame # Use previously generated frame
            
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq_list.append(out_frame.unsqueeze(1))
        
        return torch.cat(out_seq_list, dim=1) # Shape: (batch, seq_len, frame_size)
    
    def calculate_loss(self, pred_seq, groundtruth_seq):
        # pred_seq and groundtruth_seq are [batch_size, seq_len, self.out_frame_size]
        # First 3 channels: hip_pos (X_diff_scaled, Y_abs_scaled, Z_diff_scaled)
        # Remaining channels: quaternions (wxyz) per joint, absolute.
        
        hip_pos_pred = pred_seq[..., :3]
        hip_pos_gt = groundtruth_seq[..., :3]
        
        quat_pred = pred_seq[..., 3:]
        quat_gt = groundtruth_seq[..., 3:]

        loss_fn_mse = nn.MSELoss()
        loss_hip = loss_fn_mse(hip_pos_pred, hip_pos_gt)

        # Quaternion Loss for rotations
        # Reshape to (total_quaternions_in_batch, 4)
        quat_pred_reshaped = quat_pred.reshape(-1, 4)
        quat_gt_reshaped = quat_gt.reshape(-1, 4)

        # Normalize quaternions to ensure they are unit (model output might not be perfect)
        quat_pred_norm = F.normalize(quat_pred_reshaped, p=2, dim=-1)
        quat_gt_norm = F.normalize(quat_gt_reshaped, p=2, dim=-1)

        # Dot product for angle: sum(q1_i * q2_i) for each quaternion pair
        dot_product = torch.sum(quat_pred_norm * quat_gt_norm, dim=-1)
        
        # Clamp absolute dot product for arccos numerical stability (input to acos must be [-1, 1])
        # Since we use abs(), input to acos will be [0, 1]. Clamp max to avoid nan from float precision.
        abs_dot_product_clamped = torch.clamp(torch.abs(dot_product), min=0.0, max=1.0 - 1e-7)
        
        # Quaternion Loss formula: QL = 2 * arccos(|angle|)
        loss_quat_values = 2.0 * torch.acos(abs_dot_product_clamped)
        loss_rot_ql = torch.mean(loss_quat_values)
        
        # Total loss - potentially weight components if needed
        total_loss = loss_hip + loss_rot_ql 
        return total_loss

def convert_lstm_output_to_bvh_frames(scaled_lstm_output_np, hip_pos_scale_factor, num_data_channels_from_metadata):
    # scaled_lstm_output_np: [seq_len, num_data_channels]
    # Contains: scaled hip_x_diff, scaled hip_y_abs, scaled hip_z_diff, then quaternions (wxyz)
    
    global NON_END_BONES_LIST, ORIGINAL_FRAME_TIME

    bvh_frames_list = []
    num_frames = scaled_lstm_output_np.shape[0]

    current_abs_scaled_x = 0.0
    current_abs_scaled_z = 0.0

    num_rotating_joints = (num_data_channels_from_metadata - 3) // 4
    if (num_data_channels_from_metadata - 3) % 4 != 0:
        print(f"Warning: Number of data channels ({num_data_channels_from_metadata}) for quaternions seems incorrect.")
        # Fallback or raise error

    for frame_idx in range(num_frames):
        lstm_frame = scaled_lstm_output_np[frame_idx]
        bvh_frame_channels = []
        
        # 1. Hip Position Reconstruction
        # Accumulate scaled X,Z differences
        current_abs_scaled_x += lstm_frame[HIP_X_CHANNEL]
        current_abs_scaled_z += lstm_frame[HIP_Z_CHANNEL]
        # Y is absolute scaled
        
        # Unscale hip positions
        unscaled_hip_x = current_abs_scaled_x / hip_pos_scale_factor
        unscaled_hip_y = lstm_frame[HIP_Y_CHANNEL] / hip_pos_scale_factor
        unscaled_hip_z = current_abs_scaled_z / hip_pos_scale_factor
        bvh_frame_channels.extend([unscaled_hip_x, unscaled_hip_y, unscaled_hip_z])
        
        # 2. Quaternion to Euler Conversion
        quat_feature_cursor = 3
        
        # Hip rotation (ZYX order)
        hip_quat_wxyz = torch.tensor(lstm_frame[quat_feature_cursor : quat_feature_cursor+4], dtype=torch.float32)
        hip_rot_matrix = rotation_conversions.quaternion_to_matrix(hip_quat_wxyz.unsqueeze(0)) # Add batch dim
        hip_euler_rad_zyx = rotation_conversions.matrix_to_euler_angles(hip_rot_matrix, "ZYX").squeeze(0) # Remove batch dim
        hip_euler_deg_zyx = np.rad2deg(hip_euler_rad_zyx.numpy())
        bvh_frame_channels.extend(hip_euler_deg_zyx)
        quat_feature_cursor += 4
        
        # Other joint rotations (ZXY order)
        # num_rotating_joints includes hip, so loop (num_rotating_joints - 1) times
        if len(NON_END_BONES_LIST) != (num_rotating_joints -1):
             print(f"Warning: Mismatch between NON_END_BONES_LIST ({len(NON_END_BONES_LIST)}) and expected other joints ({num_rotating_joints - 1})")

        for _ in range(num_rotating_joints - 1): # Iterate for non-hip rotating joints
            if quat_feature_cursor + 4 > len(lstm_frame):
                # This case should not happen if num_data_channels_from_metadata is correct
                print(f"Error: Trying to access quaternion data out of bounds. Cursor: {quat_feature_cursor}, Frame length: {len(lstm_frame)}")
                # Pad with zeros for this joint's rotation if error occurs
                bvh_frame_channels.extend([0.0, 0.0, 0.0]) 
                break 
            
            joint_quat_wxyz = torch.tensor(lstm_frame[quat_feature_cursor : quat_feature_cursor+4], dtype=torch.float32)
            joint_rot_matrix = rotation_conversions.quaternion_to_matrix(joint_quat_wxyz.unsqueeze(0))
            joint_euler_rad_zxy = rotation_conversions.matrix_to_euler_angles(joint_rot_matrix, "ZXY").squeeze(0)
            joint_euler_deg_zxy = np.rad2deg(joint_euler_rad_zxy.numpy())
            bvh_frame_channels.extend(joint_euler_deg_zxy)
            quat_feature_cursor += 4
            
        bvh_frames_list.append(bvh_frame_channels)
        
    return np.array(bvh_frames_list, dtype=np.float32)


def train_one_iteration(real_seq_np, model, optimizer, iteration, args, metadata):
    # real_seq_np is [batch, sampling_seq_len, num_data_channels]
    # Hip X,Z are relative to their original first frame & scaled. Hip Y is scaled absolute.
    # Quaternions (wxyz) are absolute.

    # Prepare sequence for LSTM:
    # Hip X and Z values need to be converted to frame-to-frame differences (still scaled).
    # Hip Y and all quaternion rotations remain as they are.
    dif_hip_x_scaled = real_seq_np[:, 1:, HIP_X_CHANNEL] - real_seq_np[:, :-1, HIP_X_CHANNEL]
    dif_hip_z_scaled = real_seq_np[:, 1:, HIP_Z_CHANNEL] - real_seq_np[:, :-1, HIP_Z_CHANNEL]
    
    real_seq_processed_for_lstm_np = real_seq_np[:, :-1].copy() 
    real_seq_processed_for_lstm_np[:, :, HIP_X_CHANNEL] = dif_hip_x_scaled
    real_seq_processed_for_lstm_np[:, :, HIP_Z_CHANNEL] = dif_hip_z_scaled

    device = next(model.parameters()).device
    real_seq_cuda = Variable(torch.FloatTensor(real_seq_processed_for_lstm_np)).to(device)
 
    # Input to model: first `args.seq_len` frames of the processed sequence.
    # Target for model: next `args.seq_len` frames (offset by 1) from the processed sequence.
    in_lstm_seq = real_seq_cuda[:, :-1]      
    target_lstm_seq = real_seq_cuda[:, 1:] 
    
    optimizer.zero_grad()
    pred_seq = model.forward(in_lstm_seq, Condition_num, Groundtruth_num) 
    loss = model.calculate_loss(pred_seq, target_lstm_seq)
    loss.backward()
    optimizer.step()
    
    if iteration % args.print_loss_iter == 0:
        print(f"########### iter {iteration:07d} ######################")
        print(f"loss: {loss.item():.6f}") 
    
    if iteration % args.save_bvh_iter == 0 and args.standard_bvh_file:
        # Get first sequence from batch for saving
        gt_to_save_lstm_out_np = target_lstm_seq[0].data.cpu().numpy() # LSTM target format
        pred_to_save_lstm_out_np = pred_seq[0].data.cpu().numpy()     # LSTM output format

        hip_pos_scale_factor_from_meta = metadata.get("hip_pos_scale_factor", 0.01) # Fallback if not in meta
        num_channels_from_meta = metadata.get("num_channels", model.out_frame_size)

        # Convert to BVH format
        bvh_gt = convert_lstm_output_to_bvh_frames(gt_to_save_lstm_out_np, hip_pos_scale_factor_from_meta, num_channels_from_meta)
        bvh_pred = convert_lstm_output_to_bvh_frames(pred_to_save_lstm_out_np, hip_pos_scale_factor_from_meta, num_channels_from_meta)

        global ORIGINAL_FRAME_TIME
        read_bvh.write_frames(args.standard_bvh_file, 
                              os.path.join(args.write_bvh_motion_folder, f"{iteration:07d}_gt.bvh"), 
                              bvh_gt, frame_time_override=ORIGINAL_FRAME_TIME)
        read_bvh.write_frames(args.standard_bvh_file, 
                              os.path.join(args.write_bvh_motion_folder, f"{iteration:07d}_out.bvh"), 
                              bvh_pred, frame_time_override=ORIGINAL_FRAME_TIME)


def get_dance_selection_indices(dances):
    # Creates a list where each dance's index appears proportionally to its length.
    # This is used for weighted random sampling of dances.
    index_lst = []
    for i, dance_frames in enumerate(dances):
        # Weight by length, ensure minimum occurrences for shorter dances
        occurrences = max(1, dance_frames.shape[0] // 100) 
        index_lst.extend([i] * occurrences)
    return index_lst

def load_dances_and_metadata(dance_folder, metadata_path):
    dances = []
    metadata = {}
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {e}. Using defaults.")


    dance_files = sorted([f for f in os.listdir(dance_folder) if f.endswith(".npy")])
    print(f'Loading motion files from {dance_folder}...')
    for dance_file in dance_files:
        try:
            dance = np.load(os.path.join(dance_folder, dance_file))
            # Basic validation: must be 2D, have some frames, and match channel count from metadata (if available)
            expected_channels = metadata.get("num_channels")
            if dance.ndim == 2 and dance.shape[0] > 0 and \
               (expected_channels is None or dance.shape[1] == expected_channels):
                dances.append(dance)
            else:
                print(f"Warning: Skipping invalid dance file: {dance_file}. Shape: {dance.shape}, Expected channels: {expected_channels}")
        except Exception as e:
            print(f"Warning: Could not load dance file {dance_file}: {e}")
    print(f'{len(dances)} motion files loaded.')
    return dances, metadata
    
def train(dances, metadata, args):
    # sampling_seq_len includes one extra frame for input features (diffs) and one for target
    sampling_seq_len = args.seq_len + 2 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    num_data_channels = metadata.get("num_channels")
    if num_data_channels is None:
        print("Error: 'num_channels' not found in metadata. Cannot initialize model.")
        return
    
    hip_pos_scale_factor = metadata.get("hip_pos_scale_factor", 0.01) # Default if not in metadata

    global ORIGINAL_FRAME_TIME
    ORIGINAL_FRAME_TIME = metadata.get("frame_time_original", 0.016667) # Default if not in metadata

    model = acLSTM(in_frame_size=num_data_channels, hidden_size=args.hidden_size, out_frame_size=num_data_channels)
    
    if args.read_weight_path:
        if os.path.exists(args.read_weight_path):
            print(f"Loading weights from: {args.read_weight_path}")
            model.load_state_dict(torch.load(args.read_weight_path, map_location=device))
        else:
            print(f"Warning: Weight path {args.read_weight_path} not found. Starting from scratch.")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    
    dance_selection_indices = get_dance_selection_indices(dances)
    if not dance_selection_indices:
        print("Error: No dances to train on after processing lengths.")
        return
    
    # speed: factor to adjust sampling rate if dance data FPS differs from assumed model FPS (e.g., 30 FPS)
    # For quaternion data, assume it's already at a consistent FPS from preprocessing or use metadata.
    # If metadata has frame_time, use it. Default to 60 FPS if not available.
    source_fps = 1.0 / ORIGINAL_FRAME_TIME if ORIGINAL_FRAME_TIME > 0 else 60.0
    model_processing_fps = 30.0 # Standard assumption from other scripts
    speed = source_fps / model_processing_fps
    
    for iteration in range(args.total_iterations):
        dance_batch_list = []
        for _ in range(args.batch_size):
            dance_idx = random.choice(dance_selection_indices)
            dance = dances[dance_idx]
            dance_len = dance.shape[0]
            
            # Ensure enough frames for sampling, including speed factor
            min_req_len_at_source_fps = int(sampling_seq_len * speed) + 20 # Margin for start_id offset
            
            if dance_len < min_req_len_at_source_fps:
                # Simple padding if too short: repeat last frame after sampling what's available
                indices = (np.arange(int(sampling_seq_len * speed)) * speed).astype(int)
                indices = np.clip(indices, 0, dance_len - 1) # Clip indices to be within dance_len
                sample_seq_at_source_fps = dance[indices[:sampling_seq_len]] # Ensure exact length for model
                
                if sample_seq_at_source_fps.shape[0] < sampling_seq_len:
                    padding_count = sampling_seq_len - sample_seq_at_source_fps.shape[0]
                    padding_frames = np.tile(sample_seq_at_source_fps[-1], (padding_count, 1))
                    sample_seq = np.vstack((sample_seq_at_source_fps, padding_frames))
                else:
                    sample_seq = sample_seq_at_source_fps

            else: # Dance is long enough
                max_start_id_at_source_fps = dance_len - int(sampling_seq_len * speed) - 10
                start_id_at_source_fps = random.randint(10, max_start_id_at_source_fps if max_start_id_at_source_fps >=10 else 10)
                
                sample_indices = (np.arange(sampling_seq_len) * speed + start_id_at_source_fps).astype(int)
                sample_seq = dance[sample_indices]

            # Data Augmentation for hip position (scaled translation)
            aug_T_scaled = [0.1*(random.random()-0.5) * hip_pos_scale_factor, 
                            0.0,  # No Y augmentation generally for ground contact
                            0.1*(random.random()-0.5) * hip_pos_scale_factor]
            
            sample_seq_augmented = sample_seq.copy()
            sample_seq_augmented[:, HIP_X_CHANNEL] += aug_T_scaled[0]
            # sample_seq_augmented[:, HIP_Y_CHANNEL] += aug_T_scaled[1] # Y typically not augmented
            sample_seq_augmented[:, HIP_Z_CHANNEL] += aug_T_scaled[2]
            
            dance_batch_list.append(sample_seq_augmented)
        
        dance_batch_np = np.array(dance_batch_list)
       
        train_one_iteration(dance_batch_np, model, optimizer, iteration, args, metadata)

        if iteration > 0 and iteration % args.save_model_iter == 0:
            path = os.path.join(args.write_weight_folder, f"{iteration:07d}.weight")
            torch.save(model.state_dict(), path)
            print(f"Saved model weights to {path}")

def main():
    parser = argparse.ArgumentParser(description="Train ACLSTM for Quaternion Motion with Normalized Hip")
    parser.add_argument('--dances_folder', type=str, required=True, help='Path for normalized training data (.npy files for a specific dance type, e.g., salsa)')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata_quad.json for the dances_folder')
    parser.add_argument('--write_weight_folder', type=str, required=True, help='Path to store model checkpoints')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to store sample generated BVH during training')
    parser.add_argument('--standard_bvh_file', type=str, required=True, help='Path to standard.bvh for hierarchy (required for saving BVH)')
    parser.add_argument('--read_weight_path', type=str, default="", help='Path to pre-trained model to continue training')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_HIDDEN_SIZE, help='LSTM hidden size')
    parser.add_argument('--seq_len', type=int, default=100, help='LSTM sequence length for training (model sees this many steps)')
    parser.add_argument('--total_iterations', type=int, default=50000, help='Total training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    
    parser.add_argument('--print_loss_iter', type=int, default=100, help='Print loss every N iterations')
    parser.add_argument('--save_model_iter', type=int, default=1000, help='Save model checkpoint every N iterations')
    parser.add_argument('--save_bvh_iter', type=int, default=1000, help='Save sample BVH files every N iterations')

    args = parser.parse_args()

    os.makedirs(args.write_weight_folder, exist_ok=True)
    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)
    
    if not os.path.exists(args.standard_bvh_file):
        print(f"Error: Standard BVH file not found at {args.standard_bvh_file}. BVH saving will fail.")
        return
    if not os.path.exists(args.metadata_path):
        print(f"Error: Metadata file not found at {args.metadata_path}.")
        return

    global NON_END_BONES_LIST
    try:
        _, NON_END_BONES_LIST = read_bvh_hierarchy.read_bvh_hierarchy(args.standard_bvh_file)
        print(f"Loaded {len(NON_END_BONES_LIST)} non-end-site bones from hierarchy.")
    except Exception as e:
        print(f"Error loading BVH hierarchy from {args.standard_bvh_file}: {e}")
        return

    dances_data, metadata = load_dances_and_metadata(args.dances_folder, args.metadata_path)
    if not dances_data:
        print(f"No data found in {args.dances_folder} or failed to load. Exiting.")
        return
    if not metadata.get("num_channels"): # Critical check
        print("Error: num_channels not in metadata. Exiting.")
        return

    train(dances_data, metadata, args)

if __name__ == '__main__':
    main()
