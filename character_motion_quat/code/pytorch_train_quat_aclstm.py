import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import read_bvh
import argparse
import json
import transforms3d.euler as t3d_euler

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps"); 
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda"); 
    print("Using CUDA device.")
else:
    device = torch.device("cpu"); 
    print("Using CPU device.")

# Quat training constants -- Global Vabriables --
Num_translation_channels = 3 # Usually 3 (X, Y, Z)
Viz_rotation_order = 'rzyx'
Epsilon = 1e-7 # For acos in loss
Weight_translation = 0.01
Standard_bvh_file = "train_data_bvh/standard.bvh"

Condition_num = 5
Groundtruth_num = 5

class acLSTM(nn.Module):
    # Default in_frame_size will be overridden by args
    def __init__(self, in_frame_size=175, hidden_size=1024, out_frame_size=175):
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size 
        
        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    #output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        #c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device)) # Changed (.cuda() to .to(device))
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        return  ([h0,h1,h2], [c0,c1,c2])
    
    #in_frame b*In_frame_size
    #vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    #out_frame b*In_frame_size
    #vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):

        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
     
        out_frame = self.decoder(vec_h2) #out b*150
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    
    #in cuda tensor real_seq: b*seq_len*frame_size
    #out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        
        batch=real_seq.size()[0]
        seq_len=real_seq.size()[1]
        
        condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch,1))).to(device))

        out_frame=torch.autograd.Variable(torch.FloatTensor(np.zeros((batch,self.out_frame_size))).to(device))
        
        
        for i in range(seq_len):
            
            if(condition_lst[i]==1):##input groundtruth frame
                in_frame=real_seq[:,i]
            else:
                in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
    
        return out_seq[:, 1: out_seq.size()[1]]
    
    # In class acLSTM:
    def calculate_loss(self, pred_seq_flat, true_seq_flat, in_frame_size):

        batch_size = pred_seq_flat.size(0)
        seq_len_model_output = pred_seq_flat.size(1) // in_frame_size

        pred_seq = pred_seq_flat.view(batch_size, seq_len_model_output, in_frame_size)
        true_seq = true_seq_flat.view(batch_size, seq_len_model_output, in_frame_size)

        # 1. Translation Loss (MSE on the first Num_translation_channels)
        # These channels represent dX, Y_abs_scaled, dZ
        mse_loss = nn.MSELoss()
        loss_translation = torch.tensor(0.0, device=pred_seq.device)
        if Num_translation_channels > 0:
            pred_trans = pred_seq[..., :Num_translation_channels]
            true_trans = true_seq[..., :Num_translation_channels]
            loss_translation = mse_loss(pred_trans, true_trans)

        # 2. Quaternion Loss (Angular difference for all subsequent quaternion blocks)
        loss_quaternion = torch.tensor(0.0, device=pred_seq.device)
        num_quat_value_channels = in_frame_size - Num_translation_channels
        
        if num_quat_value_channels > 0:
            
            pred_quats_all = pred_seq[..., Num_translation_channels:]
            true_quats_all = true_seq[..., Num_translation_channels:]

            # Reshape to process all quaternions: (batch_size * seq_len * num_joints, 4)
            pred_quats_flat = pred_quats_all.reshape(-1, 4)
            true_quats_flat = true_quats_all.reshape(-1, 4)

            # Normalize quaternions to ensure they are unit quaternions
            pred_quats_norm = F.normalize(pred_quats_flat, p=2, dim=-1, eps=Epsilon)
            true_quats_norm = F.normalize(true_quats_flat, p=2, dim=-1, eps=Epsilon)

            # Dot product for angular difference:
            # For unit quaternions q1 and q2, the angle 'a' between them is given by:
            # cos(a/2) = |q1 . q2|  (absolute value of dot product for shortest angle)
            # So, a = 2 * acos(|q1 . q2|)
            dot_product = torch.sum(pred_quats_norm * true_quats_norm, dim=-1)
            
            # Clamp the dot product to the range [-1+eps, 1-eps] to avoid NaN from acos
            # due to floating point inaccuracies.
            abs_dot_product = torch.abs(dot_product)
            clamped_abs_dot = torch.clamp(abs_dot_product, -1.0 + Epsilon, 1.0 - Epsilon)
            
            angle_diff_radians = 2.0 * torch.acos(clamped_abs_dot)
            loss_quaternion = torch.mean(angle_diff_radians) # Mean angular error

        # Combine losses
        translation_loss_weight = 1.0
        quaternion_loss_weight = 2.0

        total_loss = (translation_loss_weight * loss_translation) + \
                    (quaternion_loss_weight * loss_quaternion)
        
        return total_loss


# real_seq_np: batch * (seq_len_from_data_loader) * quat_frame_size
def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, seq_len, in_frame_size, joint_rotation_orders, rotating_joint_names_in_order,  print_loss=True, save_bvh_motion=True): # Removed print_loss, save_bvh_motion flags, use args

    # Data prep: set hip_x and hip_z as the difference (velocity)
    # real_seq_np contains absolute (scaled) translations and absolute quaternions
    dif = real_seq_np[:, 1:] - real_seq_np[:, :-1] # Differences for all channels
    
    # processed_seq_np will be fed to model (N-1 frames)
    # Target will be processed_seq_np shifted (N-1 frames)
    processed_seq_np = real_seq_np[:, :-1].copy() # Takes N-1 frames from original N frames

    if Num_translation_channels >=1: processed_seq_np[:,:,0] = dif[:,:,0] # dX for hip
    # Y translation (channel 1) remains absolute scaled
    if Num_translation_channels >=3: processed_seq_np[:,:,2] = dif[:,:,2] # dZ for hip
    # Quaternion channels remain absolute

    # Convert to PyTorch Variables (mimicking pos_aclstm style)
    # Processed sequence for model input

    # --- CHANGE 1: Correct Input/Target Slicing ---
    # Model input sequence (length: arg_model_process_seq_len)
    # Takes frames 0 to L-1 from processed_seq_np (which has L+1 frames)
    model_input_data_np = processed_seq_np[:, :seq_len]
    model_input_tensor = Variable(torch.from_numpy(model_input_data_np).float().to(device))
 
    # Target sequence for loss (length: arg_model_process_seq_len)
    # Takes frames 1 to L from processed_seq_np.
    # So, target_data_np[t] is the state after model_input_data_np[t] occurred.
    target_data_np = processed_seq_np[:, 1:seq_len+1]
    target_data_flat_tensor = Variable(torch.from_numpy(target_data_np).float().to(device)).view(real_seq_np.shape[0], -1)
    
    # Model's forward pass
    # Global Condition_num, Groundtruth_num are used here as in pos_aclstm
    predict_seq_flat = model.forward(model_input_tensor, Condition_num, Groundtruth_num)
    
    optimizer.zero_grad()

    loss = model.calculate_loss(predict_seq_flat, target_data_flat_tensor, in_frame_size)

    loss.backward()
    optimizer.step()
    
    if (print_loss==True) :
        print(f"###########iter {iteration:07d} # Loss: {loss.item():.6f} ###########")
    
    if (save_bvh_motion==True):
        # --- Ground Truth BVH ---
        # predict_groundtruth_seq_flat is already (dX, Y_abs_scaled, dZ, Quats_abs)
        gt_seq_for_viz_np = target_data_np[0].copy() # take first item in batch, (args.seq_len, quat_frame_size)
        
        # Accumulate hip XZ velocities to get absolute positions for BVH
        # Start accumulation from the corresponding frame in the *original absolute input*
        # real_seq_np[0, 0, :] is the first absolute frame corresponding to the start of processed_seq_np
        current_x_gt = real_seq_np[0, 1, 0] if Num_translation_channels >=1 else 0.0
        current_z_gt = real_seq_np[0, 1, 2] if Num_translation_channels >=3 else 0.0

        gt_bvh_abs_frames = np.zeros_like(gt_seq_for_viz_np)
        if Num_translation_channels >= 2: gt_bvh_abs_frames[:,1] = gt_seq_for_viz_np[:,1] # Copy Y
        if in_frame_size > Num_translation_channels:      # Copy Quaternions
            gt_bvh_abs_frames[:, Num_translation_channels:] = gt_seq_for_viz_np[:, Num_translation_channels:]

        for frame_idx in range(seq_len):
            if Num_translation_channels >=1:
                current_x_gt += gt_seq_for_viz_np[frame_idx, 0] 
                gt_bvh_abs_frames[frame_idx, 0] = current_x_gt
            if Num_translation_channels >=3:
                current_z_gt += gt_seq_for_viz_np[frame_idx, 2] 
                gt_bvh_abs_frames[frame_idx, 2] = current_z_gt
        
        # Convert Quaternions to Euler for BVH (num_quat_ch, etc. calculations)
        num_quat_ch = in_frame_size - Num_translation_channels
        num_rot_j = num_quat_ch // 4
        num_euler_ch_out = Num_translation_channels + num_rot_j * 3
        gt_euler_deg_np = np.zeros((seq_len, num_euler_ch_out))

        if Num_translation_channels > 0:
            gt_euler_deg_np[:, :Num_translation_channels] = gt_bvh_abs_frames[:, :Num_translation_channels]
        
        quat_ch_offset = Num_translation_channels
        euler_ch_offset = Num_translation_channels

        for r_j, joint_name in enumerate(rotating_joint_names_in_order):
            rot_order = joint_rotation_orders.get(joint_name, 'rzyx')  # Fallback to 'rzyx'
            q_start = quat_ch_offset + r_j * 4
            e_start = euler_ch_offset + r_j * 3
            quats_joint = gt_bvh_abs_frames[:, q_start : q_start + 4]
            for frame_k in range(seq_len):
                q_norm = quats_joint[frame_k] / (np.linalg.norm(quats_joint[frame_k]) + Epsilon)
                try:
                    axes_param = 's' + rot_order[1:] if rot_order.startswith('r') else rot_order
                    angles_rad = t3d_euler.quat2euler(q_norm, axes=axes_param)
                    gt_euler_deg_np[frame_k, e_start:e_start+3] = np.rad2deg(angles_rad)
                except Exception:
                    gt_euler_deg_np[frame_k, e_start:e_start+3] = 0.0
        
        if Weight_translation != 1.0 and Num_translation_channels > 0:
            gt_euler_deg_np[:, :Num_translation_channels] /= Weight_translation
        
        read_bvh.write_frames(Standard_bvh_file, 
                              os.path.join(save_dance_folder, f"{iteration:07d}_gt.bvh"), 
                              gt_euler_deg_np)

        # --- Predicted BVH Visualization ---
        pred_seq_for_viz_np = predict_seq_flat[0].data.cpu().numpy().reshape(seq_len, in_frame_size)
        pred_bvh_abs_frames = np.zeros_like(pred_seq_for_viz_np)
        if Num_translation_channels >= 2: pred_bvh_abs_frames[:,1] = pred_seq_for_viz_np[:,1]
        if in_frame_size > Num_translation_channels:
             pred_bvh_abs_frames[:, Num_translation_channels:] = pred_seq_for_viz_np[:, Num_translation_channels:]
        
        # --- CHANGE 2 (Continued): Correct Starting Absolute Position for Prediction Visualization ---
        # Use the same starting absolute position as the GT for a fair comparison of this segment.
        current_x_pred = real_seq_np[0, 1, 0] if Num_translation_channels >=1 else 0.0
        current_z_pred = real_seq_np[0, 1, 2] if Num_translation_channels >=3 else 0.0

        for frame_idx in range(seq_len):
            if Num_translation_channels >=1:
                current_x_pred += pred_seq_for_viz_np[frame_idx, 0] # Add predicted dX
                pred_bvh_abs_frames[frame_idx, 0] = current_x_pred
            if Num_translation_channels >=3:
                current_z_pred += pred_seq_for_viz_np[frame_idx, 2] # Add predicted dZ
                pred_bvh_abs_frames[frame_idx, 2] = current_z_pred

        pred_euler_deg_np = np.zeros((seq_len, num_euler_ch_out))
        if Num_translation_channels > 0:
            pred_euler_deg_np[:, :Num_translation_channels] = pred_bvh_abs_frames[:, :Num_translation_channels]

        # After computing pred_bvh_abs_frames
        for r_j, joint_name in enumerate(rotating_joint_names_in_order):
            rot_order = joint_rotation_orders.get(joint_name, 'rzyx')
            q_start = quat_ch_offset + r_j * 4
            e_start = euler_ch_offset + r_j * 3
            quats_joint = pred_bvh_abs_frames[:, q_start : q_start + 4]
            for frame_k in range(seq_len):
                q_norm = quats_joint[frame_k] / (np.linalg.norm(quats_joint[frame_k]) + Epsilon)
                try:
                    axes_param = 's' + rot_order[1:] if rot_order.startswith('r') else rot_order
                    angles_rad = t3d_euler.quat2euler(q_norm, axes=axes_param)
                    pred_euler_deg_np[frame_k, e_start:e_start+3] = np.rad2deg(angles_rad)
                except Exception:
                    pred_euler_deg_np[frame_k, e_start:e_start+3] = 0.0

        if Weight_translation != 1.0 and Num_translation_channels > 0:
            pred_euler_deg_np[:, :Num_translation_channels] /= Weight_translation

        read_bvh.write_frames(Standard_bvh_file, 
                              os.path.join(save_dance_folder, f"{iteration:07d}_out.bvh"), 
                              pred_euler_deg_np)


#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        #length=len(dance)/100
        length = 10
        if(length<1):
            length=1              
        len_lst=len_lst+[length]
    
    index_lst=[]
    index=0
    for length in len_lst:
        for i in range(length):
            index_lst=index_lst+[index]
        index=index+1
    return index_lst

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files = [f for f in os.listdir(dance_folder) if f.endswith('.npy')]    
    dances=[]
    print('Loading motion files...')
    for dance_file in dance_files:
        # print ("load "+dance_file)
        dance=np.load(dance_folder+dance_file, allow_pickle=True)
        dances=dances+[dance]
    print(len(dances), ' Motion files loaded')

    return dances

def train(dances, frame_rate, args, joint_rotation_orders, rotating_joint_names_in_order):

    seq_len_for_data_loader = args.seq_len + 2

    model = acLSTM(in_frame_size=args.in_frame, hidden_size=args.hidden_size, out_frame_size=args.out_frame)
    
    if args.read_weight_path and os.path.exists(args.read_weight_path): # Check path exists
        try:
            model.load_state_dict(torch.load(args.read_weight_path, map_location=device, weights_only=True))
            print(f"Loaded weights from: {args.read_weight_path}")
        except Exception as e:
            print(f"Could not load weights from {args.read_weight_path}: {e}. Training from scratch.")
    
    model.to(device)

    current_lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    model.train()
    
    dance_len_idx_list = get_dance_len_lst(dances) # Renamed for clarity
    if not dance_len_idx_list:
        print("No dances available for training or get_dance_len_lst failed.")
        return
    random_range_sampling = len(dance_len_idx_list)
    
    # Speed adjustment factor for sampling frames
    speed_adj_factor = frame_rate / 30.0 if frame_rate > 30.0 else 1.0
    
    for iteration in range(args.total_iterations):   
        dance_batch_list = [] # Renamed for clarity
        for _ in range(args.batch_size): # Use args.batch_size
            dance_id_idx = random.choice(dance_len_idx_list) # Simpler choice
            selected_dance_np = dances[dance_id_idx] # .copy() not strictly needed if not modified in place before processing
            
            # Effective number of frames to sample from source dance
            effective_sample_len = int(seq_len_for_data_loader * speed_adj_factor)

            if selected_dance_np.shape[0] < effective_sample_len + 20: # Need margin
                # Simple skip, or re-sample. For simplicity, can lead to smaller batches if many are short.
                # print(f"Skipping short dance {dance_id_idx}")
                continue 
            
            max_start_id = selected_dance_np.shape[0] - effective_sample_len - 10 # Margin from end
            if max_start_id < 10: # Margin from start
                # print(f"Skipping dance {dance_id_idx} due to insufficient length after margins.")
                continue

            start_id_in_source = random.randint(10, max_start_id)
            
            sampled_one_sequence = []
            for i in range(seq_len_for_data_loader):
                frame_to_sample_idx = min(int(i * speed_adj_factor + start_id_in_source), selected_dance_np.shape[0] - 1)
                sampled_one_sequence.append(selected_dance_np[frame_to_sample_idx])
            
            # Augmentation was specific to POS data (rotating 3D positions)
            # For QUAT data, augmentation is more complex 

            if sampled_one_sequence: # Ensure it's not empty if a dance was skipped
                 dance_batch_list.append(np.array(sampled_one_sequence))

        if not dance_batch_list: # If all dances were too short in this iteration
            # print("Warning: Could not form a batch this iteration.")
            continue
            
        dance_batch_np = np.array(dance_batch_list)
       
        print_loss=False
        save_bvh_motion=False
        if(iteration % 20==0):
            print_loss=True
        if(iteration % 1000==0):
            save_bvh_motion=True
            path = args.write_weight_folder + "%07d"%iteration +".weight"
            torch.save(model.state_dict(), path)

        train_one_iteraton(dance_batch_np, model, optimizer, iteration, 
                           args.write_bvh_motion_folder, args.seq_len, args.in_frame,joint_rotation_orders, rotating_joint_names_in_order, print_loss, save_bvh_motion) # Global


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dances_folder', type=str, required=True, help='Path for the QUATERNION .npy training data')
    parser.add_argument('--write_weight_folder', type=str, required=True, help='Path to store checkpoints')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to store test generated bvh')
    parser.add_argument('--read_weight_path', type=str, default="./output/weights/salsa/0005000.weight", help='Checkpoint model path')
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Source dance frame rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--in_frame', type=int, default=175, help='Input channels')
    parser.add_argument('--out_frame', type=int, default=175, help='Output channels')
    parser.add_argument('--hidden_size', type=int, default=1024, help='LSTM hidden size')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for model processing')
    parser.add_argument('--total_iterations', type=int, default=100000, help='Total training iterations')


    args = parser.parse_args()

    # --- VALIDATE IN_FRAME and OUT_FRAME ---
    if args.in_frame != args.out_frame:
        print("Error: --in_frame and --out_frame must be equal for this model structure.")
        exit(1)
    if args.in_frame <= Num_translation_channels or (args.in_frame - Num_translation_channels) % 4 != 0:
        print(f"Error: --in_frame ({args.in_frame}) is not valid for {Num_translation_channels} translation channels and quaternion data (must be > num_trans and remainder multiple of 4).")
        print(f"Ensure it matches 'calculated_quat_frame_size' from your data generation script.")
        exit(1)
    
    print(f"Using QUATERNION In/Out Frame Size: {args.in_frame}")
    print(f"  (Implies {Num_translation_channels} translation channels and {(args.in_frame - Num_translation_channels)//4} rotating joints)")

    if not os.path.exists(args.write_weight_folder): 
        os.makedirs(args.write_weight_folder)
    if not os.path.exists(args.write_bvh_motion_folder): 
        os.makedirs(args.write_bvh_motion_folder)
    if not os.path.exists(Standard_bvh_file):
        print(f"Error: --standard_bvh_for_writing path not found: {Standard_bvh_file}")
        exit(1)

    # we must use the same per-joint rotation orders as generate_training_quat_data.py
    metadata_path = os.path.join(args.dances_folder, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        joint_rotation_orders = metadata['joint_rotation_orders']
        rotating_joint_names_in_order = metadata['rotating_joint_names_in_order']
    else:
        print(f"Error: metadata.json not found in {args.dances_folder}")
        exit(1)

    # Load QUATERNION .npy data
    dances_data = load_dances(args.dances_folder)

    train(dances_data, args.dance_frame_rate, args, joint_rotation_orders, rotating_joint_names_in_order) # Passing full args

if __name__ == '__main__':
    main()