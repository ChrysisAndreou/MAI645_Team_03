import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
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


# Constants
standard_bvh_file = "train_data_bvh/standard.bvh"
Hip_index = read_bvh.joint_index['hip']

Seq_len = 100
Hidden_size = 1024
Joints_num = 57
Condition_num = 5
Groundtruth_num = 5
In_frame_size = 175

class acLSTM(nn.Module):
    def __init__(self, in_frame_size=175, hidden_size=1024, out_frame_size=175):
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size
        
        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)#param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    
    #output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        #c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
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
        
    
    #in cuda tensor initial_seq: b*(initial_seq_len*frame_size)
    #out cuda tensor out_seq  b* ( (intial_seq_len + generate_frame_number) *frame_size)
    def forward(self, initial_seq, generate_frames_number):
        
        batch=initial_seq.size()[0]
        
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).to(device))

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).to(device))
        
        
        for i in range(initial_seq.size()[1]):
            in_frame=initial_seq[:,i]
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
        
        for i in range(generate_frames_number):
            
            in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
    
        return out_seq[:, 1: out_seq.size()[1]]

def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder, 
                 in_frame_size, num_translation_channels, weight_translation, 
                 joint_rotation_orders, rotating_joint_names_in_order):
    # Process input: differences for X and Z translations
    dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0:initial_seq_np.shape[1]-1]
    initial_seq_dif_np = initial_seq_np[:, 0:initial_seq_np.shape[1]-1].copy()
    if num_translation_channels >= 1:
        initial_seq_dif_np[:, :, 0] = dif[:, :, 0]  # dX
    if num_translation_channels >= 3:
        initial_seq_dif_np[:, :, 2] = dif[:, :, 2]  # dZ
    
    initial_seq = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_np.tolist()).to(device))
    predict_seq = model.forward(initial_seq, generate_frames_number)
    
    batch = initial_seq_np.shape[0]
    predict_seq_np = np.array(predict_seq.data.tolist()).reshape(batch, -1, in_frame_size)
    
    for b in range(batch):
        out_seq = predict_seq_np[b].copy()
        # Reconstruct absolute translations
        if num_translation_channels >= 1:
            last_x = 0.0
            for frame in range(out_seq.shape[0]):
                out_seq[frame, 0] = out_seq[frame, 0] + last_x
                last_x = out_seq[frame, 0]
        if num_translation_channels >= 3:
            last_z = 0.0
            for frame in range(out_seq.shape[0]):
                out_seq[frame, 2] = out_seq[frame, 2] + last_z
                last_z = out_seq[frame, 2]
        
        # Convert quaternions to Euler angles
        num_rot_j = (in_frame_size - num_translation_channels) // 4
        euler_frame_size = num_translation_channels + num_rot_j * 3
        euler_seq = np.zeros((out_seq.shape[0], euler_frame_size))
        if num_translation_channels > 0:
            euler_seq[:, :num_translation_channels] = out_seq[:, :num_translation_channels]
        for r_j, joint_name in enumerate(rotating_joint_names_in_order):
            rot_order = joint_rotation_orders.get(joint_name, 'rzyx')
            q_start = num_translation_channels + r_j * 4
            e_start = num_translation_channels + r_j * 3
            quats_joint = out_seq[:, q_start:q_start + 4]
            for frame_k in range(out_seq.shape[0]):
                q_norm = quats_joint[frame_k] / (np.linalg.norm(quats_joint[frame_k]) + 1e-7)
                axes_param = 's' + rot_order[1:] if rot_order.startswith('r') else rot_order
                angles_rad = t3d_euler.quat2euler(q_norm, axes=axes_param)
                euler_seq[frame_k, e_start:e_start+3] = np.rad2deg(angles_rad)
        
        # Unscale translations
        if weight_translation != 1.0 and num_translation_channels > 0:
            euler_seq[:, :num_translation_channels] /= weight_translation
        
        # Write to BVH
        read_bvh.write_frames(standard_bvh_file, os.path.join(save_dance_folder, "out%02d.bvh" % b), euler_seq)
    return predict_seq_np

def get_dance_len_lst(dances):
    len_lst = []
    for dance in dances:
        length = len(dance) / 100
        length = 10
        if length < 1:
            length = 1              
        len_lst = len_lst + [length]
    
    index_lst = []
    index = 0
    for length in len_lst:
        for i in range(length):
            index_lst = index_lst + [index]
        index = index + 1
    return index_lst

def load_dances(dance_folder):
    dance_files = [f for f in os.listdir(dance_folder) if f.endswith('.npy')]
    dances = []
    for dance_file in dance_files:
        print("load " + dance_file)
        dance = np.load(os.path.join(dance_folder, dance_file))
        print("frame number: " + str(dance.shape[0]))
        dances = dances + [dance]
    return dances
    
def test(dances, frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path,
         write_bvh_motion_folder, in_frame_size=171, hidden_size=1024, out_frame_size=171,
         num_translation_channels=3, weight_translation=1.0, joint_rotation_orders=None, 
         rotating_joint_names_in_order=None):
    # torch.cuda.set_device(0)
    model = acLSTM(in_frame_size, hidden_size, out_frame_size)
    model.load_state_dict(torch.load(read_weight_path, weights_only=True, map_location=device))
    model.to(device)

    dance_len_lst = get_dance_len_lst(dances)
    random_range = len(dance_len_lst)
    speed = frame_rate / 30
    
    dance_batch = []
    for b in range(batch):
        dance_id = dance_len_lst[np.random.randint(0, random_range)]
        dance = dances[dance_id].copy()
        dance_len = dance.shape[0]
        start_id = random.randint(10, int(dance_len - initial_seq_len * speed - 10))
        sample_seq = [dance[int(i * speed + start_id)] for i in range(initial_seq_len)]
        dance_batch = dance_batch + [sample_seq]
            
    dance_batch_np = np.array(dance_batch)
    generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder,
                 in_frame_size, num_translation_channels, weight_translation, 
                 joint_rotation_orders, rotating_joint_names_in_order)

# Main execution
read_weight_path = "output_rigid/weights/salsa/0038000.weight"
write_bvh_motion_folder = "synthesized_output/salsa"
dances_folder = "train_data_quat/salsa"
dance_frame_rate = 60
batch = 32
initial_seq_len = 15
generate_frames_number = 400

if not os.path.exists(write_bvh_motion_folder):
    os.makedirs(write_bvh_motion_folder)

# Load metadata
metadata_path = "train_data_quat/salsa/metadata.json"
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    joint_rotation_orders = metadata['joint_rotation_orders']
    rotating_joint_names_in_order = metadata['rotating_joint_names_in_order']
    num_translation_channels = metadata['num_translation_channels']
    in_frame_size = metadata['calculated_quat_frame_size']
    out_frame_size = in_frame_size
    weight_translation = metadata.get('weight_translation_factor', 0.01)
else:
    print(f"Error: metadata.json not found in {dances_folder}")
    exit(1)

dances = load_dances(dances_folder)
hidden_size = 1024

test(dances, dance_frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path,
     write_bvh_motion_folder, in_frame_size, hidden_size, out_frame_size,
     num_translation_channels, weight_translation, joint_rotation_orders, rotating_joint_names_in_order)