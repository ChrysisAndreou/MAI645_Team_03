import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse


# Determine the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    # Keep CUDA as an option if run elsewhere
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")
# --- END Device Selection Logic ---

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
Condition_num=5
Groundtruth_num=5
# The input frame size is 132 for euler data
In_frame_size = 132

weight_translation = 0.01


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=132, hidden_size=1024, out_frame_size=132):
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
        
    
    #in cuda tensor real_seq: b*seq_len*frame_size
    #out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        
        batch=real_seq.size()[0]
        seq_len=real_seq.size()[1]
        
        condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).to(device))

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).to(device))
        
        
        for i in range(seq_len):
            
            if(condition_lst[i]==1):##input groundtruth frame
                in_frame=real_seq[:,i]
            else:
                in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
    
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size)
     
    # Modified loss claculation for euler angles
    # Calculates a combined loss: MSE for translations, (1 - cos(error)) for angles.
    def calculate_loss(self, out_seq_flat, groundtruth_seq_flat, translation_weight=0.1, angle_weight=1.0):
        batch_size = out_seq_flat.size(0)
        # Reshape to [batch_size, seq_len, frame_size]
        # The frame_size is self.out_frame_size (e.g., 132)
        out_seq = out_seq_flat.view(batch_size, -1, self.out_frame_size)
        groundtruth_seq = groundtruth_seq_flat.view(batch_size, -1, self.out_frame_size)

        # Translations are the first 3 channels
        translations_out = out_seq[:, :, :3]
        translations_gt = groundtruth_seq[:, :, :3]
        mse_loss_fn = nn.MSELoss()
        loss_translation = mse_loss_fn(translations_out, translations_gt)

        loss_angle = 0
        angles_out = out_seq[:, :, 3:]       
        angles_gt = groundtruth_seq[:, :, 3:]

        # Cosine similarity loss: 1 - cos(angle_true - angle_pred)
        angle_error = angles_out - angles_gt # Difference in radians
        cos_error = torch.cos(angle_error)
        # Loss for each angle component is (1 - cos_error)
        # We want to sum this over all angle components and average
        loss_angle_components = 1.0 - cos_error
        loss_angle = torch.mean(loss_angle_components) # Mean over all angle components, all frames, all batches

        # Combine losses
        total_loss = (translation_weight * loss_translation) + (angle_weight * loss_angle)
        return total_loss


#numpy array real_seq_np: batch*seq_len*frame_size
def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False, save_bvh_motion=True):

    # Creating the processed NumPy array with hip XZ differences.
    dif_all_channels = real_seq_np[:, 1:real_seq_np.shape[1]] - real_seq_np[:, 0:real_seq_np.shape[1]-1]
    real_seq_dif_hip_x_z_np = real_seq_np[:, 0:real_seq_np.shape[1]-1].copy()

    # Replacing hip X (channel 0) and Z (channel 2) absolute values with their calculated differences.
    real_seq_dif_hip_x_z_np[:, :, 0] = dif_all_channels[:, :, 0]  # Hip X position difference
    real_seq_dif_hip_x_z_np[:, :, 2] = dif_all_channels[:, :, 2]  # Hip Z position difference

    # Converting the processed NumPy array to a PyTorch tensor.
    real_seq = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np)).to(device)

    # Defining LSTM processing sequence length and slice input/target.
    seq_len = real_seq.size(1) - 1

    in_real_seq = real_seq[:, 0:seq_len]

    predict_groundtruth_seq = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np[:, 1:seq_len+1])).to(device).view(real_seq_np.shape[0], -1)

    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)

    optimizer.zero_grad()

    # Using spefified loss function for angular loss calculation.
    loss = model.calculate_loss(predict_seq, predict_groundtruth_seq, translation_weight=1.0)

    loss.backward()
    optimizer.step()

    if (print_loss==True):
        print ("###########"+"iter %07d"%iteration +"######################")
        print ("loss: "+str(loss.detach().cpu().numpy()))

    if (save_bvh_motion==True):

        gt_seq = np.array(predict_groundtruth_seq[0].data.cpu().numpy()).reshape(
            seq_len, In_frame_size 
        )
        
        last_x_gt = 0.0
        last_z_gt = 0.0
        
        for frame_idx in range(gt_seq.shape[0]):
            # The value in gt_seq[frame_idx, 0] is dx for that frame
            # The value in gt_seq[frame_idx, 2] is dz for that frame
            current_dx_gt = gt_seq[frame_idx, 0]
            current_dz_gt = gt_seq[frame_idx, 2]
            
            last_x_gt += current_dx_gt
            last_z_gt += current_dz_gt
            
            gt_seq[frame_idx, 0] = last_x_gt # Store absolute X
            gt_seq[frame_idx, 2] = last_z_gt # Store absolute Z
        # Now `gt_seq` has absolute (accumulated) X and Z hip positions.

        # Convert rotations (channels 3:) from RADIANS to DEGREES for BVH
        gt_seq_degrees = gt_seq.copy() # Make a copy before degree conversion
        gt_seq_degrees[:, 3:] = np.rad2deg(gt_seq_degrees[:, 3:])

        # Unscale X (ch 0), Y (ch 1), Z (ch 2) hip translations using global weight_translation
        if weight_translation != 0 and weight_translation != 1.0:
            gt_seq_degrees[:, 0:3] /= weight_translation # gt_seq is modified in place


        # --- Model Output Reconstruction & Saving ---
        out_seq = np.array(predict_seq[0].data.cpu().numpy()).reshape(
            seq_len, In_frame_size
        )
        # `out_seq` also has hip X,Z as velocities from the model's prediction.

        # Inlined reconstruction of absolute hip X, Z positions for Model Output
        last_x_out = 0.0
        last_z_out = 0.0
        # Modifying `out_seq`
        for frame_idx in range(out_seq.shape[0]):
            current_dx_out = out_seq[frame_idx, 0]
            current_dz_out = out_seq[frame_idx, 2]

            last_x_out += current_dx_out
            last_z_out += current_dz_out

            out_seq[frame_idx, 0] = last_x_out # Store absolute X
            out_seq[frame_idx, 2] = last_z_out # Store absolute Z
        # Now `out_seq` has absolute (accumulated) X and Z hip positions.

        # Convert rotations from RADIANS to DEGREES for BVH
        out_seq_degrees = out_seq.copy()
        if out_seq_degrees.shape[1] > 3:
            out_seq_degrees[:, 3:] = np.rad2deg(out_seq_degrees[:, 3:])

        # Unscale X (ch 0), Y (ch 1), Z (ch 2) hip translations
        if weight_translation != 0 and weight_translation != 1.0:
            out_seq_degrees[:, 0:3] /= weight_translation


        # --- Write to BVH files ---
        # `gt_seq_degrees` and `out_seq_degrees` now contain the final multi-channel frame data for BVH.
        standard_bvh_template_path = read_bvh.standard_bvh_file

        gt_bvh_filename = os.path.join(save_dance_folder, f"{iteration:07d}_gt.bvh")
        out_bvh_filename = os.path.join(save_dance_folder, f"{iteration:07d}_out.bvh")

        read_bvh.write_frames(standard_bvh_template_path, gt_bvh_filename, gt_seq_degrees)
        read_bvh.write_frames(standard_bvh_template_path, out_bvh_filename, out_seq_degrees)

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
    dance_files=os.listdir(dance_folder)
    dances=[]
    print('Loading motion files...')
    for dance_file in dance_files:
        # print ("load "+dance_file)
        dance=np.load(dance_folder+dance_file)
        dances=dances+[dance]
    print(len(dances), ' Motion files loaded')

    return dances
    
# dances: [dance1, dance2, dance3,....]
def train(dances, frame_rate, batch, seq_len, read_weight_path, write_weight_folder,
          write_bvh_motion_folder, in_frame, out_frame, hidden_size=1024, total_iter=500000):
    
    seq_len=seq_len+2

    model = acLSTM(in_frame_size=in_frame, hidden_size=hidden_size, out_frame_size=out_frame)
    
    if(read_weight_path!=""):
        model.load_state_dict(torch.load(read_weight_path, map_location=device))
    
    model.to(device)
    #model=torch.nn.DataParallel(model, device_ids=[0,1])

    current_lr=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    
    model.train()
    
    #dance_len_lst contains the index of the dance, the occurance number of a dance's index is proportional to the length of the dance
    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)
    
    speed=frame_rate/30 # we train the network with frame rate of 30
    
    for iteration in range(total_iter):   
        #get a batch of dances
        dance_batch=[]
        for b in range(batch):
            # randomly pick up one dance. the longer the dance is the more likely the dance is picked up
            dance_id = dance_len_lst[np.random.randint(0,random_range)]
            dance=dances[dance_id].copy()
            dance_len = dance.shape[0]
            
            start_id=random.randint(10, dance_len-seq_len*speed-10)#the first and last several frames are sometimes noisy. 
            sample_seq=[]
            for i in range(seq_len):
                sample_seq=sample_seq+[dance[int(i*speed+start_id)]]
            
            # Removed augmentatio (as it's for positional)
            # augment the direction and position of the dance, helps the model to not overfeed
            # T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
            # R=[0,1,0,(random.random()-0.5)*np.pi*2]
            # sample_seq_augmented=read_bvh.augment_train_data(sample_seq, T, R)
            # dance_batch=dance_batch+[sample_seq_augmented]
            dance_batch=dance_batch+[np.array(sample_seq)]
        dance_batch_np=np.array(dance_batch)
       
        
        print_loss=False
        save_bvh_motion=False
        if(iteration % 20==0):
            print_loss=True
        if(iteration % 1000==0):
            save_bvh_motion=True
            path = write_weight_folder + "%07d"%iteration +".weight"
            torch.save(model.state_dict(), path)
            
        train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss, save_bvh_motion)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dances_folder', type=str, required=True, help='Path for the training data')
    parser.add_argument('--write_weight_folder', type=str, required=True, help='Path to store checkpoints')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to store test generated bvh')
    parser.add_argument('--read_weight_path', type=str, default="", help='Checkpoint model path')
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Dance frame rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--in_frame', type=int, default=132, help='Input channel')
    parser.add_argument('--out_frame', type=int, default=132, help='Output channels')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Checkpoint model path')
    parser.add_argument('--seq_len', type=int, default=100, help='Checkpoint model path')
    parser.add_argument('--total_iterations', type=int, default=100000, help='Checkpoint model path')

    args = parser.parse_args()


    if not os.path.exists(args.write_weight_folder):
        os.makedirs(args.write_weight_folder)
    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    dances= load_dances(args.dances_folder)

    train(dances, args.dance_frame_rate, args.batch_size, args.seq_len, args.read_weight_path, args.write_weight_folder,
          args.write_bvh_motion_folder, args.in_frame, args.out_frame, args.hidden_size, total_iter=args.total_iterations)

if __name__ == '__main__':
    main()