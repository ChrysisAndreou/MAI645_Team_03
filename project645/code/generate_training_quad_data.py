import read_bvh
import numpy as np
import os
import torch
import json
# Ensure rotation_conversions can be imported. 
# If generate_training_quad_data.py and rotation_conversions.py are in the same directory (code/),
# a direct import should work when the script is run from that directory or if the path is set up.
# For now, assuming direct import works, or python path is configured.
import rotation_conversions 

HIP_POS_SCALE_FACTOR = 0.01

def get_bvh_frame_time(bvh_filepath):
    """Parses the frame time from a BVH file."""
    try:
        with open(bvh_filepath, 'r') as f:
            for line in f:
                if line.strip().startswith("Frame Time:"):
                    return float(line.split(":")[1].strip())
    except Exception as e:
        print(f"Warning: Could not parse frame time from {bvh_filepath}: {e}")
    return None

def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder, skeleton_info, hip_pos_scale_factor):
    """
    Converts BVH files to quaternion-based training data.
    Hip positions are normalized (scaled, XZ relative to first frame).
    Joint rotations are converted from Euler angles (degrees in BVH) to quaternions.
    """
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
        print(f"Created directory: {tar_traindata_folder}")

    skeleton, non_end_bones = skeleton_info
    bvh_files = [f for f in os.listdir(src_bvh_folder) if f.endswith(".bvh")]
    print(f"Found {len(bvh_files)} BVH files in {src_bvh_folder}")

    if not bvh_files:
        print(f"No BVH files found in {src_bvh_folder}. Skipping.")
        return

    first_bvh_path_for_metadata = os.path.join(src_bvh_folder, bvh_files[0])
    
    num_rotating_joints = 1 + len(non_end_bones) 
    num_output_channels = 3 + num_rotating_joints * 4
    
    original_frame_time = get_bvh_frame_time(first_bvh_path_for_metadata)
    if original_frame_time is None:
        print(f"Warning: Could not determine frame time from {bvh_files[0]}. Defaulting to 0.016666 (60 FPS) for metadata.")
        original_frame_time = 0.016666 

    for bvh_file_name in bvh_files:
        src_path = os.path.join(src_bvh_folder, bvh_file_name)
        print(f"Processing (encode): {src_path}")

        raw_frames_np = read_bvh.parse_frames(src_path)
        if raw_frames_np is None or raw_frames_np.shape[0] == 0:
            print(f"Warning: No motion data found in {bvh_file_name}. Skipping.")
            continue

        num_frames = raw_frames_np.shape[0]
        processed_frames_list = []
        
        first_frame_hip_x_abs = raw_frames_np[0, 0]
        first_frame_hip_z_abs = raw_frames_np[0, 2]

        for frame_idx in range(num_frames):
            raw_frame = raw_frames_np[frame_idx]
            current_feature_vector = []
            channel_cursor = 0

            hip_pos_abs = raw_frame[channel_cursor : channel_cursor+3]
            norm_hip_x = (hip_pos_abs[0] - first_frame_hip_x_abs) * hip_pos_scale_factor
            norm_hip_y = hip_pos_abs[1] * hip_pos_scale_factor
            norm_hip_z = (hip_pos_abs[2] - first_frame_hip_z_abs) * hip_pos_scale_factor
            current_feature_vector.extend([norm_hip_x, norm_hip_y, norm_hip_z])
            channel_cursor += 3

            hip_rot_Z_deg = raw_frame[channel_cursor]
            hip_rot_Y_deg = raw_frame[channel_cursor+1]
            hip_rot_X_deg = raw_frame[channel_cursor+2]
            hip_euler_rad = torch.tensor(np.deg2rad([hip_rot_Z_deg, hip_rot_Y_deg, hip_rot_X_deg]), dtype=torch.float32)
            hip_rot_matrix = rotation_conversions.euler_angles_to_matrix(hip_euler_rad.unsqueeze(0), "ZYX")
            hip_quat = rotation_conversions.matrix_to_quaternion(hip_rot_matrix).squeeze(0)
            current_feature_vector.extend(hip_quat.tolist())
            channel_cursor += 3

            for _ in non_end_bones:
                joint_rot_Z_deg = raw_frame[channel_cursor]
                joint_rot_X_deg = raw_frame[channel_cursor+1]
                joint_rot_Y_deg = raw_frame[channel_cursor+2]
                joint_euler_rad = torch.tensor(np.deg2rad([joint_rot_Z_deg, joint_rot_X_deg, joint_rot_Y_deg]), dtype=torch.float32)
                joint_rot_matrix = rotation_conversions.euler_angles_to_matrix(joint_euler_rad.unsqueeze(0), "ZXY")
                joint_quat = rotation_conversions.matrix_to_quaternion(joint_rot_matrix).squeeze(0)
                current_feature_vector.extend(joint_quat.tolist())
                channel_cursor += 3
            
            processed_frames_list.append(current_feature_vector)

        processed_data_np = np.array(processed_frames_list, dtype=np.float32)
        
        if processed_data_np.shape[1] != num_output_channels:
             print(f"CRITICAL ERROR in {bvh_file_name}: Expected {num_output_channels} channels but got {processed_data_np.shape[1]}. Skipping file.")
             continue

        target_filename = os.path.splitext(bvh_file_name)[0] + ".npy"
        target_path = os.path.join(tar_traindata_folder, target_filename)
        np.save(target_path, processed_data_np)
        print(f"Saved quaternion data to {target_path} with shape {processed_data_np.shape}")

    metadata = {
        "num_channels": num_output_channels,
        "hip_pos_scale_factor": hip_pos_scale_factor,
        "frame_time_original": original_frame_time,
        "source_bvh_folder": os.path.abspath(src_bvh_folder),
        "quaternion_order": "wxyz"
    }
    metadata_path = os.path.join(tar_traindata_folder, "metadata_quad.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata {metadata_path}: {e}")


def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder, skeleton_info, hip_pos_scale_factor, standard_bvh_ref_path):
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
        print(f"Created directory: {tar_bvh_folder}")

    skeleton, non_end_bones = skeleton_info
    npy_files = [f for f in os.listdir(src_train_folder) if f.endswith(".npy")]
    print(f"Found {len(npy_files)} NPY files in {src_train_folder}")

    if not npy_files:
        print(f"No .npy files found in {src_train_folder}. Skipping reconstruction.")
        return

    metadata_path = os.path.join(src_train_folder, "metadata_quad.json")
    bvh_output_frame_time = 0.016666 
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if "frame_time_original" in metadata:
                bvh_output_frame_time = metadata["frame_time_original"]
                print(f"Using frame_time_original from metadata: {bvh_output_frame_time}")
    except Exception as e:
        print(f"Warning: Could not load or parse metadata from {metadata_path}. Using default frame time {bvh_output_frame_time}. Error: {e}")

    for npy_file_name in npy_files:
        src_path = os.path.join(src_train_folder, npy_file_name)
        print(f"Processing (decode): {src_path}")

        quad_data_np = np.load(src_path)
        if quad_data_np is None or quad_data_np.shape[0] == 0:
            print(f"Warning: No motion data found in {npy_file_name}. Skipping.")
            continue
        
        num_frames = quad_data_np.shape[0]
        reconstructed_bvh_frames_list = []

        for frame_idx in range(num_frames):
            processed_frame = quad_data_np[frame_idx]
            current_bvh_channels = []
            feature_cursor = 0

            scaled_hip_pos = processed_frame[feature_cursor : feature_cursor+3]
            unscaled_hip_x = scaled_hip_pos[0] / hip_pos_scale_factor
            unscaled_hip_y = scaled_hip_pos[1] / hip_pos_scale_factor
            unscaled_hip_z = scaled_hip_pos[2] / hip_pos_scale_factor
            current_bvh_channels.extend([unscaled_hip_x, unscaled_hip_y, unscaled_hip_z])
            feature_cursor += 3

            hip_quat_wxyz = torch.tensor(processed_frame[feature_cursor : feature_cursor+4], dtype=torch.float32)
            hip_rot_matrix = rotation_conversions.quaternion_to_matrix(hip_quat_wxyz.unsqueeze(0))
            hip_euler_rad_zyx = rotation_conversions.matrix_to_euler_angles(hip_rot_matrix, "ZYX").squeeze(0)
            hip_euler_deg_zyx = np.rad2deg(hip_euler_rad_zyx.tolist())
            current_bvh_channels.extend(hip_euler_deg_zyx)
            feature_cursor += 4
            
            for _ in non_end_bones:
                joint_quat_wxyz = torch.tensor(processed_frame[feature_cursor : feature_cursor+4], dtype=torch.float32)
                joint_rot_matrix = rotation_conversions.quaternion_to_matrix(joint_quat_wxyz.unsqueeze(0))
                joint_euler_rad_zxy = rotation_conversions.matrix_to_euler_angles(joint_rot_matrix, "ZXY").squeeze(0)
                joint_euler_deg_zxy = np.rad2deg(joint_euler_rad_zxy.tolist())
                current_bvh_channels.extend(joint_euler_deg_zxy)
                feature_cursor += 4
            
            reconstructed_bvh_frames_list.append(current_bvh_channels)

        reconstructed_bvh_data_np = np.array(reconstructed_bvh_frames_list, dtype=np.float32)
        
        target_bvh_filename = os.path.splitext(npy_file_name)[0] + ".bvh"
        target_path = os.path.join(tar_bvh_folder, target_bvh_filename)
        
        read_bvh.write_frames(standard_bvh_ref_path, target_path, reconstructed_bvh_data_np, frame_time_override=bvh_output_frame_time)
        print(f"Saved reconstructed BVH file to {target_path}")


if __name__ == "__main__":
    standard_bvh_file_path = "../train_data_bvh/standard.bvh"
    if not os.path.exists(standard_bvh_file_path):
        alt_standard_bvh_path = "train_data_bvh/standard.bvh"
        if os.path.exists(alt_standard_bvh_path):
            standard_bvh_file_path = alt_standard_bvh_path
        else:
            print(f"ERROR: Standard BVH file not found at {standard_bvh_file_path} or {alt_standard_bvh_path}.")
            exit()
    print(f"Using standard BVH for hierarchy: {standard_bvh_file_path}")
    
    try:
        # This line needs read_bvh_hierarchy to be importable.
        # If read_bvh.py already imports and uses read_bvh_hierarchy globally, this might be redundant
        # or could conflict if paths differ. Assuming direct import for clarity here.
        import read_bvh_hierarchy # Ensure it's available
        current_skeleton, current_non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file_path)
        skeleton_info_tuple = (current_skeleton, current_non_end_bones)
        print(f"Loaded hierarchy: {len(current_skeleton)} total joints, {len(current_non_end_bones)} non-end-site rotating joints (excluding hip).")
    except Exception as e:
        print(f"Error loading BVH hierarchy from {standard_bvh_file_path}: {e}")
        exit()

    base_bvh_dir = "../train_data_bvh/" 
    data_types = ["salsa", "indian", "martial"]

    for data_type in data_types:
        print(f"\n--- Processing data type: {data_type} ---")
        src_bvh_folder_path = os.path.join(base_bvh_dir, data_type)
        
        if not os.path.isdir(src_bvh_folder_path):
            print(f"Source BVH folder for {data_type} not found: {src_bvh_folder_path}. Skipping.")
            continue

        quad_traindata_folder_path = f"../train_data_quad/{data_type}/"
        reconstructed_bvh_folder_path = f"../reconstructed_bvh_data_quad/{data_type}/"
        
        print(f"--- Quaternion Data Generation for {data_type} ---")
        print(f"Source BVH: {src_bvh_folder_path}")
        print(f"Target .npy: {quad_traindata_folder_path}")
        print(f"Target reconstructed BVH: {reconstructed_bvh_folder_path}")
        print(f"Hip Position Scale Factor: {HIP_POS_SCALE_FACTOR}")

        print("\nStep 1: Converting BVH to Normalized Quaternion (.npy)...")
        generate_quad_traindata_from_bvh(src_bvh_folder_path, quad_traindata_folder_path, skeleton_info_tuple, HIP_POS_SCALE_FACTOR)
        
        print("\nStep 2: Converting .npy back to BVH for verification...")
        generate_bvh_from_quad_traindata(quad_traindata_folder_path, reconstructed_bvh_folder_path, skeleton_info_tuple, HIP_POS_SCALE_FACTOR, standard_bvh_file_path)
        
        print(f"\nQuaternion data processing for {data_type} completed.")

    print("\n--- All data types processed. ---")