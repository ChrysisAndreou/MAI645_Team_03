import read_bvh
import numpy as np
from os import listdir
import os

def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    """
    Reads BVH files, extracts raw Euler angle and translation motion data,
    scales translation, converts angles to radians, and saves as .npy files.
    """
    if not src_bvh_folder or not tar_traindata_folder:
        print("Error: Source BVH folder or target train data folder not specified.")
        return
    if not os.path.exists(src_bvh_folder):
        print(f"Error: Source directory not found: {src_bvh_folder}")
        return
    if not os.path.exists(tar_traindata_folder):
        print(f"Creating target directory: {tar_traindata_folder}")
        os.makedirs(tar_traindata_folder)

    bvh_files_names = listdir(src_bvh_folder)
    print(f"Found {len(bvh_files_names)} files/folders in {src_bvh_folder}.")
    processed_count = 0

    for bvh_file_name in bvh_files_names:
        if bvh_file_name.lower().endswith(".bvh"):
            src_bvh_path = os.path.join(src_bvh_folder, bvh_file_name)
            print(f"Processing: {src_bvh_path}")
            try:
                # Using parse_frames from read_bvh to get the raw motion data
                motion_data = read_bvh.parse_frames(src_bvh_path) # Shape: (num_frames, num_channels)

                if motion_data is not None and motion_data.shape[0] > 0:

                    # Start Preprocessing Step ---
                    # Scale the root translation (first 3 channels: Xpos, Ypos, Zpos)
                    if weight_translation != 1.0:
                        motion_data[:, 0:3] *= weight_translation
                    # Converting euler angles (all channels from 4th onwards) to radians
                    motion_data[:, 3:] = np.deg2rad(motion_data[:, 3:])
                    # End Preprocessing Step ---

                    # Define the output .npy file path
                    npy_file_name = bvh_file_name + ".npy"
                    tar_npy_path = os.path.join(tar_traindata_folder, npy_file_name)

                    # Save the processed data
                    np.save(tar_npy_path, motion_data)
                    processed_count += 1

                else:
                    print(f"Warning: No motion data found or parsed in {bvh_file_name}")

            except Exception as e:
                print(f"Error processing {bvh_file_name}: {e}")


    print(f"\nFinished generating Euler data. Processed {processed_count} BVH files.")
    print(f"Output saved to: {tar_traindata_folder}")

    
def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder):
    """
    Reads .npy files containing Euler angle (In radians)/translation data,
    reverses scaling, converts angles to degrees, and writes them back into BVH format.
    """
    if not src_train_folder or not tar_bvh_folder:
        print("Error: Source train data folder or target BVH folder not specified.")
        return
    if not os.path.exists(src_train_folder):
        print(f"Error: Source directory not found: {src_train_folder}")
        return
    if not os.path.exists(tar_bvh_folder):
        print(f"Creating target directory: {tar_bvh_folder}")
        os.makedirs(tar_bvh_folder)
    if not os.path.exists(standard_bvh_file):
         print(f"Error: Standard BVH file for header not found: {standard_bvh_file}")
         print("Ensure the script is run from a location where this relative path is valid, or use an absolute path.")
         return

    npy_files_names = listdir(src_train_folder)
    print(f"Found {len(npy_files_names)} files/folders in {src_train_folder}.")
    processed_count = 0

    for npy_file_name in npy_files_names:
        if npy_file_name.lower().endswith(".npy"):
            src_npy_path = os.path.join(src_train_folder, npy_file_name)
            print(f"Processing: {src_npy_path}")
            try:
                motion_data = np.load(src_npy_path) # Euler angles in radians

                if motion_data is not None and motion_data.shape[0] > 0:

                    # Start Reverse Preprocessing Step ---
                    # Un-scale the root translation if it was scaled during encoding
                    if weight_translation != 1.0:
                        motion_data[:, 0:3] /= weight_translation

                    # Convert Euler angles (all channels from 4th onwards) to DEGREES
                    motion_data[:, 3:] = np.rad2deg(motion_data[:, 3:])
                    # End Reverse Preprocessing Step ---

                    tar_bvh_path = os.path.join(tar_bvh_folder, npy_file_name+".bvh")

                    # Write the data to a BVH file
                    read_bvh.write_frames(standard_bvh_file, tar_bvh_path, motion_data)
                    processed_count += 1
                else:
                    print(f"Warning: No motion data found in {npy_file_name}")

            except Exception as e:
                print(f"Error processing {npy_file_name}: {e}")

    print(f"\nFinished reconstructing BVH files. Processed {processed_count} NPY files.")
    print(f"Output saved to: {tar_bvh_folder}")



standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

print('skeleton: ', skeleton)

bvh_dir_path = "train_data_bvh/salsa/"
euler_enc_dir_path = "train_data_euler/salsa/"
bvh_reconstructed_dir_path = "reconstructed_bvh_data_euler/salsa/"

# Encode data from bvh to positional encoding
generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path)
