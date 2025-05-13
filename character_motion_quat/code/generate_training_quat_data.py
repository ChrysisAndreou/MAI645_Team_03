import read_bvh
import numpy as np
from os import listdir
import os
import transforms3d.euler as euler
import json

standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
bvh_dir_path = "train_data_bvh/salsa/"
quat_enc_dir_path = "train_data_quat/salsa/"
bvh_reconstructed_dir_path = "reconstructed_bvh_data_quat/salsa/"

# --- Ensuring output directories exist ---
if not os.path.exists(quat_enc_dir_path):
    os.makedirs(quat_enc_dir_path)
    print(f"Created directory: {quat_enc_dir_path}")
if not os.path.exists(bvh_reconstructed_dir_path):
    os.makedirs(bvh_reconstructed_dir_path)
    print(f"Created directory: {bvh_reconstructed_dir_path}")
# --- ---

skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

# Storing rotation orders for each joint
joint_rotation_orders = {}
num_translation_channels = 0
num_euler_rotation_channels_total = 0 # For calculating expected Euler frame size
num_quat_rotation_channels_total = 0  # For calculating quaternion frame size
rotating_joint_names_in_order = [] # To maintain order for data construction

# Process Root
if 'hip' in skeleton and 'channels' in skeleton['hip']:
    root_channels = skeleton['hip']['channels']
    if 'Xposition' in root_channels: num_translation_channels += 1
    if 'Yposition' in root_channels: num_translation_channels += 1
    if 'Zposition' in root_channels: num_translation_channels += 1

    rot_ch_names = [ch for ch in root_channels if 'rotation' in ch.lower()]
    if len(rot_ch_names) == 3:
        if rot_ch_names == ['Zrotation', 'Yrotation', 'Xrotation']:
            joint_rotation_orders['hip'] = 'rzyx'
        elif rot_ch_names == ['Zrotation', 'Xrotation', 'Yrotation']:
            joint_rotation_orders['hip'] = 'rzxy'
        elif rot_ch_names == ['Xrotation', 'Yrotation', 'Zrotation']:
            joint_rotation_orders['hip'] = 'rxyz'
        elif rot_ch_names == ['Yrotation', 'Zrotation', 'Xrotation']:
            joint_rotation_orders['hip'] = 'ryzx'
        else:
            print(f"Warning: Unhandled rotation order for root 'hip': {rot_ch_names}. Assuming 'rzyx'.")
            joint_rotation_orders['hip'] = 'rzyx'
        num_euler_rotation_channels_total += 3
        num_quat_rotation_channels_total += 4
        rotating_joint_names_in_order.append('hip')
    elif len(rot_ch_names) != 0 :
        print(f"Warning: Root 'hip' has {len(rot_ch_names)} rotation channels. Expected 0 or 3.")

_temp_rotating_joints_after_root = []
for joint_name in non_end_bones:
    if joint_name == 'hip':
        continue
    if joint_name in skeleton and 'channels' in skeleton[joint_name]:
        joint_ch = skeleton[joint_name]['channels']
        rot_ch_names = [ch for ch in joint_ch if 'rotation' in ch.lower()]
        if len(rot_ch_names) == 3:
            if rot_ch_names == ['Zrotation', 'Yrotation', 'Xrotation']:
                joint_rotation_orders[joint_name] = 'rzyx'
            elif rot_ch_names == ['Zrotation', 'Xrotation', 'Yrotation']:
                joint_rotation_orders[joint_name] = 'rzxy'
            elif rot_ch_names == ['Xrotation', 'Yrotation', 'Zrotation']:
                joint_rotation_orders[joint_name] = 'rxyz'
            elif rot_ch_names == ['Yrotation', 'Zrotation', 'Xrotation']:
                joint_rotation_orders[joint_name] = 'ryzx'
            else:
                print(f"Warning: Unhandled rotation order for joint '{joint_name}': {rot_ch_names}. Assuming 'rzxy'.")
                joint_rotation_orders[joint_name] = 'rzxy'
            num_euler_rotation_channels_total += 3
            num_quat_rotation_channels_total += 4
            _temp_rotating_joints_after_root.append(joint_name)
        elif len(rot_ch_names) != 0:
            print(f"Warning: Joint '{joint_name}' has {len(rot_ch_names)} rotation channels. Expected 0 or 3.")

rotating_joint_names_in_order.extend(_temp_rotating_joints_after_root)

calculated_euler_frame_size = num_translation_channels + num_euler_rotation_channels_total
calculated_quat_frame_size = num_translation_channels + num_quat_rotation_channels_total

print(f"Skeleton Info (Revised):")
print(f"  - Translation Channels: {num_translation_channels}")
print(f"  - Rotating Joints in order: {rotating_joint_names_in_order}")
print(f"  - Rotation orders: {joint_rotation_orders}")
print(f"  - Total Euler Rotation Channels: {num_euler_rotation_channels_total}")
print(f"  - Expected Euler Frame Size: {calculated_euler_frame_size}")
print(f"  - Total Quaternion Rotation Channels: {num_quat_rotation_channels_total}")
print(f"  - Calculated Quaternion Frame Size: {calculated_quat_frame_size}")

def get_quaternion_data_from_euler(euler_motion_data, joint_names_with_rotations, per_joint_orders):
    num_frames = euler_motion_data.shape[0]
    num_euler_channels_in_data = euler_motion_data.shape[1]

    if num_euler_channels_in_data != calculated_euler_frame_size:
        print(f"DATA WARNING: Input Euler data has {num_euler_channels_in_data} channels, "
              f"but skeleton parsing expected {calculated_euler_frame_size}. Proceeding with data size.")
        _num_trans = num_translation_channels
        _num_euler_rot_ch_data = num_euler_channels_in_data - _num_trans
        if _num_euler_rot_ch_data % 3 != 0:
            raise ValueError("Euler rotation channels in data not multiple of 3.")
        _num_rot_joints_in_data = _num_euler_rot_ch_data // 3
        _quat_frame_size = _num_trans + _num_rot_joints_in_data * 4
    else:
        _num_rot_joints_in_data = len(joint_names_with_rotations)
        _quat_frame_size = calculated_quat_frame_size

    quat_motion_data = np.zeros((num_frames, _quat_frame_size))
    if num_translation_channels > 0:
        quat_motion_data[:, :num_translation_channels] = euler_motion_data[:, :num_translation_channels]

    current_euler_channel_idx = num_translation_channels
    current_quat_channel_idx = num_translation_channels

    for i in range(_num_rot_joints_in_data):
        if i >= len(joint_names_with_rotations):
            print(f"  Data has more rotating joint blocks ({_num_rot_joints_in_data}) than defined in standard.bvh ({len(joint_names_with_rotations)}). Using generic name for extra joint {i}.")
            joint_name = f"unknown_joint_{i}"
        else:
            joint_name = joint_names_with_rotations[i]

        rot_order = per_joint_orders.get(joint_name, 'rzyx')

        angle1_deg = euler_motion_data[:, current_euler_channel_idx]
        angle2_deg = euler_motion_data[:, current_euler_channel_idx + 1]
        angle3_deg = euler_motion_data[:, current_euler_channel_idx + 2]

        angle1_rad = np.deg2rad(angle1_deg)
        angle2_rad = np.deg2rad(angle2_deg)
        angle3_rad = np.deg2rad(angle3_deg)

        # Assuming rot_order is like 'rzyx', transforms3d expects 'szyx' for intrinsic Euler.
        # The order of angles (a1, a2, a3) must match the axes string.
        # If rot_order is 'rzyx', it means Rz(a1) * Ry(a2) * Rx(a3).
        # For euler2quat, axes='szyx' means apply Z, then Y, then X. So a1 is Z, a2 is Y, a3 is X.
        # This matches the 'rzyx' convention if a1, a2, a3 are Z, Y, X angles respectively.
        quats = np.array([euler.euler2quat(a1, a2, a3, axes='s'+rot_order[1:]) # Use 's' prefix for intrinsic, and actual order from rot_order
                          for a1, a2, a3 in zip(angle1_rad, angle2_rad, angle3_rad)])
        quat_motion_data[:, current_quat_channel_idx : current_quat_channel_idx + 4] = quats

        current_euler_channel_idx += 3
        current_quat_channel_idx += 4
    return quat_motion_data

def get_euler_data_from_quaternion(quat_motion_data, joint_names_with_rotations, per_joint_orders):
    num_frames = quat_motion_data.shape[0]
    num_quat_channels_in_data = quat_motion_data.shape[1]

    if num_quat_channels_in_data != calculated_quat_frame_size:
        print(f"DATA WARNING: Input Quat data has {num_quat_channels_in_data} channels, "
              f"but skeleton parsing expected {calculated_quat_frame_size}. Proceeding with data size.")
        _num_trans = num_translation_channels
        _num_quat_rot_ch_data = num_quat_channels_in_data - _num_trans
        if _num_quat_rot_ch_data % 4 != 0:
            raise ValueError("Quat rotation channels in data not multiple of 4.")
        _num_rot_joints_in_data = _num_quat_rot_ch_data // 4
        _euler_frame_size = _num_trans + _num_rot_joints_in_data * 3
    else:
        _num_rot_joints_in_data = len(joint_names_with_rotations)
        _euler_frame_size = calculated_euler_frame_size

    euler_motion_data = np.zeros((num_frames, _euler_frame_size))
    if num_translation_channels > 0:
        euler_motion_data[:, :num_translation_channels] = quat_motion_data[:, :num_translation_channels]

    current_quat_channel_idx = num_translation_channels
    current_euler_channel_idx = num_translation_channels

    for i in range(_num_rot_joints_in_data):
        if i >= len(joint_names_with_rotations):
            joint_name = f"unknown_joint_{i}"
        else:
            joint_name = joint_names_with_rotations[i]
        rot_order = per_joint_orders.get(joint_name, 'rzyx')

        quats = quat_motion_data[:, current_quat_channel_idx : current_quat_channel_idx + 4]
        norm_quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)
        euler_angles_rad = np.array([euler.quat2euler(q, axes='s'+rot_order[1:]) for q in norm_quats])
        euler_motion_data[:, current_euler_channel_idx : current_euler_channel_idx + 3] = np.rad2deg(euler_angles_rad)

        current_quat_channel_idx += 4
        current_euler_channel_idx += 3
    return euler_motion_data


def generate_quat_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if not os.path.exists(tar_traindata_folder): os.makedirs(tar_traindata_folder)
    bvh_files_names = [f for f in listdir(src_bvh_folder) if f.lower().endswith(".bvh")]
    print(f"Found {len(bvh_files_names)} BVH files in {src_bvh_folder}.")
    processed_count = 0
    for bvh_file_name_with_ext in bvh_files_names: # e.g., "01.bvh"
        src_bvh_path = os.path.join(src_bvh_folder, bvh_file_name_with_ext)
        print(f"Processing: {src_bvh_path}")
        try:
            motion_data_euler = read_bvh.parse_frames(src_bvh_path)
            if motion_data_euler is not None and motion_data_euler.shape[0] > 0:
                if motion_data_euler.shape[1] != calculated_euler_frame_size:
                    print(f"  BVH data from {bvh_file_name_with_ext} has {motion_data_euler.shape[1]} channels, expected {calculated_euler_frame_size} from standard.bvh...")

                if weight_translation != 1.0 and num_translation_channels > 0:
                    motion_data_euler[:, :num_translation_channels] *= weight_translation

                motion_data_quat = get_quaternion_data_from_euler(motion_data_euler, rotating_joint_names_in_order, joint_rotation_orders)

                tar_npy_filename = bvh_file_name_with_ext + ".npy" # "01.bvh" + ".npy" -> "01.bvh.npy"
                tar_npy_path = os.path.join(tar_traindata_folder, tar_npy_filename)
                np.save(tar_npy_path, motion_data_quat)
                processed_count += 1
            else:
                print(f"Warning: No motion data parsed in {bvh_file_name_with_ext}")
        except Exception as e:
            print(f"Error processing {bvh_file_name_with_ext}: {e}")
    print(f"\nFinished generating Quaternion data. Processed {processed_count} BVH files.")
    print(f"Output saved to: {tar_traindata_folder}")

    if processed_count > 0:
        metadata = {
            "joint_rotation_orders": joint_rotation_orders,
            "rotating_joint_names_in_order": rotating_joint_names_in_order,
            "calculated_quat_frame_size": calculated_quat_frame_size,
            "calculated_euler_frame_size": calculated_euler_frame_size,
            "num_translation_channels": num_translation_channels,
            "weight_translation_factor": weight_translation,
            "standard_bvh_file_source": standard_bvh_file,
        }
        metadata_path = os.path.join(tar_traindata_folder, "metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Saved metadata to {metadata_path}")
        except Exception as e:
            print(f"Error saving metadata.json: {e}")
    else:
        print("No BVH files were successfully processed, so metadata.json was not created.")


def generate_bvh_from_quat_traindata(src_train_folder, tar_bvh_folder):
    if not os.path.exists(tar_bvh_folder): os.makedirs(tar_bvh_folder)
    if not os.path.exists(standard_bvh_file):
         print(f"Error: Standard BVH file for header not found: {standard_bvh_file}"); return
    # We expect npy files named like "01.bvh.npy"
    npy_files_names = [f for f in listdir(src_train_folder) if f.lower().endswith(".bvh.npy")]
    print(f"Found {len(npy_files_names)} NPY files (ending with .bvh.npy) in {src_train_folder}.")
    processed_count = 0
    for npy_file_name_with_ext in npy_files_names: # e.g., "01.bvh.npy"
        src_npy_path = os.path.join(src_train_folder, npy_file_name_with_ext)
        print(f"Processing: {src_npy_path}")
        try:
            motion_data_quat = np.load(src_npy_path)
            if motion_data_quat is not None and motion_data_quat.shape[0] > 0:
                if motion_data_quat.shape[1] != calculated_quat_frame_size:
                     print(f"  NPY data from {npy_file_name_with_ext} has {motion_data_quat.shape[1]} channels, expected {calculated_quat_frame_size}...")

                if weight_translation != 1.0 and num_translation_channels > 0:
                     motion_data_quat[:, :num_translation_channels] /= weight_translation

                motion_data_euler_deg = get_euler_data_from_quaternion(motion_data_quat, rotating_joint_names_in_order, joint_rotation_orders)

                tar_bvh_filename = npy_file_name_with_ext + ".bvh" # "01.bvh.npy" + ".bvh" -> "01.bvh.npy.bvh"
                tar_bvh_path = os.path.join(tar_bvh_folder, tar_bvh_filename)
                read_bvh.write_frames(standard_bvh_file, tar_bvh_path, motion_data_euler_deg)
                processed_count += 1
            else:
                print(f"Warning: No motion data found in {npy_file_name_with_ext}")
        except Exception as e:
            print(f"Error processing {npy_file_name_with_ext}: {e}")
    print(f"\nFinished reconstructing BVH files from Quat data. Processed {processed_count} NPY files.")
    print(f"Output saved to: {tar_bvh_folder}")


if __name__ == "__main__":
    print(f"Using Standard BVH: {standard_bvh_file}")
    print(f"Target Quaternion Frame Size based on standard.bvh: {calculated_quat_frame_size}")
    print(f"Expected Euler Frame Size based on standard.bvh: {calculated_euler_frame_size}")

    print("\n--- Encoding BVH to Quaternion NPY ---")
    generate_quat_traindata_from_bvh(bvh_dir_path, quat_enc_dir_path)

    print("\n--- Decoding Quaternion NPY to BVH ---")
    generate_bvh_from_quat_traindata(quat_enc_dir_path, bvh_reconstructed_dir_path)

    print("\nProcessing complete.")