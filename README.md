# MAI645-Motion-Synthesis

This project provides a PyTorch implementation of an Auto-Conditioned Recurrent Neural Network (acRNN) for synthesizing complex human motion, based on the work presented in the paper ["Auto-Conditioned Recurrent Networks for Extended Complex Human Motion Synthesis"](https://arxiv.org/abs/1707.05363).

The system learns from motion capture data (in BVH format) and can generate new, continuous dance sequences. This implementation explores three different representations for encoding the motion data:
*   **Positional:** Direct XYZ coordinates for each joint.
*   **Euler Angles:** Rotational data for each joint.
*   **Quaternions:** An alternative rotational representation to avoid issues like gimbal lock.

## Directory Structure
The repository is organized as follows:
```
└── chrysisandreou-mai645_team_03/
    ├── README.md
    └── project645/
        ├── README.md
        ├── LICENSE
        ├── mai645.yml
        └── code/
            ├── analysis.py
            ├── fix_feet.py
            ├── generate_training_euler_data.py
            ├── generate_training_pos_data.py
            ├── generate_training_quad_data.py
            ├── plot_euler_loss.py
            ├── plot_loss.py
            ├── pytorch_train_euler_aclstm.py
            ├── pytorch_train_pos_aclstm.py
            ├── pytorch_train_quad_aclstm.py
            ├── read_bvh.py
            ├── read_bvh_hierarchy.py
            ├── rotation2xyz.py
            ├── rotation_conversions.py
            ├── synthesize_euler_motion.py
            ├── synthesize_pos_motion.py
            ├── synthesize_pos_motion_original_colab.py
            ├── synthesize_quad_motion.py
            ├── test_encodings.py
            └── analysis modify train pos euler to device/
```

## Setup and Installation

### Prerequisites
*   [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) package manager.
*   An NVIDIA GPU is recommended for training.

### Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd chrysisandreou-mai645_team_03/project645
    ```

2.  **Create the Conda environment:**
    The `mai645.yml` file contains all the necessary dependencies. Create and activate the environment using the following commands:
    ```bash
    conda env create -f mai645.yml
    conda activate mai645
    ```

## Usage Workflow

The process involves three main stages: data preparation, training, and synthesis.

### 1. Data Preparation

This project uses motion data in the [Biovision Hierarchy (BVH)](https://en.wikipedia.org/wiki/Biovision_Hierarchy) format. Sample data, including "salsa," "martial," and "indian" dance styles, can be found in the `train_data_bvh` directory (once downloaded).

Before training, you must process the `.bvh` files into a numerical format suitable for the network. Run one of the following scripts from the `code/` directory depending on the desired motion representation.

*   **For Positional representation:**
    ```bash
    python code/generate_training_pos_data.py
    ```
    This script will read `.bvh` files from `../train_data_bvh/salsa/` and save the processed `.npy` files to `../train_data_pos/salsa/`.

*   **For Euler Angle representation:**
    ```bash
    python code/generate_training_euler_data.py
    ```     This will process `.bvh` files and save them as `.npy` files in `../train_data_euler/salsa/`.

*   **For Quaternion representation:**
    ```bash
    python code/generate_training_quad_data.py
    ```
    This script converts `.bvh` data for all dance types (`salsa`, `indian`, `martial`) into quaternion-based `.npy` files, storing them in `../train_data_quad/`.

### 2. Training the Model

After preparing the data, you can train the acLSTM network. Each representation has its own training script.

*   **To train with Positional data:**
    ```bash
    python code/pytorch_train_pos_aclstm.py \
      --dances_folder ../train_data_pos/salsa/ \
      --write_weight_folder ../weights_pos/ \
      --write_bvh_motion_folder ../motion_output_pos/ \
      --in_frame 171 \
      --out_frame 171
    ```

*   **To train with Euler Angle data:**
    ```bash
    python code/pytorch_train_euler_aclstm.py \
      --dances_folder ../train_data_euler/salsa/ \
      --write_weight_folder ../weights_euler/ \
      --write_bvh_motion_folder ../motion_output_euler/ \
      --standard_bvh_reference ../train_data_bvh/standard.bvh 
    ```

*   **To train with Quaternion data:**
    ```bash
    python code/pytorch_train_quad_aclstm.py \
      --dances_folder ../train_data_quad/salsa/ \
      --metadata_path ../train_data_quad/salsa/metadata_quad.json \
      --write_weight_folder ../weights_quad/ \
      --write_bvh_motion_folder ../motion_output_quad/ \
      --standard_bvh_file ../train_data_bvh/standard.bvh
    ```

### 3. Synthesizing New Motion

Once the model is trained, use a synthesis script to generate new motion sequences.

*   **To synthesize with a Positional model:**
    ```bash
    python code/synthesize_pos_motion.py
    ```

*   **To synthesize with an Euler Angle model:**
    ```bash
    python code/synthesize_euler_motion.py
    ```

*   **To synthesize with a Quaternion model:**
    ```bash
    python code/synthesize_quad_motion.py
    ```
    
*You will need to modify the script to point to the correct trained model weights (`.weight` file).*

### 4. Visualizing the Output
The synthesized motion is saved as `.bvh` files. You can visualize these files using various 3D software packages like Blender, Maya, or MotionBuilder. For a quick and easy option, use an online BVH player such as the [Online 3D Viewer](http://lo-th.github.io/olympe/BVH_player.html).

## Pre-trained Models and Data

Due to their large size, the full dataset, project outputs, and final pre-trained model weights are hosted on Google Drive.

*   [**Project Files (Code, Outputs, Models)**](https://drive.google.com/drive/folders/1qNsZ1jETzibiupnLSW3orPf_AzkGtQ3A?usp=sharing)
*   [**Additional Outputs**](https://drive.google.com/drive/folders/1oTZ4W_yMDA3_8ZB-_9wYI_OEWsU2mASB?usp=sharing)

The **`final weights`** subfolder in the first link contains the trained models for each representation, which can be used for synthesis without needing to retrain.

## License
This project is available under the MIT License. See the `LICENSE` file for more details. The license is based on the original implementation by the authors of the paper.
