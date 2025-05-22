import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import json # For metadata if needed for quad
import shutil

# --- Configuration ---
# Assuming this script is in project645/code/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(BASE_DIR, "code") # Should be the directory this script is in
OUTPUT_DIR_FOR_PLOTS = os.path.join(BASE_DIR, "results_quantitative_partB")
os.makedirs(OUTPUT_DIR_FOR_PLOTS, exist_ok=True)

# --- Model and Data Paths (USER MUST VERIFY/PROVIDE THESE) ---

# Common paths - Ensure these are correct relative to your project structure
STANDARD_BVH_REF_PATH = os.path.join(BASE_DIR, "train_data_bvh", "standard.bvh")
POS_DATA_DIR = os.path.join(BASE_DIR, "train_data_pos", "salsa")
EULER_DATA_DIR = os.path.join(BASE_DIR, "train_data_euler", "salsa")
QUAD_DATA_DIR = os.path.join(BASE_DIR, "train_data_quad", "salsa")
QUAD_METADATA_PATH = os.path.join(QUAD_DATA_DIR, "metadata_quad.json")

# Weight paths - VERY IMPORTANT: Replace with actual paths to your .weight files
# These are examples. Adjust filenames/paths as needed.
# e.g., /path/to/your/google_drive_outputs/weights_pos/0049000.weight
POS_WEIGHT_PATH = "/content/drive/MyDrive/mai645_project_outputs/weights_pos/0049000.weight"
EULER_WEIGHT_PATH = "/content/drive/MyDrive/mai645_project_outputs/weights_euler/0051000.weight"
QUAD_CON5_WEIGHT_PATH = "/content/drive/MyDrive/mai645_project_outputs_con5/weights_quad/0028000.weight" 
QUAD_CON30_WEIGHT_PATH = "/content/drive/MyDrive/mai645_project_outputs_con30/weights_quad/0031000.weight" 
QUAD_CON45_WEIGHT_PATH = "/content/drive/MyDrive/mai645_project_outputs_con45/weights_quad/0049000.weight" 

# Check if essential common files exist
if not os.path.exists(STANDARD_BVH_REF_PATH):
    print(f"ERROR: Standard BVH reference not found at {STANDARD_BVH_REF_PATH}"); exit()
if not os.path.exists(QUAD_METADATA_PATH):
    print(f"ERROR: Quaternion metadata not found at {QUAD_METADATA_PATH}"); exit()

# --- Experiment Configurations ---
EXPERIMENTS = [
    {
        "name": "Positional",
        "script": "synthesize_pos_motion.py",
        "weight_path": POS_WEIGHT_PATH,
        "data_folder": POS_DATA_DIR + os.sep,
        "metric_regex": r"Quantitative Evaluation - MSE for first \d+ generated frames: (\d+\.\d+)",
        "fixed_args": [ 
            "--in_frame_size", "171",
            "--out_frame_size", "171",
            "--hidden_size", "1024",
            "--dance_frame_rate", "60", 
        ],
        "metric_type": "MSE"
    },
    {
        "name": "Euler",
        "script": "synthesize_euler_motion.py",
        "weight_path": EULER_WEIGHT_PATH,
        "data_folder": EULER_DATA_DIR,
        "metric_regex": r"Hip MSE: (\d+\.\d+).*Rot Ang Dist: (\d+\.\d+)", # Captures both
        "fixed_args": [
            "--standard_bvh_reference", STANDARD_BVH_REF_PATH,
            "--in_frame_size", "132", 
            "--out_frame_size", "132",
            "--hidden_size", "1024",
             "--dance_frame_rate", "60", 
        ],
        "metric_type": "Summed Loss (Hip MSE + Rot AD)"
    },
    {
        "name": "Quaternion (Con5)",
        "script": "synthesize_quad_motion.py",
        "weight_path": QUAD_CON5_WEIGHT_PATH,
        "data_folder": QUAD_DATA_DIR,
        "metric_regex": r"Total Loss: (\d+\.\d+)",
        "fixed_args": [
            "--standard_bvh_file", STANDARD_BVH_REF_PATH,
            "--metadata_path", QUAD_METADATA_PATH,
            "--hidden_size", "1024",
        ],
        "metric_type": "Total Loss (Hip MSE + Quat Loss)"
    },
    {
        "name": "Quaternion (Con30)",
        "script": "synthesize_quad_motion.py",
        "weight_path": QUAD_CON30_WEIGHT_PATH,
        "data_folder": QUAD_DATA_DIR,
        "metric_regex": r"Total Loss: (\d+\.\d+)",
        "fixed_args": ["--standard_bvh_file", STANDARD_BVH_REF_PATH, "--metadata_path", QUAD_METADATA_PATH, "--hidden_size", "1024"],
        "metric_type": "Total Loss (Hip MSE + Quat Loss)"
    },
    {
        "name": "Quaternion (Con45)",
        "script": "synthesize_quad_motion.py",
        "weight_path": QUAD_CON45_WEIGHT_PATH,
        "data_folder": QUAD_DATA_DIR,
        "metric_regex": r"Total Loss: (\d+\.\d+)",
        "fixed_args": ["--standard_bvh_file", STANDARD_BVH_REF_PATH, "--metadata_path", QUAD_METADATA_PATH, "--hidden_size", "1024"],
        "metric_type": "Total Loss (Hip MSE + Quat Loss)"
    },
]

# --- Experiment Parameters ---
INITIAL_SEQ_LENS = [100, 200, 300]
NUM_RUNS_PER_CONFIG = 3
QUANTITATIVE_COMPARISON_LEN = 20
SYNTHESIS_BVH_OUTPUT_TEMP_DIR = os.path.join(CODE_DIR, "temp_synthesis_output_partB")
os.makedirs(SYNTHESIS_BVH_OUTPUT_TEMP_DIR, exist_ok=True)

# --- Helper Function to Parse Metric ---
def parse_metric_from_output(output_str, regex_pattern, experiment_name_for_parsing):
    match = re.search(regex_pattern, output_str)
    if match:
        if experiment_name_for_parsing == "Euler" and len(match.groups()) == 2:
            hip_loss = float(match.group(1))
            rot_loss = float(match.group(2))
            # print(f"    Parsed Euler - Hip: {hip_loss:.4f}, Rot: {rot_loss:.4f}, Sum: {hip_loss + rot_loss:.4f}")
            return hip_loss + rot_loss
        elif len(match.groups()) == 1:
            metric_val = float(match.group(1))
            # print(f"    Parsed {experiment_name_for_parsing}: {metric_val:.4f}")
            return metric_val
        else:
            print(f"    Warning: Regex for {experiment_name_for_parsing} matched but incorrect group count: {len(match.groups())}")
            return None
    else:
        # print(f"    Warning: Metric pattern not found for {experiment_name_for_parsing} in output.") 
        return None

# --- Main Loop ---
results_data = {}

for config in EXPERIMENTS:
    exp_name = config["name"]
    print(f"\n--- Running Experiment: {exp_name} ---")

    if not os.path.exists(config["weight_path"]) or "YOUR_" in config["weight_path"].upper() :
        print(f"SKIPPING {exp_name}: Weight file not found or is a placeholder at {config['weight_path']}")
        continue
    if not os.path.exists(config["data_folder"]):
        print(f"SKIPPING {exp_name}: Data folder not found at {config['data_folder']}")
        continue

    results_data[exp_name] = {"metric_type": config["metric_type"], "errors_mean": [], "errors_std": []}
    
    for seq_len in INITIAL_SEQ_LENS:
        print(f"  Testing with initial_seq_len: {seq_len}")
        current_config_errors = []
        for run_idx in range(NUM_RUNS_PER_CONFIG):
            print(f"    Run {run_idx + 1}/{NUM_RUNS_PER_CONFIG} for seq_len {seq_len}...")
            
            cmd_parts = [
                "python", os.path.join(CODE_DIR, config["script"]),
                "--read_weight_path", config["weight_path"],
                "--dances_folder", config["data_folder"],
                "--initial_seq_len", str(seq_len),
                "--quantitative_comparison_len", str(QUANTITATIVE_COMPARISON_LEN),
                "--generate_frames_number", str(QUANTITATIVE_COMPARISON_LEN + 10), # Generate enough for comparison
                "--batch_size", "1",
                "--write_bvh_motion_folder", SYNTHESIS_BVH_OUTPUT_TEMP_DIR
            ]
            cmd_parts.extend(config["fixed_args"])

            try:
                # Use a timeout to prevent hanging indefinitely if a script has issues
                process = subprocess.run(cmd_parts, capture_output=True, text=True, check=False, cwd=CODE_DIR, timeout=300) # 5 min timeout
                
                if process.returncode != 0:
                    print(f"    ERROR running {config['script']} (ret {process.returncode}):\nCmd: {' '.join(cmd_parts)}\nStderr: {process.stderr[:500]}...")
                    continue

                metric = parse_metric_from_output(process.stdout, config["metric_regex"], exp_name)
                if metric is not None:
                    current_config_errors.append(metric)
                    print(f"      Run {run_idx+1} metric: {metric:.4f}")
                else:
                    print(f"    Failed to parse metric for {exp_name}, seq_len {seq_len}, run {run_idx+1}. Stdout was:\n{process.stdout[:1000]}...")
            
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT running {config['script']} for {exp_name}, seq_len {seq_len}, run {run_idx+1}.")
            except Exception as e:
                print(f"    UNEXPECTED EXCEPTION running {config['script']}: {e}")

        if current_config_errors:
            mean_err = np.mean(current_config_errors)
            std_err = np.std(current_config_errors)
            print(f"  Avg error for initial_seq_len {seq_len}: {mean_err:.4f} (+/- {std_err:.4f}) from {len(current_config_errors)} runs.")
            results_data[exp_name]["errors_mean"].append(mean_err)
            results_data[exp_name]["errors_std"].append(std_err)
        else:
            print(f"  No successful runs recorded for initial_seq_len {seq_len}. Appending NaN.")
            results_data[exp_name]["errors_mean"].append(np.nan)
            results_data[exp_name]["errors_std"].append(np.nan)

# --- Plotting ---
# Determine if we need subplots (if metric types are different)
unique_metric_types = sorted(list(set(data["metric_type"] for name, data in results_data.items() if data["errors_mean"])))

if not unique_metric_types:
    print("No results to plot.")
else:
    num_subplots = len(unique_metric_types)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(14, 7 * num_subplots), sharex=True, squeeze=False)
    axes = axes.flatten() # Ensure axes is always a 1D array

    for i, metric_type_to_plot in enumerate(unique_metric_types):
        ax = axes[i]
        for exp_name, data in results_data.items():
            if data["metric_type"] == metric_type_to_plot and data["errors_mean"]:
                # Plotting with error bars if std is available
                means = np.array(data["errors_mean"])
                stds = np.array(data["errors_std"])
                
                valid_indices = ~np.isnan(means)
                plot_seq_lens = np.array(INITIAL_SEQ_LENS)[valid_indices]
                plot_means = means[valid_indices]
                plot_stds = stds[valid_indices]

                if len(plot_means) > 0:
                    ax.plot(plot_seq_lens, plot_means, marker='o', linestyle='-', label=exp_name)
                    ax.fill_between(plot_seq_lens, plot_means - plot_stds, plot_means + plot_stds, alpha=0.2)
        
        ax.set_ylabel(f"Average Error ({metric_type_to_plot})")
        ax.set_title(f"Impact of Seed Length on Prediction Error ({metric_type_to_plot})")
        ax.legend(loc='best')
        ax.grid(True)
        ax.set_xticks(INITIAL_SEQ_LENS) # Ensure x-ticks are exactly the tested sequence lengths

    axes[-1].set_xlabel("Initial Seed Sequence Length (frames)")
    fig.suptitle("Quantitative Performance vs. Seed Length by Representation", fontsize=18, y=0.99 if num_subplots==1 else 1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96 if num_subplots > 1 else 0.95])


    plot_filename = os.path.join(OUTPUT_DIR_FOR_PLOTS, "quantitative_comparison_plot.png")
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    plt.show()

# --- Clean up temporary synthesis directory ---
try:
    if os.path.exists(SYNTHESIS_BVH_OUTPUT_TEMP_DIR):
        shutil.rmtree(SYNTHESIS_BVH_OUTPUT_TEMP_DIR)
        print(f"Cleaned up temporary directory: {SYNTHESIS_BVH_OUTPUT_TEMP_DIR}")
except Exception as e:
    print(f"Warning: Could not clean up temp directory {SYNTHESIS_BVH_OUTPUT_TEMP_DIR}: {e}")

print("\nExperiment Script Finished.")
