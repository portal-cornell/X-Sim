#!/usr/bin/env python3
"""
X-Sim Pipeline Runner
Automatically runs the complete X-Sim pipeline for a given task.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Task configurations
TASK_CONFIGS = {
    "Mustard-Place": {
        "num_steps": 200,
        "num_eval_steps": 200,
        "trajectory_length": 100,
        "num_trajectories": 10
    },
    "Corn-in-Basket": {
        "num_steps": 300,
        "num_eval_steps": 300,
        "trajectory_length": 200,
        "num_trajectories": 10
    },
    "Letter-Arrange": {
        "num_steps": 50,
        "num_eval_steps": 50,
        "trajectory_length": 50,
        "num_trajectories": 10
    },
    "Shoe-on-Rack": {
        "num_steps": 200,
        "num_eval_steps": 200,
        "trajectory_length": 100,
        "num_trajectories": 10
    },
    "Mug-Insert": {
        "num_steps": 100,
        "num_eval_steps": 100,
        "trajectory_length": 100,
        "num_trajectories": 10
    }
}

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    # print(f"Running: {cmd}")
    # print("-" * 60)
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    print(f"SUCCESS: {description} completed!")

def find_latest_checkpoint(base_path, pattern="ckpt_*.pt"):
    """Find the latest checkpoint file in a directory."""
    import glob
    checkpoints = glob.glob(os.path.join(base_path, "**", pattern), recursive=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_path}")
    
    # Sort by modification time and return the latest
    latest = max(checkpoints, key=os.path.getmtime)
    return latest

def main():
    parser = argparse.ArgumentParser(description="Run complete X-Sim pipeline")
    parser.add_argument("--env_id", required=True, choices=list(TASK_CONFIGS.keys()),
                       help="Environment ID to run pipeline for")
    parser.add_argument("--exp_name", default=None,
                       help="Experiment name (default: auto-generated)")
    parser.add_argument("--num_envs", type=int, default=1024,
                       help="Number of parallel environments for RL training")
    parser.add_argument("--total_timesteps", type=int, default=200000000,
                       help="Total timesteps for RL training (overrides task default)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--skip_rl", action="store_true",
                       help="Skip RL training step")
    parser.add_argument("--skip_data_gen", action="store_true",
                       help="Skip synthetic data generation step")
    parser.add_argument("--skip_dp_baseline", action="store_true",
                       help="Skip baseline diffusion policy training")
    parser.add_argument("--skip_calibration", action="store_true",
                       help="Skip auto-calibration step")
    parser.add_argument("--skip_dp_aux", action="store_true",
                       help="Skip diffusion policy training with auxiliary loss")
    
    args = parser.parse_args()
    
    # Get task configuration
    if args.env_id not in TASK_CONFIGS:
        print(f"Error: Unknown environment {args.env_id}")
        print(f"Available environments: {list(TASK_CONFIGS.keys())}")
        sys.exit(1)
    
    config = TASK_CONFIGS[args.env_id]
    
    # Generate experiment name if not provided
    exp_name = args.exp_name or f"pipeline/{args.env_id.lower().replace('-', '_')}"
    
    # Create directory structure
    base_dir = f"experiments/{exp_name}"
    rl_dir = f"{base_dir}/rl"
    data_dir = f"{base_dir}/synthetic_data"
    dp_baseline_dir = f"{base_dir}/dp_baseline"
    dp_aux_dir = f"{base_dir}/dp_with_aux"
    
    print(f"\n🚀 Starting X-Sim pipeline for {args.env_id}")
    print(f"Experiment directory: {base_dir}")
    print(f"Configuration: {config}")
    
    # Step 1: RL Training
    if not args.skip_rl:
        rl_cmd = (
            f"cd simulation && python -m scripts.rl_training "
            f"--env_id='{args.env_id}' "
            f"--exp-name='../{rl_dir}' "
            f"--num_envs={args.num_envs} "
            f"--seed={args.seed} "
            f"--total_timesteps={args.total_timesteps} "
            f"--num_steps={config['num_steps']} "
            f"--num_eval_steps={config['num_eval_steps']}"
        )
        run_command(rl_cmd, "RL Training")
    
    # Find RL checkpoint
    try:
        rl_checkpoint = find_latest_checkpoint(rl_dir)
        print(f"Using RL checkpoint: {rl_checkpoint}")
    except FileNotFoundError:
        print(f"ERROR: No RL checkpoint found in {rl_dir}")
        if not args.skip_rl:
            sys.exit(1)
        else:
            print("Skipping remaining steps due to missing RL checkpoint")
            return
    
    # Step 2: Synthetic Data Generation
    if not args.skip_data_gen:
        data_cmd = (
            f"cd simulation && python -m scripts.data_generation_rgb "
            f"--evaluate "
            f"--num_trajectories={config['num_trajectories']} "
            f"--trajectory_length={config['trajectory_length']} "
            f"--randomize_init_config "
            f"--checkpoint='../{rl_checkpoint}' "
            f"--env_id='{args.env_id}-Eval' "
            f"--randomize_camera "
            f"--output_dir='../{data_dir}'"
        )
        run_command(data_cmd, "Synthetic Data Generation")
    
    # Step 3: Baseline Diffusion Policy Training
    if not args.skip_dp_baseline:
        dp_baseline_cmd = (
            f"cd diffusion_policy && python -m scripts.dp_training_rgb "
            f"--config_path=cfgs/sim2real.yaml "
            f"--dp.use_aux_loss=0 "
            f"--dp.aux_loss_weight=0.1 "
            f"--dp.distance_type='contrastive_cosine' "
            f"--save_dir='../{dp_baseline_dir}' "
            f"--dataset.paths=['../{data_dir}'] "
            f"--eval_freq=5 "
            f"--eval.env_id='{args.env_id}-Eval' "
            f"--eval.num_episodes=10 "
            f"--eval.save_trajectories=false "
            f"--dp.use_additional_aug=0 "
            f"--epoch_len=100 "
            f"--num_epoch=50"
        )
        run_command(dp_baseline_cmd, "Baseline Diffusion Policy Training")
    
    # Step 4: Auto-Calibration (requires real data - placeholder for now)
    if not args.skip_calibration:
        print(f"\n{'='*60}")
        print("STEP: Auto-Calibration")
        print(f"{'='*60}")
        print("NOTE: Auto-calibration requires real robot rollout data.")
        print("Please collect real rollouts and run:")
        print(f"cd diffusion_policy && python -m scripts.auto_calibration --input_dir='<path_to_real_rollouts>' --env_id='{args.env_id}-Eval'")
        print("Then update the paths below for Step 5.")
    
    # Step 5: Diffusion Policy Training with Auxiliary Loss
    if not args.skip_dp_aux:
        print(f"\n{'='*60}")
        print("STEP: Diffusion Policy Training with Auxiliary Loss")
        print(f"{'='*60}")
        print("NOTE: This step requires real data pairing from auto-calibration.")
        print("Please update the paths below with your real data:")
        print()
        dp_aux_cmd = (
            f"cd diffusion_policy && python -m scripts.dp_training_rgb "
            f"--config_path=cfgs/sim2real.yaml "
            f"--dp.use_aux_loss=1 "
            f"--dp.aux_loss_weight=0.1 "
            f"--dp.distance_type='contrastive_cosine' "
            f"--save_dir='../{dp_aux_dir}' "
            f"--dataset.paths=['../{data_dir}'] "
            f"--dataset.real_pairing='<path_to_real_data>' "
            f"--dataset.sim_pairing='<path_to_sim_pairing>' "
            f"--eval_freq=5 "
            f"--eval.env_id='{args.env_id}-Eval' "
            f"--eval.num_episodes=10 "
            f"--eval.save_trajectories=false "
            f"--dp.use_additional_aug=0 "
            f"--epoch_len=100 "
            f"--num_epoch=50"
        )
        print(f"Command to run: {dp_aux_cmd}")
    
    print(f"\n🎉 Pipeline completed for {args.env_id}!")
    print(f"Results saved in: {base_dir}")
    print("\nNext steps:")
    print("1. Collect real robot rollouts for auto-calibration")
    print("2. Run auto-calibration script")
    print("3. Train final policy with auxiliary loss using real data pairing")

if __name__ == "__main__":
    main() 