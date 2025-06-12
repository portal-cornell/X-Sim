from dataclasses import dataclass, field
import os
import sys
import yaml
import numpy as np
import torch
import torch.backends.cudnn
import diffusers
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Optional, List
import pyrallis

# Import our evaluation module
from .eval_dp import EvalConfig, setup_environment, run_evaluation

# Import utility classes
import utils.common_utils as common_utils
from utils.dataset_utils.dataset import BCDataset, BCDatasetConfig, ObsProcessor
from utils.models.diffusion_policy import DiffusionPolicyWithAuxLoss, DiffusionPolicyWithAuxLossConfig

from datetime import datetime

@dataclass
class MainConfig(common_utils.RunConfig):
    dataset: BCDatasetConfig = field(default_factory=BCDatasetConfig)
    dp: DiffusionPolicyWithAuxLossConfig = field(default_factory=DiffusionPolicyWithAuxLossConfig)
    method: str = "dp"  # dp, act
    norm_action: int = 1
    # training
    seed: int = 1
    num_epoch: int = 20
    epoch_len: int = 10000
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 5
    weight_decay: float = 0
    use_ema: int = 1
    ema_tau: float = 0.01
    cosine_schedule: int = 0
    lr_warm_up_steps: int = 0
    val_freq: int = 5  # Validation frequency 
    eval_freq: int = 5  # Simulation evaluation frequency (every K epochs)
    # evaluation
    eval: EvalConfig = field(default_factory=EvalConfig)
    # log
    save_dir: Optional[str] = None
    use_wb: int = 0
    use_pretrained: str = ''

    def __post_init__(self):
        if self.save_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            self.save_dir = f"baseline_experiment_logs/run_{self.method}_{date_str}"

def run(cfg: MainConfig):
    print(common_utils.wrap_ruler("Train dataset"))
    dataset = BCDataset(cfg.dataset)
    date_str = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    cfg.save_dir = f"{cfg.save_dir}_{date_str}"
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Save action range and camera info
    action_min, action_max = dataset.get_action_range()
    np.savez(os.path.join(cfg.save_dir, "dataset_info.npz"), 
             action_min=action_min.cpu().numpy(),
             action_max=action_max.cpu().numpy(),
             camera_views=dataset.camera_views,
             image_size=cfg.dataset.image_size,
             obs_shape=dataset.obs_shape,
             prop_dim=dataset.prop_dim)
    
    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
    print(common_utils.wrap_ruler("config"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read(), end="")
    cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

    stat = common_utils.MultiCounter(
        cfg.save_dir,
        bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
    )

    # Create the policy
    policy = DiffusionPolicyWithAuxLoss(
        obs_horizon=1,
        obs_shape=dataset.obs_shape,
        prop_dim=dataset.prop_dim,
        action_dim=dataset.action_dim,
        camera_views=dataset.camera_views,
        cfg=cfg.dp,
    ).to("cuda")

    if cfg.use_pretrained and cfg.use_pretrained != 'None':
        print(f"Loading pretrained model from {cfg.use_pretrained}")
        policy.load_state_dict(torch.load(cfg.use_pretrained))
        print("Pretrained model loaded successfully")
        
    if cfg.norm_action:
        policy.init_action_normalizer(*dataset.get_action_range())

    print(common_utils.wrap_ruler("policy weights"))
    print(policy)

    # Create evaluation environment once
    if cfg.eval_freq > 0:
        print(common_utils.wrap_ruler("Setting up evaluation environment"))
        eval_env = setup_environment(cfg.eval)
    else:
        eval_env = None

    ema_policy = None
    if cfg.use_ema:
        ema_policy = common_utils.EMA(policy, power=3 / 4)

    common_utils.count_parameters(policy)
    if cfg.weight_decay == 0:
        print("Using Adam optimzer")
        optim = torch.optim.Adam(policy.parameters(), cfg.lr)
    else:
        print("Using AdamW optimzer")
        optim = torch.optim.AdamW(policy.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.cosine_schedule:
        lr_scheduler = diffusers.get_cosine_schedule_with_warmup(
            optim, cfg.lr_warm_up_steps, cfg.num_epoch * cfg.epoch_len
        )
    else:
        lr_scheduler = diffusers.get_constant_schedule(optim)

    saver = common_utils.TopkSaver(cfg.save_dir, 1)
    stopwatch = common_utils.Stopwatch()
    optim_step = 0
    best_success_rate = 0

    try:
        for epoch_num in tqdm(range(cfg.num_epoch)):
            stopwatch.reset()
            # Training loop
            for iter_num in tqdm(range(cfg.epoch_len)):
                with stopwatch.time("sample"):
                    if cfg.method == "dp":
                        batch = dataset.sample_dp(cfg.batch_size, cfg.dp.prediction_horizon, "cuda:0")
                    elif cfg.method == "act":
                        batch = dataset.sample_dp(cfg.batch_size, cfg.act.model.num_queries, "cuda:0")
                    
                    pairing_batch = dataset.sample_pairing(cfg.batch_size // 2, "cuda:0")
                
                with stopwatch.time("train"):
                    loss: torch.Tensor = policy.loss(batch)
                    if pairing_batch is not None and (hasattr(cfg.dp, 'use_aux_loss') and cfg.dp.use_aux_loss):
                        aux_loss = policy.aux_loss(pairing_batch)
                        total_loss = loss + (cfg.dp.aux_loss_weight * aux_loss)
                    else:
                        aux_loss = torch.tensor(0.0)
                        total_loss = loss
                    
                    optim.zero_grad()
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(  # type: ignore
                        policy.parameters(), max_norm=cfg.grad_clip
                    )
                    optim.step()
                    lr_scheduler.step()
                    torch.cuda.synchronize()

                    stat["train/lr(x1000)"].append(lr_scheduler.get_last_lr()[0] * 1000)
                    stat["train/loss"].append(loss.item())
                    if hasattr(cfg.dp, 'use_aux_loss') and cfg.dp.use_aux_loss:
                        stat["train/aux_loss"].append(aux_loss.item())
                    stat["train/total_loss"].append(total_loss.item())
                    stat["train/grad_norm"].append(grad_norm.item())

                    optim_step += 1
                    if ema_policy is not None:
                        decay = ema_policy.step(policy, optim_step=optim_step)
                        stat["train/decay"].append(decay)

            # Validation after each epoch
            if epoch_num % cfg.val_freq == 0:
                policy.eval()
                with torch.no_grad():
                    train_mse_loss, train_img_emb_norm = policy.mse_action_loss(batch)
                    stat["train/mse_loss"].append(train_mse_loss.item())
                    if hasattr(train_img_emb_norm, 'item'):
                        stat["train/img_emb_norm"].append(train_img_emb_norm.item())
                    
                    val_batch = dataset.sample_validation(cfg.batch_size, 
                                                     cfg.dp.prediction_horizon if cfg.method == "dp" else cfg.act.model.num_queries,
                                                     "cuda:0")
                    if val_batch is not None:
                        val_loss = policy.loss(val_batch)
                        stat["val/loss"].append(val_loss.item())

                        val_mse_loss, val_img_emb_norm = policy.mse_action_loss(val_batch)
                        stat["val/mse_loss"].append(val_mse_loss.item())
                        if hasattr(val_img_emb_norm, 'item'):
                            stat["val/img_emb_norm"].append(val_img_emb_norm.item())

                        val_pairing_batch = dataset.sample_validation_pairing(cfg.batch_size // 2, "cuda:0")
                        if val_pairing_batch is not None and hasattr(cfg.dp, 'use_aux_loss') and cfg.dp.use_aux_loss:
                            val_aux_loss = policy.aux_loss(val_pairing_batch)
                            stat["val/aux_loss"].append(val_aux_loss.item())
                policy.train()
                
            # Run simulation evaluation
            if eval_env is not None and (epoch_num % cfg.eval_freq == 0 or epoch_num == cfg.num_epoch - 1):
                policy_to_evaluate = ema_policy.stable_model if ema_policy else policy
                policy_to_evaluate.eval()
                
                # Save checkpoint for evaluation
                temp_ckpt_path = os.path.join(cfg.save_dir, f"temp_eval_ckpt_{epoch_num}.pt")
                torch.save(policy_to_evaluate.state_dict(), temp_ckpt_path)
                
                success_rate = run_evaluation(
                    eval_env,
                    policy_to_evaluate, 
                    dataset, 
                    cfg.eval, 
                    epoch_num, 
                    cfg.save_dir, 
                    stat,
                    seed=cfg.seed
                )
                
                # Clean up temporary checkpoint
                if os.path.exists(temp_ckpt_path):
                    os.remove(temp_ckpt_path)
                    
                # Track best model by success rate
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    print(f"New best success rate: {best_success_rate:.2f}%")
                    # Save best model by success rate
                    best_ckpt_path = os.path.join(cfg.save_dir, "best_success_rate.pt")
                    torch.save(policy_to_evaluate.state_dict(), best_ckpt_path)
                    with open(os.path.join(cfg.save_dir, "best_success_rate_info.txt"), "w") as f:
                        f.write(f"Epoch: {epoch_num}\n")
                        f.write(f"Success Rate: {success_rate:.2f}%\n")
                
                policy_to_evaluate.train()

            # Standard metrics and saving
            epoch_time = stopwatch.elapsed_time_since_reset
            stat["other/speed"].append(cfg.epoch_len / epoch_time)
            policy_to_save = ema_policy.stable_model if ema_policy else policy
            metric = -stat["train/loss"].mean()
            stat.summary(optim_step)
            stopwatch.summary()
            saver.save(policy_to_save.state_dict(), metric, save_latest=True, epoch_num=epoch_num)

    finally:
        # Make sure to clean up the environment
        if eval_env is not None:
            eval_env.close()

    # quit this way to avoid wandb hangs
    if bool(cfg.use_wb):
        wandb.finish()

def load_model(weight_file, device, *, verbose=True):
    run_folder = os.path.dirname(weight_file)
    cfg_path = os.path.join(run_folder, f"cfg.yaml")
    if verbose:
        print(common_utils.wrap_ruler("config of loaded agent"))
        with open(cfg_path, "r") as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    
    # Load dataset info from npz
    dataset_info = np.load(os.path.join(run_folder, "dataset_info.npz"))
    action_min = torch.from_numpy(dataset_info['action_min']).to(device)
    action_max = torch.from_numpy(dataset_info['action_max']).to(device)
    camera_views = dataset_info['camera_views'].tolist()
    image_size = int(dataset_info['image_size'])
    obs_shape = tuple(dataset_info['obs_shape'])
    prop_dim = int(dataset_info['prop_dim'])
    
    # Create minimal dataset class for observation processing
    class MinimalDataset:
        def __init__(self):
            self.camera_views = camera_views
            self.obs_shape = obs_shape
            self.action_dim = len(action_min)
            self.prop_dim = prop_dim
            
        def process_observation(self, obs):
            return ObsProcessor(self.camera_views, image_size).process(obs)
            
    dataset = MinimalDataset()

    policy = DiffusionPolicyWithAuxLoss(
        obs_horizon=1,
        obs_shape=dataset.obs_shape,
        prop_dim=dataset.prop_dim,
        action_dim=dataset.action_dim,
        camera_views=dataset.camera_views,
        cfg=cfg.dp,
    ).to(device)

    policy.load_state_dict(torch.load(weight_file))
    if cfg.norm_action:
        policy.init_action_normalizer(action_min, action_max)
    return policy, dataset, cfg


if __name__ == "__main__":
    import rich.traceback
    import pyrallis

    rich.traceback.install()
    torch.set_printoptions(linewidth=100)
    
    # Setup for reproducibility and performance
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    common_utils.set_all_seeds(cfg.seed)
    log_path = os.path.join(cfg.save_dir, "train.log")
    # sys.stdout = common_utils.TinyLogger(log_path, print_to_stdout=True)
    run(cfg)