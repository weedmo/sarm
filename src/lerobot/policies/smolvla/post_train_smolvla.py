#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA RA-BC Post-Training Script

Post-trains a pretrained SmolVLA policy using RA-BC (Reward-Aligned Behavior Cloning)
with SARM progress values. Standalone argparse script that reuses LeRobot infrastructure.

Usage:
    # With pre-computed SARM progress
    python -m lerobot.policies.smolvla.post_train_smolvla \
        --smolvla-path lerobot/smolvla_base \
        --dataset-repo-id my_user/my_dataset \
        --rabc-progress-path ./sarm_progress.parquet \
        --steps 30000 --batch-size 64

    # Auto-compute progress from SARM model
    python -m lerobot.policies.smolvla.post_train_smolvla \
        --smolvla-path lerobot/smolvla_base \
        --dataset-repo-id my_user/my_dataset \
        --sarm-model-path pepijn223/sarm_single_uni4 \
        --steps 30000 --batch-size 64

    # Generate dummy progress for testing (no SARM needed)
    python -m lerobot.policies.smolvla.post_train_smolvla \
        --smolvla-path lerobot/smolvla_base \
        --dataset-repo-id weedmo/bimanual_merged \
        --generate-dummy-progress \
        --steps 10 --batch-size 2 --log-freq 5 --save-freq 10
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    PRETRAINED_MODEL_DIR,
)
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_identifier,
    save_training_state,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging


def generate_dummy_progress(dataset: LeRobotDataset, output_path: Path) -> Path:
    """Generate dummy linear progress (0→1) for testing without a SARM model.

    Creates a parquet file matching the format of compute_sarm_progress output.

    Args:
        dataset: The loaded LeRobotDataset.
        output_path: Where to save the parquet file.

    Returns:
        Path to the saved parquet file.
    """
    logging.info("Generating dummy linear progress for testing...")

    records = []
    episodes = dataset.meta.episodes
    for ep_idx in range(dataset.num_episodes):
        from_idx = int(episodes["dataset_from_index"][ep_idx])
        to_idx = int(episodes["dataset_to_index"][ep_idx])
        ep_len = to_idx - from_idx

        for i, global_idx in enumerate(range(from_idx, to_idx)):
            progress = i / max(ep_len - 1, 1)
            records.append(
                {
                    "index": global_idx,
                    "episode_index": ep_idx,
                    "frame_index": i,
                    "progress_sparse": progress,
                }
            )

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(f"Saved dummy progress ({len(df)} frames) to {output_path}")
    return output_path


def save_checkpoint_standalone(
    checkpoint_dir: Path,
    step: int,
    policy,
    optimizer,
    scheduler=None,
    preprocessor=None,
    postprocessor=None,
):
    """Save a checkpoint without requiring TrainPipelineConfig.

    Lightweight wrapper that saves model + training state
    in the standard LeRobot checkpoint format.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)

    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def post_train_smolvla(args):
    """Main RA-BC post-training function."""
    init_logging()

    # Step 1: Resolve RA-BC progress path
    progress_path = None
    if args.rabc_progress_path:
        progress_path = args.rabc_progress_path
        logging.info(f"Using pre-computed SARM progress: {progress_path}")
    elif args.sarm_model_path:
        logging.info(f"Computing SARM progress from model: {args.sarm_model_path}")
        from lerobot.policies.sarm.compute_rabc_weights import compute_sarm_progress

        compute_sarm_progress(
            dataset_repo_id=args.dataset_repo_id,
            reward_model_path=args.sarm_model_path,
            head_mode=args.sarm_head_mode,
            device=args.device,
            num_visualizations=0,
            stride=args.sarm_stride,
        )
        # compute_sarm_progress saves to dataset cache dir by default
        ds_meta_tmp = LeRobotDatasetMetadata(args.dataset_repo_id)
        progress_path = str(ds_meta_tmp.root / "sarm_progress.parquet")
        logging.info(f"SARM progress saved to: {progress_path}")
    elif not args.generate_dummy_progress:
        raise ValueError(
            "Must provide one of: --rabc-progress-path, --sarm-model-path, or --generate-dummy-progress"
        )

    # Step 2: Load SmolVLA config and dataset
    logging.info(f"Loading SmolVLA config from: {args.smolvla_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.smolvla_path)
    policy_cfg.pretrained_path = args.smolvla_path

    logging.info(f"Loading dataset: {args.dataset_repo_id}")
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        delta_timestamps=delta_timestamps,
    )
    logging.info(
        f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames, {ds_meta.fps} fps"
    )

    # Generate dummy progress if requested (after dataset is loaded)
    if args.generate_dummy_progress and progress_path is None:
        dummy_path = dataset.root / "sarm_progress.parquet"
        generate_dummy_progress(dataset, dummy_path)
        progress_path = str(dummy_path)

    # Parse rename_map if provided
    rename_map = None
    if args.rename_map:
        import json as _json

        rename_map = _json.loads(args.rename_map)
        logging.info(f"Using rename_map: {rename_map}")

    # Step 3: Load SmolVLA policy
    logging.info("Loading SmolVLA policy...")
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(f"Parameters: {format_big_number(num_learnable_params)} learnable / {format_big_number(num_total_params)} total")

    # Step 4: Create preprocessor/postprocessor
    logging.info("Creating preprocessor/postprocessor...")
    device = torch.device(args.device)

    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    if rename_map:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": rename_map}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.smolvla_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    # Step 5: Create optimizer and scheduler from SmolVLA presets
    logging.info("Creating optimizer and scheduler...")
    optimizer_cfg = policy_cfg.get_optimizer_preset()
    scheduler_cfg = policy_cfg.get_scheduler_preset()

    # Apply CLI overrides
    if args.lr is not None:
        optimizer_cfg.lr = args.lr
        scheduler_cfg.peak_lr = args.lr
    if args.warmup_steps is not None:
        scheduler_cfg.num_warmup_steps = args.warmup_steps
    if args.grad_clip_norm is not None:
        optimizer_cfg.grad_clip_norm = args.grad_clip_norm

    scheduler_cfg.num_decay_steps = args.steps

    params = policy.get_optim_params()
    optimizer = optimizer_cfg.build(params)
    lr_scheduler = scheduler_cfg.build(optimizer, args.steps)

    logging.info(
        f"Optimizer: lr={optimizer_cfg.lr}, warmup={scheduler_cfg.num_warmup_steps}, "
        f"decay_steps={scheduler_cfg.num_decay_steps}, grad_clip={optimizer_cfg.grad_clip_norm}"
    )

    # Step 6: Create RABCWeights
    logging.info("Setting up RA-BC weights...")
    from lerobot.utils.rabc import RABCWeights

    rabc_weights = RABCWeights(
        progress_path=progress_path,
        chunk_size=policy_cfg.chunk_size,
        head_mode=args.sarm_head_mode,
        kappa=args.rabc_kappa,
        device=device,
    )
    rabc_stats = rabc_weights.get_stats()
    logging.info(
        f"RA-BC: {rabc_stats['num_frames']} frames, "
        f"delta_mean={rabc_stats['delta_mean']:.4f}, delta_std={rabc_stats['delta_std']:.4f}, "
        f"kappa={rabc_stats['kappa']}"
    )

    # Step 7: Setup Accelerator + DataLoader
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=(args.device == "cpu"),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/post_train_smolvla/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    # WandB
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"rabc_{args.dataset_repo_id.split('/')[-1]}",
            )
            logging.info(f"WandB initialized: {wandb_run.url}")
        except ImportError:
            logging.warning("wandb not installed, skipping WandB logging")

    # Seed
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Step 8: Training loop
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        args.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=0,
        accelerator=accelerator,
    )

    logging.info(
        f"Starting RA-BC post-training: {args.steps} steps, batch_size={args.batch_size}, "
        f"chunk_size={policy_cfg.chunk_size}"
    )

    step = 0
    for _ in range(args.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            optimizer_cfg.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        step += 1
        train_tracker.step()

        is_log_step = args.log_freq > 0 and step % args.log_freq == 0
        is_saving_step = step % args.save_freq == 0 or step == args.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_run:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                rabc_s = rabc_weights.get_stats()
                wandb_log_dict.update(
                    {
                        "rabc_delta_mean": rabc_s["delta_mean"],
                        "rabc_delta_std": rabc_s["delta_std"],
                    }
                )
                wandb_run.log(wandb_log_dict, step=step)
            train_tracker.reset_averages()

        if is_saving_step:
            logging.info(f"Saving checkpoint at step {step}")
            checkpoint_dir = output_dir / CHECKPOINTS_DIR / get_step_identifier(step, args.steps)
            unwrapped_policy = accelerator.unwrap_model(policy)
            save_checkpoint_standalone(
                checkpoint_dir=checkpoint_dir,
                step=step,
                policy=unwrapped_policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)

    # Step 9: Final save + Hub upload
    logging.info("Post-training complete!")

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub-repo-id is required when --push-to-hub is set")
        logging.info(f"Pushing to Hub: {args.hub_repo_id}")
        unwrapped_policy = accelerator.unwrap_model(policy)
        # Save final model and push
        final_dir = output_dir / "final_model"
        unwrapped_policy.save_pretrained(final_dir)
        preprocessor.save_pretrained(final_dir)
        postprocessor.save_pretrained(final_dir)
        unwrapped_policy.push_to_hub(args.hub_repo_id)
        preprocessor.push_to_hub(args.hub_repo_id)
        postprocessor.push_to_hub(args.hub_repo_id)
        logging.info(f"Pushed to: https://huggingface.co/{args.hub_repo_id}")

    if wandb_run:
        wandb_run.finish()

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA RA-BC Post-Training: Fine-tune SmolVLA with SARM reward-aligned weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--smolvla-path",
        type=str,
        required=True,
        help="Pretrained SmolVLA checkpoint (local path or HF repo ID)",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Dataset path (local or HF repo ID)",
    )

    parser.add_argument(
        "--rename-map",
        type=str,
        default=None,
        help='JSON dict mapping dataset keys to policy keys, e.g. \'{"observation.images.left_wrist": "observation.images.camera1"}\'',
    )

    # RA-BC source
    rabc_group = parser.add_argument_group("RA-BC source (provide one)")
    rabc_group.add_argument(
        "--rabc-progress-path",
        type=str,
        default=None,
        help="Path to pre-computed sarm_progress.parquet (supports hf:// URLs)",
    )
    rabc_group.add_argument(
        "--sarm-model-path",
        type=str,
        default=None,
        help="SARM model path for auto-computing progress",
    )
    rabc_group.add_argument(
        "--generate-dummy-progress",
        action="store_true",
        help="Generate dummy linear progress (0->1) for testing without SARM",
    )

    # RA-BC params
    rabc_params = parser.add_argument_group("RA-BC parameters")
    rabc_params.add_argument("--sarm-head-mode", type=str, default="sparse", choices=["sparse", "dense"])
    rabc_params.add_argument("--rabc-kappa", type=float, default=0.01, help="High-quality sample threshold")
    rabc_params.add_argument("--sarm-stride", type=int, default=1, help="SARM progress computation stride")

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch-size", type=int, default=64)
    train_group.add_argument("--steps", type=int, default=30000)
    train_group.add_argument("--lr", type=float, default=None, help="Override SmolVLA default (1e-4)")
    train_group.add_argument("--warmup-steps", type=int, default=None, help="Override SmolVLA default (1000)")
    train_group.add_argument("--grad-clip-norm", type=float, default=None, help="Override SmolVLA default (10)")
    train_group.add_argument("--num-workers", type=int, default=4)
    train_group.add_argument("--save-freq", type=int, default=5000)
    train_group.add_argument("--log-freq", type=int, default=200)
    train_group.add_argument("--seed", type=int, default=1000)
    train_group.add_argument("--device", type=str, default="cuda")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", type=str, default=None, help="Checkpoint save path (auto-generated if not set)")
    output_group.add_argument("--push-to-hub", action="store_true", help="Push final model to HF Hub")
    output_group.add_argument("--hub-repo-id", type=str, default=None, help="HF Hub repo ID for upload")

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--wandb-project", type=str, default=None, help="WandB project name (disabled if not set)")

    args = parser.parse_args()
    post_train_smolvla(args)


if __name__ == "__main__":
    main()
