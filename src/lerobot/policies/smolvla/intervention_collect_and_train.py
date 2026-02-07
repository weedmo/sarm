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
SmolVLA Intervention Collection + Post-Training Pipeline

DAgger-like system: human interrupts policy inference via keyboard ('i' key),
takes over with teleop leader arm, then post-trains on collected intervention data.

Usage:
    # Collection only
    python -m lerobot.policies.smolvla.intervention_collect_and_train \
        --policy-path /path/to/smolvla \
        --dataset-repo-id user/intervention_data \
        --single-task "Pick up the cup" \
        --robot-port /dev/ttyUSB0 --teleop-port /dev/ttyUSB1 \
        --num-episodes 5 --collect-only

    # Post-training only
    python -m lerobot.policies.smolvla.intervention_collect_and_train \
        --policy-path /path/to/smolvla \
        --dataset-repo-id user/intervention_data \
        --post-train-steps 5000 --enable-augmentation --train-only

    # Full pipeline (collect then post-train)
    python -m lerobot.policies.smolvla.intervention_collect_and_train \
        --policy-path /path/to/smolvla \
        --dataset-repo-id user/intervention_data \
        --single-task "Pick up the cup" \
        --robot-port /dev/ttyUSB0 --teleop-port /dev/ttyUSB1 \
        --num-episodes 5 --post-train-steps 5000 --enable-augmentation
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, cycle
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.post_train_smolvla import save_checkpoint_standalone
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.constants import ACTION, CHECKPOINTS_DIR, OBS_STR
from lerobot.utils.control_utils import is_headless, predict_action
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.train_utils import get_step_identifier, update_last_checkpoint
from lerobot.utils.utils import format_big_number, get_safe_torch_device, init_logging, log_say

# ---------------------------------------------------------------------------
# Keyboard listener with intervention mode
# ---------------------------------------------------------------------------


def init_intervention_keyboard_listener():
    """
    Initialize keyboard listener with intervention mode toggling.

    Extends the standard init_keyboard_listener() with an 'i' key toggle
    for switching between policy and teleop control.

    Returns:
        Tuple of (listener, events) where events includes "intervention_mode".
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "save_without_task": False,
        "intervention_mode": False,
    }

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.space:
                print("Space key pressed. Exiting loop and saving episode without task instruction...")
                events["save_without_task"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif hasattr(key, "char") and key.char == "i":
                events["intervention_mode"] = not events["intervention_mode"]
                mode_str = "TELEOP (human)" if events["intervention_mode"] else "POLICY (auto)"
                print(f"\n>>> Switched to {mode_str} mode <<<\n")
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


# ---------------------------------------------------------------------------
# Intervention record loop
# ---------------------------------------------------------------------------


@safe_stop_image_writer
def intervention_record_loop(
    robot,
    events: dict,
    fps: int,
    teleop,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    dataset: LeRobotDataset,
    control_time_s: float,
    single_task: str,
) -> dict:
    """
    Record loop supporting both policy and teleop, switchable per-frame.

    When intervention_mode=False: runs policy inference.
    When intervention_mode=True: reads teleop leader arm.
    On switch from teleop→policy: resets policy and preprocessor.

    Returns:
        Stats dict with intervention_frames, total_frames, intervention_ratio.
    """
    if dataset.fps != fps:
        raise ValueError(f"Dataset fps mismatch: {dataset.fps} != {fps}")

    # Reset policy state at episode start
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    device = get_safe_torch_device(policy.config.device)
    prev_intervention = events["intervention_mode"]
    total_frames = 0
    intervention_frames = 0

    timestamp = 0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        current_intervention = events["intervention_mode"]

        # On switch from teleop→policy: flush action queue
        if prev_intervention and not current_intervention:
            policy.reset()
            preprocessor.reset()
            print(">>> Flushed policy action queue after intervention <<<")

        prev_intervention = current_intervention

        # Get robot observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        if current_intervention:
            # TELEOP mode: read from leader arm
            act = teleop.get_action()
            act_processed = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed, obs))
            action_values = act_processed
            is_intervention = 1
            intervention_frames += 1
        else:
            # POLICY mode: run inference
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            act_processed = make_robot_action(action_tensor, dataset.features)
            robot_action_to_send = robot_action_processor((act_processed, obs))
            action_values = act_processed
            is_intervention = 0

        # Send action to robot
        robot.send_action(robot_action_to_send)

        # Build frame and save
        action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
        frame = {
            **observation_frame,
            **action_frame,
            "task": single_task,
            "is_intervention": np.array([is_intervention], dtype=np.int64),
        }
        dataset.add_frame(frame)

        total_frames += 1

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t

    intervention_ratio = intervention_frames / max(total_frames, 1)
    return {
        "intervention_frames": intervention_frames,
        "total_frames": total_frames,
        "intervention_ratio": intervention_ratio,
    }


# ---------------------------------------------------------------------------
# Collection phase
# ---------------------------------------------------------------------------


def run_collection_phase(args) -> str:
    """
    Run the data collection phase.

    Creates robot + teleop, loads policy, and records episodes with
    intervention support.

    Returns:
        The dataset repo_id.
    """
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

    logging.info("=== Collection Phase ===")

    # Parse camera config
    cameras = {}
    if args.robot_cameras:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        cam_dict = json.loads(args.robot_cameras)
        for name, cam_cfg in cam_dict.items():
            cameras[name] = OpenCVCameraConfig(**cam_cfg)

    # Create robot and teleop configs
    robot_cfg = SOFollowerRobotConfig(port=args.robot_port, cameras=cameras)
    teleop_cfg = SOLeaderTeleopConfig(port=args.teleop_port)

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    # Load policy
    logging.info(f"Loading policy from: {args.policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path

    # Build dataset features from robot
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    # Add is_intervention feature
    dataset_features["is_intervention"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["is_intervention"],
    }

    # Create or resume dataset
    dataset_root = Path(args.output_dir) / "dataset" if args.output_dir else None
    try:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            fps=args.fps,
            root=dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * max(len(cameras), 1),
        )
    except FileExistsError:
        logging.info("Dataset already exists, resuming...")
        dataset = LeRobotDataset(
            args.dataset_repo_id,
            root=dataset_root,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )

    # Load policy model
    rename_map = None
    if args.rename_map:
        rename_map = json.loads("".join(args.rename_map.split()))
        logging.info(f"Using rename_map: {rename_map}")

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)
    device = torch.device(args.device)

    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats if dataset.meta.stats else {},
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    if rename_map:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": rename_map}

    ds_stats = dataset.meta.stats if dataset.meta.stats else {}
    if rename_map:
        ds_stats = rename_stats(ds_stats, rename_map)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=ds_stats,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Connect hardware
    robot.connect()
    teleop.connect()

    listener, events = init_intervention_keyboard_listener()

    logging.info(
        f"Recording {args.num_episodes} episodes at {args.fps} fps, "
        f"{args.episode_time_s}s per episode"
    )
    logging.info("Press 'i' to toggle intervention mode, '->' to end episode, 'Esc' to stop")

    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < args.num_episodes and not events["stop_recording"]:
                events["exit_early"] = False
                events["intervention_mode"] = False  # Start each episode in policy mode

                log_say(f"Recording episode {dataset.num_episodes}", True)

                stats = intervention_record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=args.episode_time_s,
                    single_task=args.single_task,
                )

                logging.info(
                    f"Episode stats: {stats['total_frames']} frames, "
                    f"{stats['intervention_frames']} intervention "
                    f"({stats['intervention_ratio']:.1%})"
                )

                # Handle save_without_task
                episode_without_task = events.get("save_without_task", False)
                events["save_without_task"] = False

                if events["rerecord_episode"]:
                    log_say("Re-record episode", True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                if episode_without_task:
                    log_say("Saving episode without task instruction", True)
                    dataset.episode_buffer["task"] = [""] * len(dataset.episode_buffer["task"])

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", True, blocking=True)
        dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if args.push_to_hub:
            hub_id = args.hub_repo_id or args.dataset_repo_id
            logging.info(f"Pushing dataset to Hub: {hub_id}")
            dataset.push_to_hub()

    logging.info(f"Collection complete. Dataset: {args.dataset_repo_id}")
    return args.dataset_repo_id


# ---------------------------------------------------------------------------
# Post-training phase
# ---------------------------------------------------------------------------


def run_post_training_phase(args, dataset_repo_id: str) -> None:
    """
    Post-train SmolVLA on collected intervention data.

    Uses InterventionRABCWeights for per-sample loss weighting,
    with optional image augmentation.
    """
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from lerobot.utils.intervention_weights import InterventionRABCWeights

    logging.info("=== Post-Training Phase ===")

    # Load policy config and dataset
    logging.info(f"Loading SmolVLA config from: {args.policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path

    logging.info(f"Loading dataset: {dataset_repo_id}")
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    # Setup image transforms if requested
    image_transforms = None
    if args.enable_augmentation:
        image_transforms = ImageTransforms(ImageTransformsConfig(enable=True))
        logging.info("Image augmentation enabled")

    dataset = LeRobotDataset(
        dataset_repo_id,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
    )
    logging.info(
        f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames, {ds_meta.fps} fps"
    )

    # Parse rename_map
    rename_map = None
    if args.rename_map:
        rename_map = json.loads("".join(args.rename_map.split()))
        logging.info(f"Using rename_map: {rename_map}")

    # Load policy
    logging.info("Loading SmolVLA policy...")
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(
        f"Parameters: {format_big_number(num_learnable_params)} learnable / "
        f"{format_big_number(num_total_params)} total"
    )

    # Create preprocessor/postprocessor
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
        pretrained_path=args.policy_path,
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

    # Create optimizer and scheduler from SmolVLA presets
    logging.info("Creating optimizer and scheduler...")
    optimizer_cfg = policy_cfg.get_optimizer_preset()
    scheduler_cfg = policy_cfg.get_scheduler_preset()

    if args.lr is not None:
        optimizer_cfg.lr = args.lr
        scheduler_cfg.peak_lr = args.lr
    if args.warmup_steps is not None:
        scheduler_cfg.num_warmup_steps = args.warmup_steps

    scheduler_cfg.num_decay_steps = args.post_train_steps

    params = policy.get_optim_params()
    optimizer = optimizer_cfg.build(params)
    lr_scheduler = scheduler_cfg.build(optimizer, args.post_train_steps)

    logging.info(
        f"Optimizer: lr={optimizer_cfg.lr}, warmup={scheduler_cfg.num_warmup_steps}, "
        f"decay_steps={scheduler_cfg.num_decay_steps}, grad_clip={optimizer_cfg.grad_clip_norm}"
    )

    # Create InterventionRABCWeights
    logging.info("Setting up intervention weights...")
    intervention_weights = InterventionRABCWeights(
        dataset=dataset,
        intervention_weight=args.intervention_weight,
        policy_base_weight=args.policy_base_weight,
        device=device,
    )
    weight_stats = intervention_weights.get_stats()
    logging.info(
        f"Intervention weights: {weight_stats['total_frames']} frames, "
        f"{weight_stats['intervention_frames']} intervention "
        f"({weight_stats['intervention_ratio']:.1%}), "
        f"weights: intervention={args.intervention_weight}, policy={args.policy_base_weight}"
    )

    # Setup Accelerator + DataLoader
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
        output_dir = Path(f"outputs/intervention_post_train/{timestamp}")
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
                name=f"intervention_{dataset_repo_id.split('/')[-1]}",
            )
            logging.info(f"WandB initialized: {wandb_run.url}")
        except ImportError:
            logging.warning("wandb not installed, skipping WandB logging")

    # Seed
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Training loop
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
        f"Starting intervention post-training: {args.post_train_steps} steps, "
        f"batch_size={args.batch_size}"
    )

    for step in range(1, args.post_train_steps + 1):
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
            rabc_weights_provider=intervention_weights,
        )
        train_tracker.step()

        is_log_step = args.log_freq > 0 and step % args.log_freq == 0
        is_saving_step = step % args.save_freq == 0 or step == args.post_train_steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_run:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                w_stats = intervention_weights.get_stats()
                wandb_log_dict.update(
                    {
                        "intervention_ratio": w_stats["intervention_ratio"],
                        "intervention_weight": w_stats["intervention_weight"],
                        "policy_base_weight": w_stats["policy_base_weight"],
                    }
                )
                wandb_run.log(wandb_log_dict, step=step)
            train_tracker.reset_averages()

        if is_saving_step:
            logging.info(f"Saving checkpoint at step {step}")
            checkpoint_dir = output_dir / CHECKPOINTS_DIR / get_step_identifier(step, args.post_train_steps)
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

    logging.info("Post-training complete!")

    if args.push_to_hub and args.hub_repo_id:
        logging.info(f"Pushing model to Hub: {args.hub_repo_id}")
        unwrapped_policy = accelerator.unwrap_model(policy)
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA Intervention Collection + Post-Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Pretrained SmolVLA checkpoint (local path or HF repo ID)",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Dataset repo ID for saving/loading intervention data",
    )

    # Task
    parser.add_argument(
        "--single-task",
        type=str,
        default=None,
        help="Task description (required for collection)",
    )

    # Robot hardware
    hw_group = parser.add_argument_group("Robot hardware")
    hw_group.add_argument("--robot-port", type=str, default="/dev/ttyUSB0", help="Follower arm serial port")
    hw_group.add_argument("--teleop-port", type=str, default="/dev/ttyUSB1", help="Leader arm serial port")
    hw_group.add_argument(
        "--robot-cameras",
        type=str,
        default=None,
        help='Camera config JSON, e.g. \'{"top": {"index_or_path": 0, "width": 640, "height": 480, "fps": 30}}\'',
    )
    hw_group.add_argument("--fps", type=int, default=30, help="Recording FPS")

    # Collection
    collect_group = parser.add_argument_group("Collection")
    collect_group.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to collect")
    collect_group.add_argument("--episode-time-s", type=float, default=60, help="Seconds per episode")
    collect_group.add_argument("--collect-only", action="store_true", help="Only run collection phase")

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--post-train-steps", type=int, default=5000, help="Number of training steps")
    train_group.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    train_group.add_argument("--lr", type=float, default=None, help="Override SmolVLA default learning rate")
    train_group.add_argument("--warmup-steps", type=int, default=None, help="Override warmup steps")
    train_group.add_argument(
        "--enable-augmentation", action="store_true", help="Enable image augmentation during training"
    )
    train_group.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    train_group.add_argument("--save-freq", type=int, default=1000, help="Save checkpoint every N steps")
    train_group.add_argument("--log-freq", type=int, default=100, help="Log metrics every N steps")
    train_group.add_argument("--seed", type=int, default=1000, help="Random seed")
    train_group.add_argument("--train-only", action="store_true", help="Only run post-training phase")

    # Weights
    weight_group = parser.add_argument_group("Intervention weights")
    weight_group.add_argument(
        "--intervention-weight", type=float, default=1.0, help="Loss weight for human intervention frames"
    )
    weight_group.add_argument(
        "--policy-base-weight", type=float, default=0.3, help="Loss weight for policy autonomy frames"
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints")
    output_group.add_argument("--device", type=str, default="cuda", help="Device for training/inference")
    output_group.add_argument("--push-to-hub", action="store_true", help="Push to HF Hub")
    output_group.add_argument("--hub-repo-id", type=str, default=None, help="HF Hub repo ID for model upload")
    output_group.add_argument(
        "--rename-map",
        type=str,
        default=None,
        help="JSON dict mapping dataset keys to policy keys",
    )

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--wandb-project", type=str, default=None, help="WandB project name")

    args = parser.parse_args()
    init_logging()

    # Validate args
    if not args.train_only and args.single_task is None:
        parser.error("--single-task is required for collection (unless --train-only)")

    if args.collect_only and args.train_only:
        parser.error("Cannot use both --collect-only and --train-only")

    # Run phases
    dataset_repo_id = args.dataset_repo_id

    if not args.train_only:
        dataset_repo_id = run_collection_phase(args)

    if not args.collect_only:
        run_post_training_phase(args, dataset_repo_id)


if __name__ == "__main__":
    main()
