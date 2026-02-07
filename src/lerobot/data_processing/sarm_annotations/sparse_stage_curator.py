"""
SARM Sparse Stage Curation Tool

Interactive tool for manually annotating per-episode sparse stage boundaries
(e.g., "reach", "grasp", "lift") by watching episode data in Rerun and pressing
space to mark where each stage transition occurs.

Usage:
    python -m lerobot.data_processing.sarm_annotations.sparse_stage_curator \
        --repo-id user/dataset --root data \
        --stage-names reach grasp lift place \
        --episodes 0 1 2 3
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data

from lerobot.data_processing.sarm_annotations.subtask_annotation import (
    Subtask,
    SubtaskAnnotation,
    Timestamp,
    compute_temporal_proportions,
    load_annotations_from_dataset,
    save_annotations_to_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 torch tensor to HWC uint8 numpy array."""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    return (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# EpisodeState: tracks per-episode curation progress
# ---------------------------------------------------------------------------


@dataclass
class EpisodeState:
    """Tracks the curation state for a single episode."""

    total_frames: int
    num_transitions: int  # N-1 for N stages
    stage_names: list[str]
    cursor_frame: int = 0
    transition_frames: list[int] = field(default_factory=list)
    is_playing: bool = False

    def mark_transition(self) -> str | None:
        """Mark a transition at the current cursor frame.

        Returns None on success, or an error message string.
        """
        if len(self.transition_frames) >= self.num_transitions:
            return "All transitions already marked"
        if self.cursor_frame == 0:
            return "Cannot mark transition at frame 0"
        if self.cursor_frame >= self.total_frames - 1:
            return "Cannot mark transition at last frame"
        if self.transition_frames and self.cursor_frame <= self.transition_frames[-1]:
            return f"Must be after previous transition (frame {self.transition_frames[-1]})"
        self.transition_frames.append(self.cursor_frame)
        return None

    def undo_transition(self) -> bool:
        """Undo the last transition. Returns True if one was removed."""
        if self.transition_frames:
            self.transition_frames.pop()
            return True
        return False

    @property
    def is_complete(self) -> bool:
        return len(self.transition_frames) == self.num_transitions

    @property
    def next_transition_idx(self) -> int:
        return len(self.transition_frames)

    def step(self, delta: int):
        """Move cursor by delta frames, clamping to valid range."""
        self.cursor_frame = max(0, min(self.total_frames - 1, self.cursor_frame + delta))

    def to_subtask_annotation(self, fps: int) -> SubtaskAnnotation:
        """Convert transition frames to SubtaskAnnotation for saving."""
        subtasks = []
        boundaries = [0] + self.transition_frames + [self.total_frames]
        for i, name in enumerate(self.stage_names):
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            subtasks.append(
                Subtask(
                    name=name,
                    timestamps=Timestamp(
                        start=f"{int(start_sec) // 60:02d}:{int(start_sec) % 60:02d}",
                        end=f"{int(end_sec) // 60:02d}:{int(end_sec) % 60:02d}",
                    ),
                )
            )
        return SubtaskAnnotation(subtasks=subtasks)


# ---------------------------------------------------------------------------
# Keyboard listener (pynput, same pattern as control_utils.py)
# ---------------------------------------------------------------------------


def init_curation_keyboard_listener():
    """Initialize keyboard listener for curation controls.

    Returns (listener, events) where events is a shared dict polled by main loop.
    """
    from pynput import keyboard

    events = {
        "step_left": False,
        "step_right": False,
        "step_left_10": False,
        "step_right_10": False,
        "mark": False,
        "undo": False,
        "play_toggle": False,
        "save_next": False,
        "skip": False,
        "quit": False,
    }

    def on_press(key):
        try:
            # Arrow keys for stepping
            if key == keyboard.Key.left:
                events["step_left"] = True
            elif key == keyboard.Key.right:
                events["step_right"] = True
            elif key == keyboard.Key.up:
                events["step_left_10"] = True
            elif key == keyboard.Key.down:
                events["step_right_10"] = True
            elif key == keyboard.Key.space:
                events["mark"] = True
            elif hasattr(key, "char"):
                if key.char == "u":
                    events["undo"] = True
                elif key.char == "p":
                    events["play_toggle"] = True
                elif key.char == "s":
                    events["save_next"] = True
                elif key.char == "n":
                    events["skip"] = True
                elif key.char == "q":
                    events["quit"] = True
            elif key == keyboard.Key.backspace:
                events["undo"] = True
        except Exception as e:
            print(f"Key handler error: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def consume_event(events: dict, key: str) -> bool:
    """Check and consume a boolean flag atomically."""
    if events[key]:
        events[key] = False
        return True
    return False


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------


def prelog_episode_frames(
    dataset: LeRobotDataset,
    video_key: str | None,
):
    """Pre-log all frames of an episode to Rerun.

    The dataset should be loaded with a single episode (episodes=[ep_idx]),
    so local indices 0..len(dataset)-1 map to that episode's frames.
    """
    for idx in range(len(dataset)):
        item = dataset[idx]
        frame_idx = item["frame_index"].item()

        rr.set_time("frame_index", sequence=frame_idx)
        rr.set_time("timestamp", timestamp=item["timestamp"].item())

        # Log camera images
        keys_to_log = [video_key] if video_key else dataset.meta.camera_keys
        for key in keys_to_log:
            if key in item:
                img = to_hwc_uint8_numpy(item[key])
                rr.log(key, rr.Image(img).compress())

        # Log action dimensions
        if ACTION in item and item[ACTION].ndim > 0:
            for dim_idx, val in enumerate(item[ACTION]):
                rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

        # Log state dimensions
        if OBS_STATE in item and item[OBS_STATE].ndim > 0:
            for dim_idx, val in enumerate(item[OBS_STATE]):
                rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))


def log_cursor(frame_idx: int):
    """Log the cursor position as a scalar for Rerun timeline tracking."""
    rr.set_time("frame_index", sequence=frame_idx)
    rr.log("curation/cursor", rr.Scalars(1.0))


def log_boundary_marker(frame_idx: int, stage_from: str, stage_to: str, transition_idx: int):
    """Log a stage boundary marker at the given frame."""
    rr.set_time("frame_index", sequence=frame_idx)
    label = f"[{transition_idx + 1}] {stage_from} -> {stage_to}"
    rr.log(f"curation/boundary/{stage_from}_to_{stage_to}", rr.TextDocument(label))
    # Also log a visual marker as a scalar spike
    rr.log("curation/boundaries", rr.Scalars(1.0))


def clear_boundary_marker(frame_idx: int, stage_from: str, stage_to: str):
    """Clear a boundary marker from Rerun."""
    rr.set_time("frame_index", sequence=frame_idx)
    rr.log(f"curation/boundary/{stage_from}_to_{stage_to}", rr.Clear(recursive=False))
    rr.log("curation/boundaries", rr.Clear(recursive=False))


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------


def print_status(state: EpisodeState, ep_idx: int, total_episodes: int, fps: int):
    """Print the current curation status to terminal."""
    time_sec = state.cursor_frame / fps
    play_str = "PLAYING" if state.is_playing else "PAUSED"

    lines = []
    lines.append("\033[2J\033[H")  # Clear screen
    lines.append("=== SARM Sparse Stage Curation ===")
    lines.append("")
    lines.append(f"--- Episode {ep_idx + 1}/{total_episodes} (ep_idx={ep_idx}) ---")
    lines.append(
        f"Frame: {state.cursor_frame}/{state.total_frames - 1} | " f"Time: {time_sec:.2f}s | {play_str}"
    )
    lines.append("")
    lines.append(f"Transitions ({state.num_transitions} needed):")

    for i in range(state.num_transitions):
        from_name = state.stage_names[i]
        to_name = state.stage_names[i + 1]
        if i < len(state.transition_frames):
            f = state.transition_frames[i]
            t = f / fps
            lines.append(f"  [{i + 1}] {from_name} -> {to_name}:  frame {f}  ({t:.2f}s)  OK")
        elif i == len(state.transition_frames):
            lines.append(f"  [{i + 1}] {from_name} -> {to_name}:  << press SPACE >>")
        else:
            lines.append(f"  [{i + 1}] {from_name} -> {to_name}:  --")

    lines.append("")
    if state.is_complete:
        lines.append("All transitions marked! Press 's' to save or 'u' to undo.")
    lines.append(
        "Controls: left/right=step(+up/down=x10) | p=play | "
        "space=mark | u=undo | s=save+next | n=skip | q=quit"
    )

    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Main curation loop for a single episode
# ---------------------------------------------------------------------------


def curate_episode(
    dataset: LeRobotDataset,
    ep_idx: int,
    stage_names: list[str],
    events: dict,
    fps: int,
    play_fps_divisor: int,
    video_key: str | None,
    ep_number: int,
    total_episodes: int,
) -> SubtaskAnnotation | str | None:
    """Run the interactive curation loop for one episode.

    Returns SubtaskAnnotation on save, None on skip, or "QUIT" sentinel string.
    The dataset should be loaded with episodes=[ep_idx].
    """
    total_frames = len(dataset)

    num_transitions = len(stage_names) - 1
    state = EpisodeState(
        total_frames=total_frames,
        num_transitions=num_transitions,
        stage_names=stage_names,
    )

    # Pre-log all frames to Rerun
    print(f"Loading episode {ep_idx} ({total_frames} frames)...", flush=True)
    prelog_episode_frames(dataset, video_key)
    print("Episode loaded. Starting curation.", flush=True)

    play_interval = play_fps_divisor / fps  # seconds between play steps
    last_play_time = time.time()
    last_print_time = 0.0

    while True:
        now = time.time()

        # --- Process events ---
        if consume_event(events, "quit"):
            return "QUIT"  # Special sentinel

        if consume_event(events, "skip"):
            return None

        if consume_event(events, "save_next"):
            if state.is_complete:
                return state.to_subtask_annotation(fps)
            else:
                # Flash a message but don't exit
                print(
                    f"\n  ** Cannot save: {state.num_transitions - len(state.transition_frames)} "
                    f"transitions still needed **",
                    flush=True,
                )
                time.sleep(0.5)

        if consume_event(events, "mark"):
            err = state.mark_transition()
            if err:
                print(f"\n  ** {err} **", flush=True)
                time.sleep(0.3)
            else:
                # Log boundary marker to Rerun
                idx = len(state.transition_frames) - 1
                log_boundary_marker(
                    state.transition_frames[idx],
                    stage_names[idx],
                    stage_names[idx + 1],
                    idx,
                )

        if consume_event(events, "undo"):  # noqa: SIM102
            if state.transition_frames:
                removed_idx = len(state.transition_frames) - 1
                removed_frame = state.transition_frames[removed_idx]
                clear_boundary_marker(
                    removed_frame,
                    stage_names[removed_idx],
                    stage_names[removed_idx + 1],
                )
                state.undo_transition()

        if consume_event(events, "play_toggle"):
            state.is_playing = not state.is_playing
            last_play_time = now

        if consume_event(events, "step_left"):
            state.is_playing = False
            state.step(-1)
            log_cursor(state.cursor_frame)

        if consume_event(events, "step_right"):
            state.is_playing = False
            state.step(1)
            log_cursor(state.cursor_frame)

        if consume_event(events, "step_left_10"):
            state.is_playing = False
            state.step(-10)
            log_cursor(state.cursor_frame)

        if consume_event(events, "step_right_10"):
            state.is_playing = False
            state.step(10)
            log_cursor(state.cursor_frame)

        # --- Playback ---
        if state.is_playing and (now - last_play_time) >= play_interval:
            if state.cursor_frame < state.total_frames - 1:
                state.step(1)
                log_cursor(state.cursor_frame)
            else:
                state.is_playing = False
            last_play_time = now

        # --- Terminal display (throttled to ~10Hz) ---
        if now - last_print_time >= 0.1:
            print_status(state, ep_idx, total_episodes, fps)
            last_print_time = now

        # Sleep to avoid busy-waiting (~60Hz poll rate)
        time.sleep(1 / 60)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="SARM Sparse Stage Curation Tool - interactively annotate stage boundaries"
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo ID (e.g. user/dataset)")
    parser.add_argument("--root", type=str, default=None, help="Local dataset root directory")
    parser.add_argument(
        "--stage-names", nargs="+", required=True, help="Ordered stage names (e.g. reach grasp lift place)"
    )
    parser.add_argument("--episodes", nargs="*", type=int, default=None, help="Episode indices to annotate")
    parser.add_argument("--video-key", type=str, default=None, help="Camera key (default: first available)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already-annotated episodes")
    parser.add_argument(
        "--play-fps-divisor", type=int, default=2, help="Slow playback by this factor (default: 2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.stage_names) < 2:
        print("Error: need at least 2 stage names to define transitions.")
        sys.exit(1)

    # Load dataset
    root = Path(args.root) if args.root else None
    print(f"Loading dataset: {args.repo_id}", flush=True)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
    )
    fps = dataset.fps
    dataset_path = dataset.root

    # Determine video key
    video_key = args.video_key
    if video_key is None and dataset.meta.camera_keys:
        video_key = dataset.meta.camera_keys[0]

    # Determine episodes to curate
    total_eps = dataset.meta.total_episodes
    if args.episodes is not None:
        episode_indices = [e for e in args.episodes if 0 <= e < total_eps]
    else:
        episode_indices = list(range(total_eps))

    # Load existing annotations for skip-existing
    existing_annotations = {}
    if args.skip_existing:
        existing_annotations = load_annotations_from_dataset(dataset_path, prefix="sparse")
        before_count = len(episode_indices)
        episode_indices = [e for e in episode_indices if e not in existing_annotations]
        skipped = before_count - len(episode_indices)
        if skipped:
            print(f"Skipping {skipped} already-annotated episodes.")

    if not episode_indices:
        print("No episodes to annotate.")
        sys.exit(0)

    print("\n=== SARM Sparse Stage Curation ===")
    print(f"Dataset: {args.repo_id} | FPS: {fps}")
    print(f"Stages: {', '.join(args.stage_names)}")
    print(f"Episodes to annotate: {len(episode_indices)}")
    print(f"Transitions per episode: {len(args.stage_names) - 1}")
    print()

    # Start keyboard listener
    listener, events = init_curation_keyboard_listener()

    # Initialize Rerun
    rr.init(f"{args.repo_id}/sparse_curation", spawn=True)
    gc.collect()

    # Curate episodes
    all_annotations: dict[int, SubtaskAnnotation] = {}
    # Include any existing annotations we loaded (for proportions computation)
    all_annotations.update(existing_annotations)

    quit_requested = False
    for i, ep_idx in enumerate(episode_indices):
        if quit_requested:
            break

        # Load dataset for this specific episode
        ep_dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=root,
            episodes=[ep_idx],
        )

        result = curate_episode(
            dataset=ep_dataset,
            ep_idx=ep_idx,
            stage_names=args.stage_names,
            events=events,
            fps=fps,
            play_fps_divisor=args.play_fps_divisor,
            video_key=video_key,
            ep_number=i,
            total_episodes=len(episode_indices),
        )

        if result == "QUIT":
            quit_requested = True
            print("\nQuit requested. Saving completed annotations...")
        elif result is not None:
            all_annotations[ep_idx] = result
            print(f"\nSaved episode {ep_idx}.")
        else:
            print(f"\nSkipped episode {ep_idx}.")

    # Stop keyboard listener
    if listener is not None:
        listener.stop()

    # Save all annotations
    new_annotations = {k: v for k, v in all_annotations.items() if k in episode_indices}
    if new_annotations:
        print(f"\nSaving {len(new_annotations)} annotations to dataset...")
        save_annotations_to_dataset(dataset_path, all_annotations, fps, prefix="sparse")

        # Compute and save temporal proportions
        props = compute_temporal_proportions(all_annotations, fps, args.stage_names)
        props_path = dataset_path / "meta" / "temporal_proportions_sparse.json"
        props_path.parent.mkdir(parents=True, exist_ok=True)
        with open(props_path, "w") as f:
            json.dump(props, f, indent=2)
        print(f"Saved temporal proportions to {props_path}")
        print(f"Proportions: {props}")
    else:
        print("\nNo new annotations to save.")

    # Summary
    print("\n=== Summary ===")
    print(f"Annotated: {len(new_annotations)} episodes")
    print(f"Total annotated (including existing): {len(all_annotations)} episodes")
    print("Done.")


if __name__ == "__main__":
    main()
