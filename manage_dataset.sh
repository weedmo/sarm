#!/bin/bash
# LeRobot Dataset Manager
# Split, Merge, Delete episodes from LeRobot datasets
#
# Usage:
#   ./manage_dataset.sh info       <repo_id>                          - Show dataset info
#   ./manage_dataset.sh delete     <repo_id> <episode_indices>        - Delete episodes
#   ./manage_dataset.sh split      <repo_id> <split_config>           - Split dataset
#   ./manage_dataset.sh merge      <output_repo_id> <repo_id1> <repo_id2> ...  - Merge datasets
#
# Options (set via environment variables):
#   ROOT=<path>           - Custom dataset root (default: ~/.cache/huggingface/lerobot)
#   NEW_REPO_ID=<id>      - Save result to new repo instead of overwriting
#   PUSH_TO_HUB=true      - Push result to Hugging Face Hub
#
# Examples:
#   # Show dataset info
#   ./manage_dataset.sh info weedmo/bimanual_so100_dataset_1
#
#   # Delete episodes 0, 3, 5
#   ./manage_dataset.sh delete weedmo/bimanual_so100_dataset_1 "0,3,5"
#
#   # Delete episode range 0-4
#   ./manage_dataset.sh delete weedmo/bimanual_so100_dataset_1 "0-4"
#
#   # Delete and save as new dataset
#   NEW_REPO_ID=weedmo/dataset_cleaned ./manage_dataset.sh delete weedmo/bimanual_so100_dataset_1 "0,3,5"
#
#   # Split 80/20 by fraction
#   ./manage_dataset.sh split weedmo/bimanual_so100_dataset_1 "train:0.8,val:0.2"
#
#   # Split by specific episode indices
#   ./manage_dataset.sh split weedmo/bimanual_so100_dataset_1 "train:0-7,val:8-9"
#
#   # Merge two datasets
#   ./manage_dataset.sh merge weedmo/merged_dataset weedmo/dataset_a weedmo/dataset_b

set -euo pipefail

# ─── Colors ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

DEFAULT_ROOT="$HOME/.cache/huggingface/lerobot"
ROOT="${ROOT:-$DEFAULT_ROOT}"

# ─── Helpers ───

print_header() {
    echo -e "\n${BOLD}${CYAN}=== LeRobot Dataset Manager ===${NC}\n"
}

print_usage() {
    echo -e "${BOLD}Usage:${NC}"
    echo "  $0 info     <repo_id>"
    echo "  $0 delete   <repo_id> <episodes>"
    echo "  $0 split    <repo_id> <split_config>"
    echo "  $0 merge    <output_repo_id> <repo_id1> <repo_id2> [...]"
    echo ""
    echo -e "${BOLD}Episode format:${NC}"
    echo "  Comma-separated: 0,3,5"
    echo "  Range:           0-4"
    echo "  Mixed:           0,3-5,8"
    echo ""
    echo -e "${BOLD}Split format:${NC}"
    echo "  By fraction:  train:0.8,val:0.2"
    echo "  By episodes:  train:0-7,val:8-9"
    echo ""
    echo -e "${BOLD}Environment variables:${NC}"
    echo "  ROOT=<path>        Custom dataset root dir"
    echo "  NEW_REPO_ID=<id>   Save to new repo (delete/split)"
    echo "  PUSH_TO_HUB=true   Push result to HF Hub"
}

die() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Parse episode indices string like "0,3-5,8" into Python list string "[0,3,4,5,8]"
parse_episodes() {
    local input="$1"
    local result=()

    IFS=',' read -ra parts <<< "$input"
    for part in "${parts[@]}"; do
        part=$(echo "$part" | tr -d ' ')
        if [[ "$part" == *-* ]]; then
            local start="${part%-*}"
            local end="${part#*-}"
            for ((i=start; i<=end; i++)); do
                result+=("$i")
            done
        else
            result+=("$part")
        fi
    done

    # Build Python list string
    local py_list="["
    for i in "${!result[@]}"; do
        if [ "$i" -gt 0 ]; then
            py_list+=", "
        fi
        py_list+="${result[$i]}"
    done
    py_list+="]"
    echo "$py_list"
}

# Parse split config like "train:0.8,val:0.2" or "train:0-7,val:8-9"
parse_splits() {
    local input="$1"
    local result="{"

    IFS=',' read -ra parts <<< "$input"
    local first=true
    for part in "${parts[@]}"; do
        part=$(echo "$part" | tr -d ' ')
        local name="${part%%:*}"
        local value="${part#*:}"

        if [ "$first" = true ]; then
            first=false
        else
            result+=", "
        fi

        # Check if value is a fraction (contains .)
        if [[ "$value" == *"."* ]]; then
            result+="\"$name\": $value"
        else
            # Parse as episode indices
            local episodes
            episodes=$(parse_episodes "$value")
            result+="\"$name\": $episodes"
        fi
    done

    result+="}"
    echo "$result"
}

# Get dataset path
get_dataset_path() {
    local repo_id="$1"
    echo "$ROOT/$repo_id"
}

# ─── Commands ───

cmd_info() {
    local repo_id="$1"
    local ds_path
    ds_path=$(get_dataset_path "$repo_id")

    if [ ! -d "$ds_path" ]; then
        die "Dataset not found: $ds_path"
    fi

    local info_file="$ds_path/meta/info.json"
    if [ ! -f "$info_file" ]; then
        die "info.json not found at $info_file"
    fi

    echo -e "${BOLD}Dataset:${NC} $repo_id"
    echo -e "${BOLD}Path:${NC}    $ds_path"
    echo ""

    python3 -c "
import json, sys
from pathlib import Path

info_path = Path('$info_file')
info = json.loads(info_path.read_text())

print(f'  Total episodes:  {info.get(\"total_episodes\", \"N/A\")}')
print(f'  Total frames:    {info.get(\"total_frames\", \"N/A\")}')
print(f'  FPS:             {info.get(\"fps\", \"N/A\")}')
print(f'  Robot type:      {info.get(\"robot_type\", \"N/A\")}')
print(f'  Total tasks:     {info.get(\"total_tasks\", \"N/A\")}')
print(f'  Codebase ver:    {info.get(\"codebase_version\", \"N/A\")}')
print()

features = info.get('features', {})
print('  Features:')
for name, feat in features.items():
    dtype = feat.get('dtype', '?')
    shape = feat.get('shape', '?')
    print(f'    {name}: dtype={dtype}, shape={shape}')

print()

# Show episode details
episodes_dir = Path('$ds_path') / 'meta' / 'episodes'
if episodes_dir.exists():
    import pandas as pd
    dfs = []
    for f in sorted(episodes_dir.rglob('*.parquet')):
        dfs.append(pd.read_parquet(f))
    if dfs:
        episodes_df = pd.concat(dfs, ignore_index=True)
        print('  Episodes:')
        print(f'    {\"idx\":<6} {\"length\":<10} {\"tasks\":<50}')
        print(f'    {\"-\"*6} {\"-\"*10} {\"-\"*50}')
        for _, row in episodes_df.iterrows():
            ep_idx = row.get('episode_index', '?')
            length = row.get('length', '?')
            tasks = row.get('tasks', [])
            if isinstance(tasks, list):
                tasks_str = ', '.join(str(t) for t in tasks)
            else:
                tasks_str = str(tasks)
            if len(tasks_str) > 48:
                tasks_str = tasks_str[:45] + '...'
            print(f'    {ep_idx:<6} {length:<10} {tasks_str:<50}')
"
}

cmd_delete() {
    local repo_id="$1"
    local episodes_str="$2"

    local ds_path
    ds_path=$(get_dataset_path "$repo_id")
    if [ ! -d "$ds_path" ]; then
        die "Dataset not found: $ds_path"
    fi

    local episodes_list
    episodes_list=$(parse_episodes "$episodes_str")

    info "Deleting episodes $episodes_list from $repo_id"

    # Build CLI args
    local args=(
        --repo_id "$repo_id"
        --root "$ROOT"
        --operation.type delete_episodes
        --operation.episode_indices "$episodes_list"
    )

    if [ -n "${NEW_REPO_ID:-}" ]; then
        args+=(--new_repo_id "$NEW_REPO_ID")
        info "Output: $NEW_REPO_ID"
    else
        warn "This will overwrite the original dataset (backup saved as ${repo_id}_old)"
    fi

    if [ "${PUSH_TO_HUB:-false}" = "true" ]; then
        args+=(--push_to_hub true)
    fi

    # Confirm
    echo -e "\n${YELLOW}Proceed? [y/N]${NC} "
    read -r confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    python -m lerobot.scripts.lerobot_edit_dataset "${args[@]}"

    echo -e "\n${GREEN}Done!${NC}"
}

cmd_split() {
    local repo_id="$1"
    local split_str="$2"

    local ds_path
    ds_path=$(get_dataset_path "$repo_id")
    if [ ! -d "$ds_path" ]; then
        die "Dataset not found: $ds_path"
    fi

    local splits_json
    splits_json=$(parse_splits "$split_str")

    info "Splitting $repo_id with config: $splits_json"

    local args=(
        --repo_id "$repo_id"
        --root "$ROOT"
        --operation.type split
        --operation.splits "$splits_json"
    )

    if [ "${PUSH_TO_HUB:-false}" = "true" ]; then
        args+=(--push_to_hub true)
    fi

    # Confirm
    echo -e "\n${YELLOW}Proceed? [y/N]${NC} "
    read -r confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    python -m lerobot.scripts.lerobot_edit_dataset "${args[@]}"

    echo -e "\n${GREEN}Done! Split datasets created:${NC}"
    IFS=',' read -ra parts <<< "$split_str"
    for part in "${parts[@]}"; do
        local name="${part%%:*}"
        name=$(echo "$name" | tr -d ' ')
        echo "  - ${repo_id}_${name}"
    done
}

cmd_merge() {
    local output_repo_id="$1"
    shift
    local repo_ids=("$@")

    if [ ${#repo_ids[@]} -lt 2 ]; then
        die "At least 2 datasets required for merge"
    fi

    # Verify all datasets exist
    for rid in "${repo_ids[@]}"; do
        local ds_path
        ds_path=$(get_dataset_path "$rid")
        if [ ! -d "$ds_path" ]; then
            die "Dataset not found: $ds_path ($rid)"
        fi
    done

    # Build repo_ids list string for CLI
    local repo_list="["
    local first=true
    for rid in "${repo_ids[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            repo_list+=", "
        fi
        repo_list+="'$rid'"
    done
    repo_list+="]"

    info "Merging ${#repo_ids[@]} datasets into $output_repo_id"
    for rid in "${repo_ids[@]}"; do
        echo "  - $rid"
    done

    local args=(
        --repo_id "$output_repo_id"
        --root "$ROOT"
        --operation.type merge
        --operation.repo_ids "$repo_list"
    )

    if [ "${PUSH_TO_HUB:-false}" = "true" ]; then
        args+=(--push_to_hub true)
    fi

    # Confirm
    echo -e "\n${YELLOW}Proceed? [y/N]${NC} "
    read -r confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    python -m lerobot.scripts.lerobot_edit_dataset "${args[@]}"

    echo -e "\n${GREEN}Done! Merged dataset: $output_repo_id${NC}"
}

# ─── Main ───

print_header

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
    info)
        [ $# -lt 1 ] && die "Usage: $0 info <repo_id>"
        cmd_info "$1"
        ;;
    delete)
        [ $# -lt 2 ] && die "Usage: $0 delete <repo_id> <episodes>\n  Example: $0 delete weedmo/dataset 0,3-5"
        cmd_delete "$1" "$2"
        ;;
    split)
        [ $# -lt 2 ] && die "Usage: $0 split <repo_id> <split_config>\n  Example: $0 split weedmo/dataset train:0.8,val:0.2"
        cmd_split "$1" "$2"
        ;;
    merge)
        [ $# -lt 3 ] && die "Usage: $0 merge <output_repo_id> <repo1> <repo2> [...]\n  Example: $0 merge weedmo/merged weedmo/ds1 weedmo/ds2"
        cmd_merge "$@"
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        die "Unknown command: $COMMAND\nRun '$0 --help' for usage"
        ;;
esac
