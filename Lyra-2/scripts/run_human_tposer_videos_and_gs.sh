#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Usage:
#   bash scripts/run_human_tposer_videos_and_gs.sh
#   SKIP_GS=1 bash scripts/run_human_tposer_videos_and_gs.sh
#   GPU_IDS=0 bash scripts/run_human_tposer_videos_and_gs.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EXPERIMENT:-lyra2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/human_tposer}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
USE_DMD="${USE_DMD:-0}"
FORCE_GS="${FORCE_GS:-0}"
FORCE_VIDEO="${FORCE_VIDEO:-0}"
SKIP_GS="${SKIP_GS:-0}"
GPU_MAX_USED_MEM_MB="${GPU_MAX_USED_MEM_MB:-1024}"
DISABLE_GPU_BUSY_CHECK="${DISABLE_GPU_BUSY_CHECK:-0}"
FPS="${FPS:-16}"
RESOLUTION="${RESOLUTION:-480,832}"
ZOOM_IN_FRAMES="${ZOOM_IN_FRAMES:-81}"
ZOOM_OUT_FRAMES="${ZOOM_OUT_FRAMES:-241}"
ZOOM_IN_STRENGTH="${ZOOM_IN_STRENGTH:-0.5}"
ZOOM_OUT_STRENGTH="${ZOOM_OUT_STRENGTH:-1.5}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH=".:${PYTHONPATH:-}"
if [[ -d checkpoints/torch ]]; then export TORCH_HOME="${TORCH_HOME:-$PWD/checkpoints/torch}"; fi
if [[ -d checkpoints/huggingface ]]; then
  export HF_HOME="${HF_HOME:-$PWD/checkpoints/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PWD/checkpoints/huggingface/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/checkpoints/huggingface/hub}"
fi
if [[ -f checkpoints/vipe/droid.pth ]]; then export VIPE_DROID_CKPT="${VIPE_DROID_CKPT:-$PWD/checkpoints/vipe/droid.pth}"; fi

read -r -a VIDEO_EXTRA_ARGS_ARRAY <<< "${VIDEO_EXTRA_ARGS:-}"
read -r -a GS_EXTRA_ARGS_ARRAY <<< "${GS_EXTRA_ARGS:-}"
DMD_ARGS=(); [[ "$USE_DMD" == "1" ]] && DMD_ARGS=(--use_dmd)
GS_FORCE_ARGS=(); [[ "$FORCE_GS" == "1" ]] && GS_FORCE_ARGS=(--force)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
require_path() { [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }; }

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [[ "$PYTHON_BIN" == "python" ]] && command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; else echo "$PYTHON_BIN not found" >&2; exit 1; fi
fi

if command -v nvidia-smi >/dev/null 2>&1 && [[ "$DISABLE_GPU_BUSY_CHECK" != "1" ]]; then
  if [[ -n "${GPU_IDS:-}" ]]; then candidates="$(tr ', ' '\n' <<< "$GPU_IDS" | awk 'NF')"; elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then candidates="$(tr ', ' '\n' <<< "$CUDA_VISIBLE_DEVICES" | awk 'NF')"; else candidates="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true)"; fi
  selected=""
  while IFS= read -r gpu; do
    [[ -z "$gpu" ]] && continue
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | awk 'NR==1 {gsub(/ /, "", $1); print $1}')"
    if [[ "$used" =~ ^[0-9]+$ && "$used" -le "$GPU_MAX_USED_MEM_MB" ]]; then selected="$gpu"; break; fi
  done <<< "$candidates"
  [[ -n "$selected" ]] || { echo "No available GPU below ${GPU_MAX_USED_MEM_MB} MiB used." >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="$selected"
  log "Using GPU $selected"
fi

require_path assets/tposer.png
require_path assets/tposer.txt
require_path "$CHECKPOINT_DIR"
require_path checkpoints/text_encoder/negative_prompt.pt
[[ "$USE_DMD" == "1" ]] && require_path checkpoints/lora/dmd_distillation.safetensors
[[ "$SKIP_GS" != "1" ]] && require_path checkpoints/recon/model.pt

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
video="$OUTPUT_ROOT/zoomgs/videos/tposer.mp4"
if [[ "$FORCE_VIDEO" == "1" ]]; then rm -f "$video"; fi
if [[ ! -f "$video" ]]; then
  log "Human T-pose video generation starting"
  "$PYTHON_BIN" -m lyra_2._src.inference.lyra2_zoomgs_inference \
    --input_image_path assets/tposer.png \
    --prompt "$(tr '\n' ' ' < assets/tposer.txt)" \
    --experiment "$EXPERIMENT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_path "$OUTPUT_ROOT/zoomgs" \
    --fps "$FPS" \
    --resolution "$RESOLUTION" \
    --num_frames_zoom_in "$ZOOM_IN_FRAMES" \
    --num_frames_zoom_out "$ZOOM_OUT_FRAMES" \
    --zoom_in_strength "$ZOOM_IN_STRENGTH" \
    --zoom_out_strength "$ZOOM_OUT_STRENGTH" \
    "${DMD_ARGS[@]}" \
    "${VIDEO_EXTRA_ARGS_ARRAY[@]}" 2>&1 | tee -a "$LOG_DIR/video.log"
else
  log "Video exists, skipping generation: $video"
fi

if [[ "$SKIP_GS" == "1" ]]; then
  log "Skipping GS because SKIP_GS=1"
else
  log "Human T-pose GS reconstruction starting"
  "$PYTHON_BIN" -m lyra_2._src.inference.vipe_da3_gs_recon \
    --input_video_path "$video" \
    --output_dir "$OUTPUT_ROOT/gs" \
    "${GS_FORCE_ARGS[@]}" \
    "${GS_EXTRA_ARGS_ARRAY[@]}" 2>&1 | tee -a "$LOG_DIR/gs.log"
fi

echo "Done. Outputs are under: $OUTPUT_ROOT"
