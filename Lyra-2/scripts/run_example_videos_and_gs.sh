#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Usage:
#   bash scripts/run_example_videos_and_gs.sh
#   SKIP_GS=1 bash scripts/run_example_videos_and_gs.sh
#   RUN_CUSTOM_TRAJ=1 RUN_SAMPLES=1 bash scripts/run_example_videos_and_gs.sh
#   GPU_IDS="0 2" GPU_MAX_USED_MEM_MB=4096 bash scripts/run_example_videos_and_gs.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EXPERIMENT:-lyra2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/example_batch}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

RUN_OUR_EXAMPLES="${RUN_OUR_EXAMPLES:-1}"
RUN_CUSTOM_TRAJ="${RUN_CUSTOM_TRAJ:-0}"
RUN_SAMPLES="${RUN_SAMPLES:-0}"

USE_DMD="${USE_DMD:-0}"
FORCE_GS="${FORCE_GS:-0}"
FORCE_VIDEO="${FORCE_VIDEO:-0}"
SKIP_GS="${SKIP_GS:-0}"
DISABLE_MULTI_GPU="${DISABLE_MULTI_GPU:-0}"
DISABLE_GPU_BUSY_CHECK="${DISABLE_GPU_BUSY_CHECK:-0}"
GPU_MAX_USED_MEM_MB="${GPU_MAX_USED_MEM_MB:-1024}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-}"

FPS="${FPS:-16}"
RESOLUTION="${RESOLUTION:-480,832}"
ZOOM_IN_FRAMES="${ZOOM_IN_FRAMES:-81}"
ZOOM_OUT_FRAMES="${ZOOM_OUT_FRAMES:-241}"
ZOOM_IN_STRENGTH="${ZOOM_IN_STRENGTH:-0.5}"
ZOOM_OUT_STRENGTH="${ZOOM_OUT_STRENGTH:-1.5}"
CUSTOM_NUM_FRAMES="${CUSTOM_NUM_FRAMES:-481}"
CUSTOM_TRAJ_IDS="${CUSTOM_TRAJ_IDS:-0 1}"
SAMPLE_IDS="${SAMPLE_IDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14}"

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

TASK_TYPES=()
TASK_ARGS=()
TASK_LABELS=()
DMD_ARGS=()
[[ "$USE_DMD" == "1" ]] && DMD_ARGS=(--use_dmd)
GS_FORCE_ARGS=()
[[ "$FORCE_GS" == "1" ]] && GS_FORCE_ARGS=(--force)

timestamp() { date '+%F %T'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
require_path() { [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }; }
sample_stem() { printf "%02d" "$((10#$1))"; }

resolve_python_bin() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then return 0; fi
  if [[ "$PYTHON_BIN" == "python" ]] && command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; return 0; fi
  echo "$PYTHON_BIN not found. Activate the Lyra-2 conda env first." >&2
  exit 1
}

run_gs() {
  local video="$1" out="$2"
  require_path "$video"
  if [[ "$SKIP_GS" == "1" ]]; then log "Skipping GS because SKIP_GS=1: $video"; return 0; fi
  log "GS reconstruction starting: $video"
  "$PYTHON_BIN" -m lyra_2._src.inference.vipe_da3_gs_recon --input_video_path "$video" --output_dir "$out" "${GS_FORCE_ARGS[@]}" "${GS_EXTRA_ARGS_ARRAY[@]}"
  log "GS reconstruction finished: $out/reconstructed_scene.ply"
}

run_zoom_batch() {
  local worker="$1"; shift
  local stage="$OUTPUT_ROOT/staging/worker_$worker/zoom" image_stage="$stage/images" prompt_stage="$stage/prompts" out="$OUTPUT_ROOT/zoomgs"
  local item name image prompt gs pending=()
  (($# == 0)) && return 0
  rm -rf "$stage"; mkdir -p "$image_stage" "$prompt_stage" "$out"
  for item in "$@"; do
    IFS='|' read -r name image prompt gs <<< "$item"
    require_path "$image"; require_path "$prompt"
    [[ "$FORCE_VIDEO" == "1" ]] && rm -f "$out/videos/$name.mp4"
    if [[ ! -f "$out/videos/$name.mp4" ]]; then
      ln -sf "$(realpath "$image")" "$image_stage/$name.png"
      cp -f "$prompt" "$prompt_stage/$name.txt"
      pending+=("$item")
      log "Staged zoom item: $name"
    else
      log "Video exists, skipping generation: $out/videos/$name.mp4"
    fi
  done
  if ((${#pending[@]} > 0)); then
    log "Zoom video generation starting (${#pending[@]} items)"
    "$PYTHON_BIN" -m lyra_2._src.inference.lyra2_zoomgs_inference \
      --input_image_path "$image_stage" --num_samples "${#pending[@]}" --sample_start_idx 0 --prompt_dir "$prompt_stage" \
      --experiment "$EXPERIMENT" --checkpoint_dir "$CHECKPOINT_DIR" --output_path "$out" --fps "$FPS" --resolution "$RESOLUTION" \
      --num_frames_zoom_in "$ZOOM_IN_FRAMES" --num_frames_zoom_out "$ZOOM_OUT_FRAMES" \
      --zoom_in_strength "$ZOOM_IN_STRENGTH" --zoom_out_strength "$ZOOM_OUT_STRENGTH" \
      "${DMD_ARGS[@]}" "${VIDEO_EXTRA_ARGS_ARRAY[@]}"
  fi
  for item in "$@"; do
    IFS='|' read -r name image prompt gs <<< "$item"
    run_gs "$out/videos/$name.mp4" "$gs"
  done
}

run_custom_batch() {
  local worker="$1"; shift
  local stage="$OUTPUT_ROOT/staging/worker_$worker/custom" image_stage="$stage/images" traj_stage="$stage/trajectories" cap_stage="$stage/captions" out="$OUTPUT_ROOT/custom_trajectory_examples/video"
  local id name src pending=()
  (($# == 0)) && return 0
  rm -rf "$stage"; mkdir -p "$image_stage" "$traj_stage" "$cap_stage" "$out"
  for id in "$@"; do
    name="example_$id"; src="assets/custom_trajectory_examples/$name"
    require_path "$src/first_frame.png"; require_path "$src/trajectory.npz"; require_path "$src/captions.json"
    [[ "$FORCE_VIDEO" == "1" ]] && rm -f "$out/$name.mp4"
    if [[ ! -f "$out/$name.mp4" ]]; then
      ln -sf "$(realpath "$src/first_frame.png")" "$image_stage/$name.png"
      ln -sf "$(realpath "$src/trajectory.npz")" "$traj_stage/$name.npz"
      cp -f "$src/captions.json" "$cap_stage/$name.json"
      pending+=("$id")
      log "Staged custom trajectory item: $name"
    else
      log "Video exists, skipping generation: $out/$name.mp4"
    fi
  done
  if ((${#pending[@]} > 0)); then
    log "Custom trajectory video generation starting (${#pending[@]} items)"
    "$PYTHON_BIN" -m lyra_2._src.inference.lyra2_custom_traj_inference \
      --input_image_path "$image_stage" --trajectory_path "$traj_stage" --captions_path "$cap_stage" \
      --num_samples "${#pending[@]}" --sample_start_idx 0 --experiment "$EXPERIMENT" --checkpoint_dir "$CHECKPOINT_DIR" \
      --output_path "$out" --num_frames "$CUSTOM_NUM_FRAMES" --fps "$FPS" --resolution "$RESOLUTION" \
      "${DMD_ARGS[@]}" "${VIDEO_EXTRA_ARGS_ARRAY[@]}"
  fi
  for id in "$@"; do name="example_$id"; run_gs "$out/$name.mp4" "$OUTPUT_ROOT/custom_trajectory_examples/$name/gs"; done
}

add_task() { TASK_LABELS+=("$1"); TASK_TYPES+=("$2"); TASK_ARGS+=("$3"); }
build_tasks() {
  if [[ "$RUN_OUR_EXAMPLES" == "1" ]]; then
    add_task "our:EscRoomNoDoor_Return" zoom "EscRoomNoDoor_Return|assets/EscRoomNoDoor.png|assets/EscRoomNoDoor_Return.txt|$OUTPUT_ROOT/our_examples/EscRoomNoDoor_Return/gs"
    add_task "our:EscRoomNoDoor_MoreRooms" zoom "EscRoomNoDoor_MoreRooms|assets/EscRoomNoDoor.png|assets/EscRoomNoDoor_MoreRooms.txt|$OUTPUT_ROOT/our_examples/EscRoomNoDoor_MoreRooms/gs"
    add_task "our:supermarket" zoom "supermarket|assets/supermarket.png|assets/supermarket.txt|$OUTPUT_ROOT/our_examples/supermarket/gs"
  fi
  if [[ "$RUN_CUSTOM_TRAJ" == "1" ]]; then for id in $CUSTOM_TRAJ_IDS; do add_task "custom:example_$id" custom "$id"; done; fi
  if [[ "$RUN_SAMPLES" == "1" ]]; then for id in $SAMPLE_IDS; do stem="$(sample_stem "$id")"; add_task "sample:$stem" zoom "$stem|assets/samples/$stem.png|assets/samples/$stem.txt|$OUTPUT_ROOT/samples/gs/$stem"; done; fi
}

emit_gpu_list() { tr ', ' '\n' <<< "$1" | awk 'NF'; }
detect_gpus() {
  if [[ -n "${GPU_IDS:-}" ]]; then emit_gpu_list "$GPU_IDS"; elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then emit_gpu_list "$CUDA_VISIBLE_DEVICES"; elif command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true; fi
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk 'NR==1 {gsub(/ /, "", $1); print $1}'; }
filter_gpus() {
  if [[ "$DISABLE_GPU_BUSY_CHECK" == "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then printf '%s\n' "$@"; return; fi
  local g used ok=()
  for g in "$@"; do used="$(gpu_used "$g")"; [[ "$used" =~ ^[0-9]+$ && "$used" -le "$GPU_MAX_USED_MEM_MB" ]] && ok+=("$g") || echo "Skipping busy GPU $g (${used:-unknown} MiB used)" >&2; done
  printf '%s\n' "${ok[@]}"
}

run_worker() {
  local idx="$1" gpu="$2" workers="$3" total="${#TASK_TYPES[@]}" chunk start end type payload zoom_items=() custom_ids=()
  [[ -n "$gpu" ]] && export CUDA_VISIBLE_DEVICES="$gpu"
  mkdir -p "$LOG_DIR"
  {
    log "Worker $idx using ${gpu:+GPU $gpu}${gpu:-default CUDA device}"
    chunk="$(((total + workers - 1) / workers))"; start="$((idx * chunk))"; end="$((start + chunk))"; ((end > total)) && end="$total"
    for ((i=start; i<end; i++)); do
      type="${TASK_TYPES[$i]}"; payload="${TASK_ARGS[$i]}"
      log "Assigned task $((i + 1))/$total: ${TASK_LABELS[$i]}"
      [[ "$type" == zoom ]] && zoom_items+=("$payload")
      [[ "$type" == custom ]] && custom_ids+=("$payload")
    done
    ((${#zoom_items[@]} > 0)) && run_zoom_batch "$idx" "${zoom_items[@]}"
    ((${#custom_ids[@]} > 0)) && run_custom_batch "$idx" "${custom_ids[@]}"
  } 2>&1 | tee -a "$LOG_DIR/worker_$idx.log"
}

run_all() {
  local candidates=() gpus=() workers pids=() status=0
  build_tasks
  ((${#TASK_TYPES[@]} > 0)) || { echo "No tasks selected."; return 0; }
  mapfile -t candidates < <(detect_gpus)
  if ((${#candidates[@]} > 0)); then mapfile -t gpus < <(filter_gpus "${candidates[@]}"); else gpus=(""); fi
  ((${#gpus[@]} > 0)) || { echo "No available GPUs below ${GPU_MAX_USED_MEM_MB} MiB used." >&2; exit 1; }
  workers="${#gpus[@]}"; [[ "$DISABLE_MULTI_GPU" == "1" ]] && workers=1
  [[ -n "$MAX_PARALLEL_JOBS" && "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ && "$MAX_PARALLEL_JOBS" -gt 0 && "$MAX_PARALLEL_JOBS" -lt "$workers" ]] && workers="$MAX_PARALLEL_JOBS"
  (("$workers" > "${#TASK_TYPES[@]}")) && workers="${#TASK_TYPES[@]}"
  log "Selected tasks: ${#TASK_TYPES[@]}"; log "Candidate GPUs: ${candidates[*]:-none}"; log "Available GPUs: ${gpus[*]:-none}"; log "Workers: $workers"
  if ((workers <= 1)); then run_worker 0 "${gpus[0]}" 1; return; fi
  for ((i=0; i<workers; i++)); do (run_worker "$i" "${gpus[$i]}" "$workers") & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}

resolve_python_bin
require_path "$CHECKPOINT_DIR"
require_path checkpoints/text_encoder/negative_prompt.pt
[[ "$USE_DMD" == "1" ]] && require_path checkpoints/lora/dmd_distillation.safetensors
[[ "$SKIP_GS" != "1" ]] && require_path checkpoints/recon/model.pt
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Experiment:  $EXPERIMENT"
echo "Use DMD:     $USE_DMD"
echo "Skip GS:     $SKIP_GS"
run_all
echo "Done. Outputs are under: $OUTPUT_ROOT"
