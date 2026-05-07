#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Usage:
#   bash scripts/run_navigation_custom_traj_and_gs.sh
#   PREPARE_ONLY=1 bash scripts/run_navigation_custom_traj_and_gs.sh
#   SKIP_GS=1 bash scripts/run_navigation_custom_traj_and_gs.sh
#   GPU_IDS="0 2" GPU_MAX_USED_MEM_MB=4096 bash scripts/run_navigation_custom_traj_and_gs.sh
#   DA3_MODEL_PATH_CUSTOM=none bash scripts/run_navigation_custom_traj_and_gs.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EXPERIMENT:-lyra2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/model}"
DA3_MODEL_PATH_CUSTOM="${DA3_MODEL_PATH_CUSTOM:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/navigation_custom_traj}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
EXAMPLE_ROOT="${EXAMPLE_ROOT:-assets/custom_trajectory_examples}"

RUN_ESCROOM2="${RUN_ESCROOM2:-1}"
RUN_SUPERMARKET="${RUN_SUPERMARKET:-1}"
ESCROOM2_NAME="${ESCROOM2_NAME:-escroom2_navigation}"
SUPERMARKET_NAME="${SUPERMARKET_NAME:-supermarket_navigation}"

PREPARE_ONLY="${PREPARE_ONLY:-0}"
USE_DMD="${USE_DMD:-0}"
FORCE_VIDEO="${FORCE_VIDEO:-0}"
FORCE_GS="${FORCE_GS:-0}"
SKIP_GS="${SKIP_GS:-0}"
DISABLE_MULTI_GPU="${DISABLE_MULTI_GPU:-0}"
DISABLE_GPU_BUSY_CHECK="${DISABLE_GPU_BUSY_CHECK:-0}"
GPU_MAX_USED_MEM_MB="${GPU_MAX_USED_MEM_MB:-1024}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-}"

FPS="${FPS:-16}"
RESOLUTION="${RESOLUTION:-480,832}"
NUM_FRAMES="${NUM_FRAMES:-481}"
TRAJECTORY_FRAMES="${TRAJECTORY_FRAMES:-500}"
POSE_SCALE="${POSE_SCALE:-1.0}"

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

TASK_NAMES=()

timestamp() { date '+%F %T'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
require_path() { [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }; }
require_uint() { [[ "$2" =~ ^[0-9]+$ ]] || { echo "$1 must be an integer, got: $2" >&2; exit 1; }; }

resolve_python_bin() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then return 0; fi
  if [[ "$PYTHON_BIN" == "python" ]] && command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; return 0; fi
  echo "$PYTHON_BIN not found. Activate the Lyra-2 conda env first." >&2
  exit 1
}

validate_settings() {
  require_uint NUM_FRAMES "$NUM_FRAMES"
  require_uint TRAJECTORY_FRAMES "$TRAJECTORY_FRAMES"
  require_uint FPS "$FPS"
  require_uint GPU_MAX_USED_MEM_MB "$GPU_MAX_USED_MEM_MB"
  ((10#$TRAJECTORY_FRAMES >= 10#$NUM_FRAMES)) || {
    echo "TRAJECTORY_FRAMES ($TRAJECTORY_FRAMES) must be >= NUM_FRAMES ($NUM_FRAMES)." >&2
    exit 1
  }
}

DMD_ARGS=()
[[ "$USE_DMD" == "1" ]] && DMD_ARGS=(--use_dmd)
GS_FORCE_ARGS=()
[[ "$FORCE_GS" == "1" ]] && GS_FORCE_ARGS=(--force)
DA3_ARGS=()

resolve_da3_args() {
  if [[ "$DA3_MODEL_PATH_CUSTOM" == "auto" ]]; then
    [[ -f checkpoints/recon/model.pt ]] && DA3_ARGS=(--da3_model_path_custom checkpoints/recon/model.pt)
  elif [[ "$DA3_MODEL_PATH_CUSTOM" != "none" && -n "$DA3_MODEL_PATH_CUSTOM" ]]; then
    require_path "$DA3_MODEL_PATH_CUSTOM"
    DA3_ARGS=(--da3_model_path_custom "$DA3_MODEL_PATH_CUSTOM")
  fi
}

video_da3_label() {
  if ((${#DA3_ARGS[@]} > 0)); then printf '%s\n' "${DA3_ARGS[1]}"; else printf 'pretrained/HF default\n'; fi
}

prepare_navigation_examples() {
  require_path assets/EscRoom2.webp
  require_path assets/supermarket.png
  if [[ ! -f assets/EscRoom2.png || assets/EscRoom2.webp -nt assets/EscRoom2.png ]]; then
    log "Converting assets/EscRoom2.webp -> assets/EscRoom2.png"
    if command -v ffmpeg >/dev/null 2>&1; then
      ffmpeg -y -loglevel error -i assets/EscRoom2.webp assets/EscRoom2.png
    else
      "$PYTHON_BIN" - <<'PY'
from PIL import Image
Image.open("assets/EscRoom2.webp").convert("RGB").save("assets/EscRoom2.png")
PY
    fi
  fi

  log "Preparing navigation captions and trajectory files"
  TRAJECTORY_FRAMES="$TRAJECTORY_FRAMES" NUM_FRAMES="$NUM_FRAMES" ESCROOM2_NAME="$ESCROOM2_NAME" SUPERMARKET_NAME="$SUPERMARKET_NAME" "$PYTHON_BIN" - <<'PY'
import json, math, os, shutil
from pathlib import Path
import numpy as np

root = Path("assets/custom_trajectory_examples")
frames = int(os.environ["TRAJECTORY_FRAMES"])
num_frames = int(os.environ["NUM_FRAMES"])

esc_captions = {
  "0": "A wide first-person view inside a modern escape room lounge with red carpet, wood-paneled walls, patterned wallpaper, black leather chairs, a yellow armchair, a dark counter, monitors, framed clue art, wall lamps, a bookcase, a potted plant, and an open door on the right. Everything is perfectly still.",
  "81": "The camera glides forward across the red carpet toward the right-side doorway. The chairs, counter, monitors, framed pictures, wall lamps, patterned wallpaper, plant, shelves, and door remain frozen in the same warm interior lighting.",
  "161": "The camera passes through the open door and begins entering the connecting passage. The escape room behind it, including the counter, chairs, wall patterns, monitors, framed clue art, and lamps, stays motionless and realistic.",
  "241": "After passing the doorway, the camera turns left into a narrow corridor. The corridor walls, trim, carpet, door frame, dim lamps, and puzzle-room details remain static while only the viewpoint rotates and advances.",
  "321": "The camera moves forward down the corridor, looking along the connected escape-room passage. Warm light, textured wall surfaces, carpet, door frames, shelves, and hidden-room details remain unchanged and still.",
  "401": "At the end of the corridor, the camera completes a smooth U-turn and continues forward again. The hallway and room edges rotate around the viewer, but all objects, lights, furniture, plants, doors, and wall patterns remain fixed in place.",
  "481": "A final forward-moving view after the U-turn shows the connected corridor and nearby room geometry. The same quiet escape-room atmosphere, realistic textures, warm lamps, patterned walls, carpet, and doorway details remain perfectly still.",
}
market_captions = {
  "0": "A centered first-person view down a bright supermarket aisle under fluorescent lights, with long shelves of colorful snacks and grocery boxes on both sides, polished floor reflections, price tags, and the hanging aisle 16 sign for chocolate, sweets, biscuits, and cereal bars. Everything is still.",
  "81": "The camera glides forward down the middle of the aisle. The shelves, product boxes, hanging aisle sign, fluorescent panels, price labels, floor reflections, and distant endcap displays remain completely motionless.",
  "161": "Approaching the end of the aisle, the camera continues forward between the packed shelves. More endcap displays, cross-aisle space, overhead lights, and distant store fixtures become visible while the products remain frozen.",
  "241": "At the end of the aisle, the camera turns left into the cross aisle to reveal additional rows of supermarket aisles. The shelving, signs, product packaging, lights, polished floor, and displays remain static.",
  "321": "The camera moves forward along the cross aisle with more parallel aisles visible to the side, then prepares to turn right into another aisle. Colorful boxes, snack packets, shelf edges, price tags, overhead signs, and fluorescent light panels stay perfectly still.",
  "401": "The camera has turned right into another aisle and moves forward down the new row of shelves. The store layout, packed products, clean white lighting, endcaps, and floor reflections remain unchanged.",
  "481": "A final forward-moving view continues inside the second supermarket aisle. Rows of colorful products, shelf labels, overhead fluorescent lights, polished tiles, and distant store details remain frozen; only the camera perspective changes.",
}

def smoothstep(t): return t * t * (3.0 - 2.0 * t)
def make_path(keys):
    centers = np.zeros((frames, 3), dtype=np.float32)
    yaws = np.zeros((frames,), dtype=np.float32)
    keys = sorted(keys, key=lambda x: x[0])
    for (f0, p0, y0), (f1, p1, y1) in zip(keys[:-1], keys[1:]):
        f0, f1 = max(0, min(frames - 1, f0)), max(0, min(frames - 1, f1))
        if f1 <= f0: continue
        idx = np.arange(f0, f1 + 1)
        s = smoothstep((idx - f0).astype(np.float32) / float(f1 - f0))
        p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
        centers[idx] = p0 + (p1 - p0) * s[:, None]
        yaws[idx] = y0 + (y1 - y0) * s
    f, p, y = keys[-1]
    centers[f:] = np.asarray(p, np.float32)
    yaws[f:] = y
    return centers, yaws

def make_w2c(centers, yaws):
    mats = np.zeros((frames, 4, 4), dtype=np.float32)
    for i, (c, yd) in enumerate(zip(centers, yaws)):
        y = math.radians(float(yd))
        right = np.array([math.cos(y), 0.0, -math.sin(y)], np.float32)
        up = np.array([0.0, 1.0, 0.0], np.float32)
        fwd = np.array([math.sin(y), 0.0, math.cos(y)], np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.stack([right, up, fwd], axis=1)
        c2w[:3, 3] = c
        mats[i] = np.linalg.inv(c2w).astype(np.float32)
    return mats

def intrinsics():
    k = np.zeros((frames, 3, 3), dtype=np.float32)
    k[:, 0, 0] = 820.0; k[:, 1, 1] = 820.0
    k[:, 0, 2] = 640.0; k[:, 1, 2] = 360.0; k[:, 2, 2] = 1.0
    return k

def write_sections(path, captions):
    keys = sorted(int(k) for k in captions)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"Caption sections for Lyra-2 custom trajectory inference. Default active num_frames={num_frames}.\n")
        f.write("captions.json stores each section by start frame; the next start frame implies the previous section's end.\n")
        f.write("Keys >= num_frames are ignored by the current loader, so key 481 is active only if NUM_FRAMES > 481.\n\n")
        for i, start in enumerate(keys):
            next_active = [k for k in keys[i + 1:] if k < num_frames]
            end = (next_active[0] - 1) if start < num_frames and next_active else (num_frames - 1 if start < num_frames else start)
            state = "active" if start < num_frames else "inactive at default NUM_FRAMES"
            f.write(f"[{start}-{end}] ({state})\n{captions[str(start)]}\n\n")

def save(name, image, captions, keys):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, d / "first_frame.png")
    (d / "captions.json").write_text(json.dumps(captions, indent=2) + "\n", encoding="utf-8")
    write_sections(d / "caption_sections.txt", captions)
    centers, yaws = make_path(keys)
    np.savez(d / "trajectory.npz", w2c=make_w2c(centers, yaws), intrinsics=intrinsics(), image_height=np.asarray(720, dtype=np.int64), image_width=np.asarray(1280, dtype=np.int64))
    print(f"[prepare] {name}: {d}")

save(os.environ["ESCROOM2_NAME"], "assets/EscRoom2.png", esc_captions, [
    (0, (0.00, 0.00, 0.00), 0.0), (120, (0.35, 0.00, 1.20), 0.0),
    (180, (0.90, 0.00, 1.55), -90.0), (310, (-0.75, 0.00, 1.55), -90.0),
    (360, (-0.95, 0.00, 1.35), -180.0), (410, (-0.75, 0.00, 1.55), -270.0),
    (499, (0.45, 0.00, 1.55), -270.0)])
save(os.environ["SUPERMARKET_NAME"], "assets/supermarket.png", market_captions, [
    (0, (0.00, 0.00, 0.00), 0.0), (150, (0.00, 0.00, 1.90), 0.0),
    (215, (-0.30, 0.00, 2.15), -90.0), (320, (-1.85, 0.00, 2.15), -90.0),
    (380, (-2.15, 0.00, 2.45), 0.0), (499, (-2.15, 0.00, 4.00), 0.0)])
PY
}

run_gs() {
  local video="$1" out="$2"
  require_path "$video"
  if [[ "$SKIP_GS" == "1" ]]; then log "Skipping GS: $video"; return 0; fi
  log "GS reconstruction starting: $video"
  "$PYTHON_BIN" -m lyra_2._src.inference.vipe_da3_gs_recon --input_video_path "$video" --output_dir "$out" "${GS_FORCE_ARGS[@]}" "${GS_EXTRA_ARGS_ARRAY[@]}"
  log "GS reconstruction finished: $out/reconstructed_scene.ply"
}

emit_gpu_list() { tr ', ' '\n' <<< "$1" | awk 'NF'; }
detect_gpus() {
  if [[ -n "${GPU_IDS:-}" ]]; then emit_gpu_list "$GPU_IDS"; elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then emit_gpu_list "$CUDA_VISIBLE_DEVICES"; elif command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true; fi
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk 'NR==1 {gsub(/ /, "", $1); print $1}'; }
filter_gpus() {
  if [[ "$DISABLE_GPU_BUSY_CHECK" == "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then printf '%s\n' "$@"; return; fi
  local g used ok=()
  for g in "$@"; do
    used="$(gpu_used "$g")"
    [[ "$used" =~ ^[0-9]+$ && "$used" -le "$GPU_MAX_USED_MEM_MB" ]] && ok+=("$g") || echo "Skipping busy GPU $g (${used:-unknown} MiB used)" >&2
  done
  printf '%s\n' "${ok[@]}"
}

run_worker() {
  local idx="$1" gpu="$2" workers="$3" total="${#TASK_NAMES[@]}" chunk start end name stage out pending=()
  [[ -n "$gpu" ]] && export CUDA_VISIBLE_DEVICES="$gpu"
  mkdir -p "$LOG_DIR"
  {
    log "Worker $idx using ${gpu:+GPU $gpu}${gpu:-default CUDA device}"
    chunk="$(((total + workers - 1) / workers))"; start="$((idx * chunk))"; end="$((start + chunk))"; ((end > total)) && end="$total"
    ((start >= total)) && { log "Worker $idx has no tasks"; return 0; }
    stage="$OUTPUT_ROOT/staging/worker_$idx"; out="$OUTPUT_ROOT/videos"
    rm -rf "$stage"; mkdir -p "$stage/images" "$stage/trajectories" "$stage/captions" "$out"
    for ((i=start; i<end; i++)); do
      name="${TASK_NAMES[$i]}"
      require_path "$EXAMPLE_ROOT/$name/first_frame.png"; require_path "$EXAMPLE_ROOT/$name/trajectory.npz"; require_path "$EXAMPLE_ROOT/$name/captions.json"
      [[ "$FORCE_VIDEO" == "1" ]] && rm -f "$out/$name.mp4"
      if [[ ! -f "$out/$name.mp4" ]]; then
        ln -sf "$(realpath "$EXAMPLE_ROOT/$name/first_frame.png")" "$stage/images/$name.png"
        ln -sf "$(realpath "$EXAMPLE_ROOT/$name/trajectory.npz")" "$stage/trajectories/$name.npz"
        cp -f "$EXAMPLE_ROOT/$name/captions.json" "$stage/captions/$name.json"
        pending+=("$name")
      else
        log "Video exists, skipping generation: $out/$name.mp4"
      fi
    done
    if ((${#pending[@]} > 0)); then
      log "Video generation starting (${#pending[@]} items)"
      "$PYTHON_BIN" -m lyra_2._src.inference.lyra2_custom_traj_inference \
        --input_image_path "$stage/images" --trajectory_path "$stage/trajectories" --captions_path "$stage/captions" \
        --num_samples "${#pending[@]}" --sample_start_idx 0 --experiment "$EXPERIMENT" --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_path "$out" --num_frames "$NUM_FRAMES" --fps "$FPS" --resolution "$RESOLUTION" --pose_scale "$POSE_SCALE" \
        "${DA3_ARGS[@]}" "${DMD_ARGS[@]}" "${VIDEO_EXTRA_ARGS_ARRAY[@]}"
    fi
    for ((i=start; i<end; i++)); do name="${TASK_NAMES[$i]}"; run_gs "$out/$name.mp4" "$OUTPUT_ROOT/gs/$name"; done
  } 2>&1 | tee -a "$LOG_DIR/worker_$idx.log"
}

build_tasks() {
  [[ "$RUN_ESCROOM2" == "1" ]] && TASK_NAMES+=("$ESCROOM2_NAME")
  [[ "$RUN_SUPERMARKET" == "1" ]] && TASK_NAMES+=("$SUPERMARKET_NAME")
}

run_all() {
  build_tasks
  ((${#TASK_NAMES[@]} > 0)) || { echo "No tasks selected."; return 0; }
  local candidates=() gpus=() workers pids=() status=0
  mapfile -t candidates < <(detect_gpus)
  if ((${#candidates[@]} > 0)); then
    mapfile -t gpus < <(filter_gpus "${candidates[@]}")
    ((${#gpus[@]} > 0)) || { echo "No available GPUs below ${GPU_MAX_USED_MEM_MB} MiB used." >&2; exit 1; }
  else
    gpus=("")
  fi
  workers="${#gpus[@]}"; [[ "$DISABLE_MULTI_GPU" == "1" ]] && workers=1
  [[ -n "$MAX_PARALLEL_JOBS" && "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ && "$MAX_PARALLEL_JOBS" -gt 0 && "$MAX_PARALLEL_JOBS" -lt "$workers" ]] && workers="$MAX_PARALLEL_JOBS"
  (("$workers" > "${#TASK_NAMES[@]}")) && workers="${#TASK_NAMES[@]}"
  log "Tasks: ${TASK_NAMES[*]}"; log "Candidate GPUs: ${candidates[*]:-none}"; log "Available GPUs: ${gpus[*]:-none}"; log "Workers: $workers"
  if ((workers <= 1)); then run_worker 0 "${gpus[0]}" 1; return; fi
  for ((i=0; i<workers; i++)); do (run_worker "$i" "${gpus[$i]}" "$workers") & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}

resolve_python_bin
validate_settings
prepare_navigation_examples
if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Prepared assets/custom_trajectory_examples/$ESCROOM2_NAME and $SUPERMARKET_NAME"
  exit 0
fi
resolve_da3_args
require_path "$CHECKPOINT_DIR"
require_path checkpoints/text_encoder/negative_prompt.pt
[[ "$USE_DMD" == "1" ]] && require_path checkpoints/lora/dmd_distillation.safetensors
[[ "$SKIP_GS" != "1" ]] && require_path checkpoints/recon/model.pt
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "Output root:       $OUTPUT_ROOT"
echo "Experiment:        $EXPERIMENT"
echo "Video DA3 ckpt:    $(video_da3_label)"
echo "Use DMD:           $USE_DMD"
echo "Skip GS:           $SKIP_GS"
run_all
echo "Done. Outputs are under: $OUTPUT_ROOT"
