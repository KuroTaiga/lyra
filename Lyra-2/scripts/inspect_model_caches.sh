#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
MAX_FILES="${MAX_FILES:-80}"

section() { printf '\n== %s ==\n' "$*"; }
path_or_empty() { [[ -n "${1:-}" ]] && printf '%s\n' "$1"; }
dedupe_lines() { awk '!seen[$0]++'; }
existing_dirs_from_list() {
  local dir
  while IFS= read -r dir; do
    [[ -n "$dir" && -d "$dir" ]] && printf '%s\n' "$dir"
  done
  return 0
}
human_file_list() {
  awk '{size=$1; $1=""; sub(/^ /, ""); cmd="numfmt --to=iec --suffix=B " size; cmd | getline human; close(cmd); print human "  " $0}'
}

default_hf_home="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
default_hf_hub="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE:-$default_hf_home/hub}}"
default_torch_home="${TORCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/torch}"

section "Machine"
hostname || true
date || true
printf 'repo: %s\n' "$PWD"

section "Relevant Env Vars"
env | sort | grep -E '^(HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|TRANSFORMERS_CACHE|TORCH_HOME|XDG_CACHE_HOME|PIP_CACHE_DIR|CONDA_PKGS_DIRS|CUDA_VISIBLE_DEVICES)=' || true

section "GPU Memory"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || true
else
  echo "nvidia-smi not found"
fi

section "RAM And Linux Page Cache"
free -h || true
awk '/^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SReclaimable|SwapTotal|SwapFree):/ {print}' /proc/meminfo 2>/dev/null || true
command -v vmtouch >/dev/null 2>&1 && echo "vmtouch: installed" || echo "vmtouch: not installed"

section "Resolved Cache/Checkpoint Dirs"
cache_dirs="$(
  {
    path_or_empty "$PWD/checkpoints"
    path_or_empty "$PWD/checkpoints/model"
    path_or_empty "$PWD/checkpoints/recon"
    path_or_empty "$PWD/checkpoints/text_encoder"
    path_or_empty "$PWD/checkpoints/lora"
    path_or_empty "$PWD/checkpoints/vipe"
    path_or_empty "$PWD/checkpoints/huggingface"
    path_or_empty "$PWD/checkpoints/huggingface/hub"
    path_or_empty "$PWD/checkpoints/torch"
    path_or_empty "$default_hf_home"
    path_or_empty "$default_hf_hub"
    path_or_empty "${TRANSFORMERS_CACHE:-}"
    path_or_empty "$default_torch_home"
    path_or_empty "${PIP_CACHE_DIR:-}"
    path_or_empty "$HOME/.cache/huggingface"
    path_or_empty "$HOME/.cache/torch"
    path_or_empty "$HOME/.cache/pip"
    path_or_empty "/root/.cache/huggingface"
    path_or_empty "/root/.cache/torch"
    path_or_empty "/root/.cache/pip"
  } | dedupe_lines | existing_dirs_from_list
)"
[[ -n "$cache_dirs" ]] && printf '%s\n' "$cache_dirs" || echo "No expected cache/checkpoint dirs found."

section "Filesystem Free Space"
if [[ -n "$cache_dirs" ]]; then
  # shellcheck disable=SC2086
  df -hT . $cache_dirs 2>/dev/null | dedupe_lines || true
else
  df -hT . || true
fi

section "Directory Sizes"
if [[ -n "$cache_dirs" ]]; then
  while IFS= read -r dir; do du -sh "$dir" 2>/dev/null || true; done <<< "$cache_dirs" | sort -h
else
  echo "No directories to size."
fi

section "Hugging Face Model Repo Sizes"
hf_hub_dirs="$(
  {
    path_or_empty "$default_hf_hub"
    path_or_empty "$PWD/checkpoints/huggingface/hub"
    path_or_empty "$HOME/.cache/huggingface/hub"
    path_or_empty "/root/.cache/huggingface/hub"
  } | dedupe_lines | existing_dirs_from_list
)"
if [[ -n "$hf_hub_dirs" ]]; then
  while IFS= read -r hub; do find "$hub" -maxdepth 1 -mindepth 1 -type d -name 'models--*' -exec du -sh {} + 2>/dev/null || true; done <<< "$hf_hub_dirs" | sort -h
else
  echo "No Hugging Face hub cache dirs found."
fi

section "Largest Model/Checkpoint Files"
search_dirs="$({ printf '%s\n' "$cache_dirs"; path_or_empty "$PWD"; } | dedupe_lines | existing_dirs_from_list)"
if [[ -n "$search_dirs" ]]; then
  while IFS= read -r dir; do
    find "$dir" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.tar' -o -name '*.npz' \) -printf '%s %p\n' 2>/dev/null || true
  done <<< "$search_dirs" | sort -nr | head -n "$MAX_FILES" | human_file_list
else
  echo "No directories to search."
fi

section "Conda/Pip Cache Hints"
if command -v conda >/dev/null 2>&1; then conda info 2>/dev/null | grep -E 'package cache|envs directories|base environment' || true; else echo "conda not found"; fi
if command -v pip >/dev/null 2>&1; then pip cache dir 2>/dev/null || true; pip cache info 2>/dev/null || true; else echo "pip not found"; fi

section "Recommended Persistent Cache Exports"
cat <<'EOF'
mkdir -p checkpoints/huggingface/hub checkpoints/torch
export HF_HOME="$PWD/checkpoints/huggingface"
export HUGGINGFACE_HUB_CACHE="$PWD/checkpoints/huggingface/hub"
export TRANSFORMERS_CACHE="$PWD/checkpoints/huggingface/hub"
export TORCH_HOME="$PWD/checkpoints/torch"
EOF
