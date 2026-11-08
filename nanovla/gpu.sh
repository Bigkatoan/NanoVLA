#!/bin/bash
# Run a command while holding the single-GPU lock (Isaac Sim / training must never overlap on the 12 GB RTX 3060).
#   nanovla/gpu.sh <timeout_seconds> <command...>
# Waits for the lock, then runs the command with a hard KILL timeout (Isaac Kit can hang on exit).
T=${1:?timeout seconds}; shift
exec flock "$(dirname "$0")/../.gpu.lock" timeout -s KILL "$T" "$@"
