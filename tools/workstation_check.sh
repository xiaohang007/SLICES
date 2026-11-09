#!/bin/bash
# One-command checks of the refactor-uv branch on a GPU workstation (see HANDOFF_workstation.md).
# Everything is written to $OUT (default ~/slices_ws_test/<date_time>); send that folder back.
#
# Options (environment variables):
#   EXTRAS        uv extras to install            (default: "--extra cu128 --extra oeq")
#   TRAIN_EPOCHS  MatterGPT epochs for the timing  (default: 1; 0 skips training and generation)
#   DECODE_THREADS  decode worker counts to try    (default: "4 8")
#   OUT           output folder
#   UV_PROJECT_ENVIRONMENT  environment location (uv default: <repo>/.venv)
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$HOME/slices_ws_test/$(date +%Y%m%d_%H%M%S)}
EXTRAS=${EXTRAS:-"--extra cu128 --extra oeq"}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
DECODE_THREADS=${DECODE_THREADS:-"4 8"}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/summary.log") 2>&1

log() { echo "[$(date +%T)] $*"; }

# background monitors: GPU memory/utilisation and host memory every 5 s
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
             --format=csv -l 5 > "$OUT/gpu.csv" 2>/dev/null &
  GPU_MON=$!
fi
( while true; do echo "$(date +%T) $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)MB_available"; sleep 5; done ) > "$OUT/host_mem.log" &
MEM_MON=$!
trap 'kill ${GPU_MON:-} $MEM_MON 2>/dev/null' EXIT

# step <name> <timeout seconds> <command...>: runs with a time limit, logs to $OUT/<name>.log
step() {
  local name=$1 limit=$2; shift 2
  log "START $name"
  local t0=$(date +%s)
  timeout "$limit" bash -c "$*" > "$OUT/$name.log" 2>&1
  local rc=$?
  log "END   $name: exit=$rc, $(( $(date +%s) - t0 )) s"
  return $rc
}

log "repo $REPO, branch $(git -C "$REPO" branch --show-current), commit $(git -C "$REPO" rev-parse --short HEAD)"
log "uv $(uv --version 2>&1)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | sed 's/^/    /'
else
  log "nvidia-smi not found (no NVIDIA driver visible)"
fi

# 1. environment
cd "$REPO"
if ! step uv_sync 3600 "uv sync $EXTRAS --locked"; then
  log "uv sync with '$EXTRAS' failed; retrying without --extra oeq"
  EXTRAS=$(echo "$EXTRAS" | sed 's/--extra oeq//')
  step uv_sync_retry 3600 "uv sync $EXTRAS --locked" || { log "uv sync failed, see uv_sync*.log"; exit 1; }
fi
source "${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}/bin/activate"
export PYTHONWARNINGS=ignore
step torch_info 300 "python -c \"
import torch, importlib.util
print('torch', torch.__version__, 'CUDA build', torch.version.cuda, 'cuda available', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
print('openequivariance importable', importlib.util.find_spec('openequivariance') is not None)
print('flash_attn importable', importlib.util.find_spec('flash_attn') is not None)
\""
cat "$OUT/torch_info.log" | sed 's/^/    /'

# 2. compile/cache NequIP models (default CPU path, and the opt-in GPU path for comparison)
step prepare_cpu 1800 "python -m slices.relax_backends --prepare"
step prepare_gpu 3600 "SLICES_DEVICE=cuda python -m slices.relax_backends --prepare --relax-model nequip-l"
grep -h -E "relax_model=|compiled|failed|falling back|test relaxation" "$OUT"/prepare_*.log | sed 's/^/    /'

# 3. force-field benchmark on 32 real SLICES (one process)
step bench_cpu 3600 "python examples/3.benchmark_relax_models.py --n 32 --models auto nequip-m+l"
step bench_gpu 3600 "SLICES_DEVICE=cuda python examples/3.benchmark_relax_models.py --n 32 --models nequip-l nequip-s+l"
grep -h -A4 "^summary" "$OUT"/bench_*.log | sed 's/^/    /'

# 4. MatterGPT in a copy (so the repo stays clean)
MG="$OUT/mg/MatterGPT"; mkdir -p "$OUT/mg"
tar -C "$REPO" --exclude='job_*' --exclude='__pycache__' --exclude='*.pt' -cf - MatterGPT | tar -C "$OUT/mg" -xf -
if [ "$TRAIN_EPOCHS" -gt 0 ] && ! grep -q "cuda available True" "$OUT/torch_info.log"; then
  log "no CUDA GPU: skipping MatterGPT training (batch 36 needs ~13 GB RAM on CPU)"
  TRAIN_EPOCHS=0
fi
if [ "$TRAIN_EPOCHS" -gt 0 ]; then
  step train 7200 "cd $MG/1_train_generate && python -u train.py --run_name eform --batch_size 36 \
      --max_epochs $TRAIN_EPOCHS --n_embd 256 --n_layer 8 --n_head 8 --learning_rate 3.3e-4 \
      --train_dataset ../0_dataset/train_data.csv --val_dataset ../0_dataset/val_data.csv \
      --slices_column_index 0 --prop_column_index_list 1"
  tr '\r' '\n' < "$OUT/train.log" | grep -E "Epoch [0-9]+ (Train|Val)|Saving|Error|out of memory" | tail -6 | sed 's/^/    /'
  step generate 3600 "cd $MG/1_train_generate && python -u generate.py --model_weight eform.pt --output_csv eform.csv \
      --batch_size 5 --gen_size 10 --train_dataset ../0_dataset/train_data.csv --prop_targets [[-2.0]]"
  tr '\r' '\n' < "$OUT/generate.log" | grep -E "valid ratio|Error" | tail -3 | sed 's/^/    /'
fi

# 5. MatterGPT decode + formation energy + novelty on 32 real SLICES, with several worker counts
python - "$MG" <<'EOF'
import csv, random, sys
csv.field_size_limit(10**9)
mg = sys.argv[1]
rows = list(csv.DictReader(open(f"{mg}/0_dataset/mp20_eform_bandgap_nonmetal.csv")))
random.Random(0).shuffle(rows)
with open(f"{mg}/1_train_generate/input_32.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["prop_0", "crystal_system", "SLICES"])
    for r in rows[:32]:
        w.writerow([r["formation_energy_per_atom"], r["crystal_system"], r["SLICES"]])
EOF
for n in $DECODE_THREADS; do
  step decode_t$n 3600 "cd $MG/demo_decode_novelty_check_eform_m3gnet && rm -rf job_* structure_database.db && \
      python run.py --input_csv ../1_train_generate/input_32.csv \
      --structure_json_for_novelty_check ../0_dataset/cifs_filtered.json --training_file ../0_dataset/train_data.csv \
      --output_csv decode_t$n.csv --threads $n"
  python - "$MG/demo_decode_novelty_check_eform_m3gnet/decode_t$n.csv" <<'EOF' | sed 's/^/    /'
import csv, statistics as st, sys
csv.field_size_limit(10**9)
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
    err = [abs(float(r["eform_chgnet"]) - float(r["prop_0"])) for r in rows if r.get("eform_chgnet")]
    print(f"decoded {len(rows)}/32, |eform - DFT| median {st.median(err):.3f} eV/atom")
except Exception as e:
    print("no decode result:", e)
EOF
done

# 6. peaks
if [ -s "$OUT/gpu.csv" ]; then
  awk -F', ' 'NR>1 {gsub(/ MiB/,"",$3); if ($3+0>m) m=$3+0} END {print "    peak GPU memory used: " m " MiB"}' "$OUT/gpu.csv"
fi
awk '{gsub(/MB_available/,"",$2); if (min=="" || $2+0<min) min=$2+0} END {print "    lowest host MemAvailable: " min " MB"}' "$OUT/host_mem.log"
log "done; results in $OUT"
