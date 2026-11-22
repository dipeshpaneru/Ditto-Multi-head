#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${DITTO_PYTHON:-"$root/.venv/bin/python"}

case "${1:-stsb}" in
    stsb) task_args=(--task_set na --tasks STSBenchmark) ;;
    all) task_args=(--task_set sts) ;;
    *) echo "Usage: bash reproduce.sh [stsb|all]" >&2; exit 2 ;;
esac

if [[ "$python" == */* ]]; then
    python=$(cd -- "$(dirname -- "$python")" && pwd)/$(basename -- "$python")
fi

mkdir -p "$root/results"
cd "$root/SpokenNLP/ditto"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

for pooler in avg_first_last att_first_last; do
    "$python" evaluation_ditto.py \
        --model_name_or_path "$root/models/bert-base-uncased" \
        --pooler "$pooler" --layer 0 --head 9 --mode test \
        "${task_args[@]}" 2>&1 | tee "$root/results/${1:-stsb}-$pooler.log"
done
