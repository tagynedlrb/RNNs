#!/bin/bash

TRACE_LIST=(10 283 372 661)
MODELS=(rnn lstm gru)
SUBS=(lba lba_o lba_delta lba_o_delta)

for t in "${TRACE_LIST[@]}"; do
  for m in "${MODELS[@]}"; do
    for s in "${SUBS[@]}"; do

      INFER="test_trace_plot/trace_${t}_${m}_${s}"
      ORIG="test_trace_plot/trace_${t}"

      echo "Running: $INFER"

      # INFER_CSV 교체
      sed -i "s|^INFER_CSV = .*|INFER_CSV = \"${INFER}\"|" LBA_delta_plot.py

      # ORIGINAL_TRACE 교체
      sed -i "s|^ORIGINAL_TRACE = .*|ORIGINAL_TRACE = \"${ORIG}\"|" LBA_delta_plot.py

      # 실행
      python3 LBA_delta_plot.py

    done
  done
done

echo "All runs complete."
