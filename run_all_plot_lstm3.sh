#!/bin/bash

TRACE_LIST=(283)
MODELS=(lstm_lba_o_delta)
SUBS=(lr2 lr3 lr4 lr5 lr6)

for t in "${TRACE_LIST[@]}"; do
  for m in "${MODELS[@]}"; do
    for s in "${SUBS[@]}"; do

      INFER="test_trace_lstm3_plot/trace_${t}_${m}_${s}"
      ORIG="test_trace_lstm3_plot/trace_${t}"

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
