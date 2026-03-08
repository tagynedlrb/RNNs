import os
import shutil

BASE_DIR = os.getcwd()
DEST_DIR = os.path.join(BASE_DIR, "test_trace")

os.makedirs(DEST_DIR, exist_ok=True)

# ==========================================================
# (source_path, destination_filename_without_extension)
# ==========================================================

FILES = [

# ==========================
# RNN
# ==========================
# LBA
("RNN/trained_rnn_lba/20260225/v0/trace_10/inference/inference_20260304_input20952_gen32768_v0.csv",
 "trace_10_rnn_lba"),

("RNN/trained_rnn_lba/20260226/v3/trace_283/inference/inference_20260302_input20952_gen32768_v3.csv",
 "trace_283_rnn_lba"),

("RNN/trained_rnn_lba/20260228/v0/trace_372/inference/inference_20260302_input5238_gen32768_v0.csv",
 "trace_372_rnn_lba"),

("RNN/trained_rnn_lba/20260302/v0/trace_661/inference/inference_20260302_input163_gen32768_v0.csv",
 "trace_661_rnn_lba"),

# LBA delta
("RNN/trained_rnn_lba_delta/20260226/v0/trace_10/inference/inference_20260302_gen32768_recon.csv",
 "trace_10_rnn_lba_delta"),

("RNN/trained_rnn_lba_delta/20260226/v1/trace_283/inference/inference_20260302_gen32768_recon.csv",
 "trace_283_rnn_lba_delta"),

("RNN/trained_rnn_lba_delta/20260226/v2/trace_372/inference/inference_20260302_gen32768_recon.csv",
 "trace_372_rnn_lba_delta"),

("RNN/trained_rnn_lba_delta/20260302/v0/trace_661/inference/inference_20260302_gen32768_recon.csv",
 "trace_661_rnn_lba_delta"),

# ORBD
("RNN/trained_rnn_orbd/20260226/v0/trace_10/inference/inference_20260302_input20952_gen32768_v0.csv",
 "trace_10_rnn_lba_o"),

("RNN/trained_rnn_orbd/20260302/v1/trace_283/inference/inference_20260302_input20952_gen32768_v1.csv",
 "trace_283_rnn_lba_o"),

("RNN/trained_rnn_orbd/20260302/v0/trace_372/inference/inference_20260302_input5238_gen32768_v0.csv",
 "trace_372_rnn_lba_o"),

("RNN/trained_rnn_orbd/20260302/v2/trace_661/inference/inference_20260302_input163_gen32768_v2.csv",
 "trace_661_rnn_lba_o"),

# ORBD delta
("RNN/trained_rnn_orbd_delta/20260226/v0/trace_10/inference/recon_20260302_input20952_gen32768_v0.csv",
 "trace_10_rnn_lba_o_delta"),

("RNN/trained_rnn_orbd_delta/20260302/v0/trace_283/inference/recon_20260302_input20952_gen32768_v0.csv",
 "trace_283_rnn_lba_o_delta"),

("RNN/trained_rnn_orbd_delta/20260302/v1/trace_372/inference/recon_20260302_input5238_gen32768_v1.csv",
 "trace_372_rnn_lba_o_delta"),

("RNN/trained_rnn_orbd_delta/20260302/v2/trace_661/inference/recon_20260302_input163_gen32768_v2.csv",
 "trace_661_rnn_lba_o_delta"),

# ==========================
# GRU
# ==========================
# LBA
("GRU/trained_gru_lba/20260227/v0/trace_10/inference/inference_20260302_input20952_gen32768_v0.csv",
 "trace_10_gru_lba"),

("GRU/trained_gru_lba/20260301/v0/trace_283/inference/inference_20260302_input20952_gen32768_v0.csv",
 "trace_283_gru_lba"),

("GRU/trained_gru_lba/20260302/v0/trace_372/inference/inference_20260302_input5238_gen32768_v0.csv",
 "trace_372_gru_lba"),

("GRU/trained_gru_lba/20260302/v1/trace_661/inference/inference_20260302_input163_gen32768_v1.csv",
 "trace_661_gru_lba"),

# LBA delta
("GRU/trained_gru_lba_delta/20260228/v0/trace_10/inference/inference_20260302_gen32768_recon.csv",
 "trace_10_gru_lba_delta"),

("GRU/trained_gru_lba_delta/20260301/v0/trace_283/inference/inference_20260302_gen32768_recon.csv",
 "trace_283_gru_lba_delta"),

("GRU/trained_gru_lba_delta/20260302/v0/trace_372/inference/inference_20260302_gen32768_recon.csv",
 "trace_372_gru_lba_delta"),

("GRU/trained_gru_lba_delta/20260302/v1/trace_661/inference/inference_20260302_gen32768_recon.csv",
 "trace_661_gru_lba_delta"),

# ORBD
("GRU/trained_gru_orbd/20260302/v0/trace_10/inference/inference_20260302_input20952_gen32768_v0.csv",
 "trace_10_gru_lba_o"),

("GRU/trained_gru_orbd/20260302/v2/trace_283/inference/inference_20260302_input20952_gen32768_v2.csv",
 "trace_283_gru_lba_o"),

("GRU/trained_gru_orbd/20260302/v3/trace_372/inference/inference_20260302_input5238_gen32768_v3.csv",
 "trace_372_gru_lba_o"),

("GRU/trained_gru_orbd/20260302/v4/trace_661/inference/inference_20260302_input163_gen32768_v4.csv",
 "trace_661_gru_lba_o"),

# ORBD delta
("GRU/trained_gru_orbd_delta/20260302/v1/trace_10/inference/recon_20260302_input20952_gen32768_v1.csv",
 "trace_10_gru_lba_o_delta"),

("GRU/trained_gru_orbd_delta/20260302/v2/trace_283/inference/recon_20260302_input20952_gen32768_v2.csv",
 "trace_283_gru_lba_o_delta"),

("GRU/trained_gru_orbd_delta/20260302/v5/trace_372/inference/recon_20260302_input5238_gen32768_v5.csv",
 "trace_372_gru_lba_o_delta"),

("GRU/trained_gru_orbd_delta/20260302/v6/trace_661/inference/recon_20260302_input163_gen32768_v6.csv",
 "trace_661_gru_lba_o_delta"),

# ==========================
# LSTM
# ==========================
# LBA
("LSTM/trained_lstm_lba/20260227/v0/trace_10/inference/inference_20260302_input20952_gen32768_v0.csv",
 "trace_10_lstm_lba"),

("LSTM/trained_lstm_lba/20260228/v1/trace_283/inference/inference_20260302_input20952_gen32768_v1.csv",
 "trace_283_lstm_lba"),

("LSTM/trained_lstm_lba/20260302/v0/trace_372/inference/inference_20260302_input5238_gen32768_v0.csv",
 "trace_372_lstm_lba"),

("LSTM/trained_lstm_lba/20260302/v1/trace_661/inference/inference_20260302_input163_gen32768_v1.csv",
 "trace_661_lstm_lba"),

# LBA delta
("LSTM/trained_lstm_lba_delta/20260228/v0/trace_10/inference/inference_20260302_gen32768_recon.csv",
 "trace_10_lstm_lba_delta"),

("LSTM/trained_lstm_lba_delta/20260228/v1/trace_283/inference/inference_20260302_gen32768_recon.csv",
 "trace_283_lstm_lba_delta"),

("LSTM/trained_lstm_lba_delta/20260302/v0/trace_372/inference/inference_20260302_gen32768_recon.csv",
 "trace_372_lstm_lba_delta"),

("LSTM/trained_lstm_lba_delta/20260302/v1/trace_661/inference/inference_20260302_gen32768_recon.csv",
 "trace_661_lstm_lba_delta"),

# ORBD
("LSTM/trained_lstm_orbd/20260302/v0/trace_10/inference/inference_20260302_input20952_gen32768_v0.csv",
 "trace_10_lstm_lba_o"),

("LSTM/trained_lstm_orbd/20260302/v2/trace_283/inference/inference_20260302_input20952_gen32768_v2.csv",
 "trace_283_lstm_lba_o"),

("LSTM/trained_lstm_orbd/20260302/v4/trace_372/inference/inference_20260302_input5238_gen32768_v4.csv",
 "trace_372_lstm_lba_o"),

("LSTM/trained_lstm_orbd/20260302/v5/trace_661/inference/inference_20260302_input163_gen32768_v5.csv",
 "trace_661_lstm_lba_o"),

# ORBD delta
("LSTM/trained_lstm_orbd_delta/20260302/v0/trace_10/inference/recon_20260302_input20952_gen32768_v0.csv",
 "trace_10_lstm_lba_o_delta"),

("LSTM/trained_lstm_orbd_delta/20260302/v2/trace_283/inference/recon_20260302_input20952_gen32768_v2.csv",
 "trace_283_lstm_lba_o_delta"),

("LSTM/trained_lstm_orbd_delta/20260302/v3/trace_372/inference/recon_20260302_input5238_gen32768_v3.csv",
 "trace_372_lstm_lba_o_delta"),

("LSTM/trained_lstm_orbd_delta/20260302/v4/trace_661/inference/recon_20260302_input163_gen32768_v4.csv",
 "trace_661_lstm_lba_o_delta"),
]

# ==========================================================

for src, new_name in FILES:
    full_src = os.path.join(BASE_DIR, src)

    if not os.path.exists(full_src):
        print(f"[ERROR] Not found: {full_src}")
        continue

    dst_path = os.path.join(DEST_DIR, new_name)
    shutil.copy2(full_src, dst_path)
    print(f"[OK] {new_name}")

print("\nDone.")
