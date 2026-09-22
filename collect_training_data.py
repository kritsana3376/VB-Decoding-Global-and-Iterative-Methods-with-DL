"""Generate the training and validation sets for the neural fault locators.

Each frame is a received word pushed through the baseline E-hMP decoder
(`EhMP.hMP_gc_solve_easy`). The stored label is the **residual** symbol error
mask: 1 means the symbol is still wrong *after* that baseline pass. The
locators are therefore trained to predict what E-hMP failed to fix, not what
the channel originally corrupted.

Training and validation are collected as two independent splits with separate
seeds, and a hash of every (codeword, error) realisation is tracked so no
frame can appear in both.

Output: three pickle files sharing a timestamped prefix, plus metadata
describing the code parameters, the target semantics, and the stored shapes.
The training notebooks read that metadata and refuse to run if the target
semantics do not match.
"""

import hashlib
import os
import pickle
import time
from datetime import datetime

import numpy as np

import Channel_Coding as cc
import EhMP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Code parameters.
CODE_N = 31
CODE_K = 21
BITS_PER_SYMBOL = 32  # r

# Split sizes and seeds. The seeds are distinct from the evaluation seed
# (2028) used by run_all_for_journal.py.
TRAIN_PER_SER = 100_000
VAL_PER_SER = 25_000
TRAIN_SEED = 2026
VAL_SEED = 2027

# Channel operating points sampled for the dataset.
SER_VALUES = np.round(np.arange(0.00, 0.16, 0.01), 2)

TARGET_SEMANTICS = "residual symbol error after E-hMP"

# Output location. Override with AI_DECODER_DATA_DIR rather than editing this.
DATA_DIR = os.environ.get("AI_DECODER_DATA_DIR", "data")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_dataset_split(split_name, samples_per_ser, seed, seen_keys,
                          n, k, r, H):
    """Collect one independent split and return its samples plus class counts."""
    np.random.seed(seed)
    split_dataset = []
    positive_labels = 0

    for ser_index, SER in enumerate(SER_VALUES):
        print(f"{split_name}: SER={SER:.2f} ({ser_index + 1}/{len(SER_VALUES)})")

        accepted = 0
        while accepted < samples_per_ser:
            # The collector uses iid information bits (the all-zero frame is
            # allowed) and forces every flagged symbol to carry a nonzero
            # error value.
            _, V_dec, V, Error, _, Y_dec, _ = EhMP.create_received_word(
                n=n, k=k, r=r, H=H, SER=float(SER), H_type="Q|I",
                nonzero_data=False, nonzero_error=True,
            )

            # Never let one codeword/noise realisation land in two splits.
            frame_key = hashlib.blake2b(
                V.tobytes() + Error.tobytes(), digest_size=16
            ).digest()
            if frame_key in seen_keys:
                continue
            seen_keys.add(frame_key)

            # Baseline E-hMP pass: the state the locator will see at inference.
            recovered_dec, verified_list = EhMP.hMP_gc_solve_easy(H, Y_dec, r)
            recovered_dec = np.asarray(recovered_dec).reshape(-1)
            transmitted_dec = np.asarray(V_dec).reshape(-1)
            recovered_bin = cc.MatrixDec_to_MatrixBinary(recovered_dec, r).astype(np.int8)

            total_check_node = cc.Compute_Check_Nodes_Matrix(H, recovered_dec)
            bin_total_check_node = cc.MatrixDec_to_MatrixBinary(
                total_check_node, r
            ).astype(np.int8)

            # Label 1 means the symbol is still wrong after the baseline pass.
            residual_error_list = (recovered_dec != transmitted_dec).astype(np.int8)

            split_dataset.append({
                "Y_bin": recovered_bin,
                "verified_list": verified_list.astype(np.int8),
                "bin_total_check_node": bin_total_check_node,
                "true_error_List": residual_error_list,
                "SER": np.float32(SER),
            })
            positive_labels += int(np.sum(residual_error_list))
            accepted += 1

    total_labels = len(split_dataset) * n
    stats = {
        "name": split_name,
        "seed": seed,
        "num_frames": len(split_dataset),
        "frames_per_ser": samples_per_ser,
        "positive_labels": positive_labels,
        "negative_labels": total_labels - positive_labels,
    }
    return split_dataset, stats


def main():
    start_time = time.time()

    n, k, r = CODE_N, CODE_K, BITS_PER_SYMBOL
    G, H = EhMP.BCH_sys(n, k)
    print("G shape:", G.shape, "H shape:", H.shape)

    os.makedirs(DATA_DIR, exist_ok=True)
    file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_prefix = os.path.join(
        DATA_DIR, f"error_location_BCH_{n}_{k}_r{r}_{file_timestamp}_residual"
    )

    print("=" * 30, "START", "=" * 30)
    seen_frame_keys = set()
    train_dataset, train_stats = collect_dataset_split(
        "train", TRAIN_PER_SER, TRAIN_SEED, seen_frame_keys, n, k, r, H
    )
    val_dataset, val_stats = collect_dataset_split(
        "validation", VAL_PER_SER, VAL_SEED, seen_frame_keys, n, k, r, H
    )

    train_file = save_prefix + "_train.pkl"
    val_file = save_prefix + "_validation.pkl"
    metadata_file = save_prefix + "_metadata.pkl"

    with open(train_file, "wb") as f:
        pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_file, "wb") as f:
        pickle.dump(val_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "code": {"n": n, "k": k, "r": r},
        "ser_values": SER_VALUES.tolist(),
        "target": TARGET_SEMANTICS,
        "stored_shapes": {
            "Y_bin": [n, r],
            "verified_list": [n],
            "bin_total_check_node": [n - k, r],
            "true_error_List": [n],
        },
        "train": train_stats,
        "validation": val_stats,
    }
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training dataset:", train_file, train_stats)
    print("Validation dataset:", val_file, val_stats)
    print("Metadata:", metadata_file)
    print(f"run time = {time.time() - start_time:.4f} sec")
    print("=" * 30, "FINISHED", "=" * 30)

    # The notebooks take this prefix as DATA_PREFIX.
    print("\nDATA_PREFIX for the training notebooks:")
    print(f"  {save_prefix}")


if __name__ == "__main__":
    main()
