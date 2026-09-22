"""Monte Carlo evaluation of neural-network-guided BCH decoders.

Benchmarks nine decoder configurations on a systematic BCH(n, k) code whose
symbols carry ``r`` bits each, over a q-ary symbol error channel. Every
decoder sees the same received word in each trial, so the comparison is
paired.

    EhMP          E-hMP, loop-fast "new single" variant (algebraic)
    E_hMP_guid_CNN     EhMP + CNN-guided verified-flag flipping
    E_hMP_guid_GNN     EhMP + GNN-guided verified-flag flipping
    MhMP          M-hMP, loop-fast "common back" variant (algebraic)
    M_hMP_guid_CNN     MhMP + CNN-guided verified-flag flipping
    M_hMP_guid_GNN     MhMP + GNN-guided verified-flag flipping
    VSD_PV            Verified Symbol Decoding, accepting false verifications
    VSD_guid_CNN/GNN   VSD driven by CNN/GNN fault probabilities

Per-trial results are appended to a CSV in ``OUTPUT_DIR``, one file per
symbol error rate, flushed every ``CHUNK_SIZE`` trials to bound memory use.

Configuration comes from environment variables and the constants below; see
README.md.
"""

import os
import random as python_random
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import Channel_Coding as cc
import EhMP
import fault_locator as fl
import VSD as vm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Code parameters.
CODE_N = 31
CODE_K = 21
BITS_PER_SYMBOL = 32  # r

# Simulation parameters.
NUM_TRIALS = 10_000_000
SER_VALUES = np.linspace(0.02, 0.15, 14)
SIMULATION_SEED = 2028  # independent from the training/validation seeds 2026/2027
CHUNK_SIZE = 10_000     # rows buffered in RAM before being appended to the CSV

# Decoder parameters.
NONZERO = True   # resample until the information frame is not all-zero
MAX_ERROR = 1    # maximum number of verified flags a guided decoder may flip

# Paths. Override with environment variables to avoid editing this file.
OUTPUT_DIR = os.environ.get("AI_DECODER_OUTPUT_DIR", "results")
MODEL_DIR = os.environ.get("AI_DECODER_MODEL_DIR", "models")
MODEL_FILE_CNN = "cnn_error_locator_BCH_31_21_r32_Update.pth"
MODEL_FILE_GNN = "gnn_error_locator_BCH_31_21_r32_Update.pth"

OUTPUT_FILENAME_TEMPLATE = "p{ser:.2f}_raw_data_add_EMhMPVSD_2.csv"


# ---------------------------------------------------------------------------
# Neural-guided decoding
# ---------------------------------------------------------------------------

def flip_top_k_Verified(verified_list, ranked_faults, K, Parity_Check_Matrix_H):
    """Promote up to K ranked unverified flags in H=[Q|I]'s identity block.

    The identity columns are k..n-1 (zero-based), where k = n - H.shape[0].
    Filter before taking the top K: ignore data positions, already verified
    positions, and duplicate indices, and never demote a verified flag.
    The input mask is not modified.
    """
    H = np.asarray(Parity_Check_Matrix_H)
    if H.ndim != 2 or not 0 < H.shape[0] <= H.shape[1]:
        raise ValueError("H must be a nonempty parity-check matrix with rows <= columns")
    m, n = H.shape
    k = n - m
    if not np.isin(H, [0, 1]).all() or not np.array_equal(H[:, k:], np.eye(m)):
        raise ValueError("Identity-only flipping requires systematic H=[Q|I]")

    mask = np.asarray(verified_list)
    if mask.shape != (n,) or not np.isin(mask, [0, 1]).all():
        raise ValueError("verified_list must be a binary vector of length H.shape[1]")
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K must be a nonnegative integer")

    result = mask.astype(np.int8, copy=True)
    if K == 0:
        return result

    promoted = 0
    for i, _ in ranked_faults:
        if not isinstance(i, (int, np.integer)) or not 0 <= i < n:
            raise ValueError("Ranked symbol indices must be integers in [0, n)")
        if i < k or result[i] != 0:
            continue
        result[i] = 1
        promoted += 1
        if promoted == K:
            break
    return result


def guided_retry(retry_decoder, fault_model, Parity_Check_Matrix_H,
                 recovered_pure_dec_codeword, BitPerSymbol_r, verified_list):
    """Retry decoding with neural guidance over the identity block.

    The fault locator scores every unverified symbol; the symbols least likely
    to be faulty are optimistically marked verified, up to MAX_ERROR of them,
    and ``retry_decoder`` is re-run with that mask as its starting point. A
    retry is accepted as soon as the syndrome vanishes. Promotions are
    restricted to H=[Q|I]'s last n-k columns, though the algebraic decoder may
    still correct and verify other positions.

    Returns:
        best_recovered, success_process_flip, best_verified_list, prob,
        verified_list_make_success
    """
    Parity_Check_Matrix_H = np.asarray(Parity_Check_Matrix_H)

    # Validate the systematic layout and take a private copy of the mask.
    verified_list = flip_top_k_Verified(verified_list, [], 0, Parity_Check_Matrix_H)
    identity_start = Parity_Check_Matrix_H.shape[1] - Parity_Check_Matrix_H.shape[0]

    success_process_flip = False
    best_recovered = np.asarray(recovered_pure_dec_codeword).copy()
    best_verified_list = verified_list
    verified_list_make_success = np.zeros(Parity_Check_Matrix_H.shape[1], dtype=np.int8)

    # Check-node values of the current estimate, in binary.
    Total_Check_Node = cc.Compute_Check_Nodes_Matrix(
        Parity_Check_Matrix_H, recovered_pure_dec_codeword
    )
    bin_total_check_node = cc.MatrixDec_to_MatrixBinary(Total_Check_Node, BitPerSymbol_r)
    Y_Bin = cc.MatrixDec_to_MatrixBinary(recovered_pure_dec_codeword, BitPerSymbol_r)

    _, prob = fl.predict_fault_positions_model(
        fault_model, Parity_Check_Matrix_H, Y_Bin, verified_list, bin_total_check_node
    )

    # Unverified identity-block symbols, least likely to be faulty first.
    ranked_correct = sorted(
        [
            (i, prob[i])
            for i in range(identity_start, len(verified_list))
            if verified_list[i] == 0
        ],
        key=lambda x: x[1],
    )

    for K in range(1, MAX_ERROR + 1):
        Verified_flipped = flip_top_k_Verified(
            verified_list, ranked_correct, K, Parity_Check_Matrix_H
        )

        # The decoder copies both arguments internally.
        recovered_retry, verified_recovered_retry = retry_decoder(
            Parity_Check_Matrix_H,
            best_recovered,
            BitPerSymbol_r,
            initial_verified=Verified_flipped,
        )

        # Accept the retry on a zero syndrome.
        retry_check_nodes = cc.Compute_Check_Nodes_Matrix(
            Parity_Check_Matrix_H, recovered_retry
        )
        if np.sum(retry_check_nodes) == 0:
            success_process_flip = True
            best_recovered = recovered_retry
            best_verified_list = verified_recovered_retry
            verified_list_make_success = Verified_flipped
            break

    return (best_recovered, success_process_flip, best_verified_list, prob,
            verified_list_make_success)


def E_hMP_guide_model(fault_model, H, recovered_pure_dec_codeword, r, verified_list):
    """Neural-guided retries on top of the E-hMP decoder."""
    return guided_retry(
        EhMP.EhMP,
        fault_model, H, recovered_pure_dec_codeword, r, verified_list,
    )


def M_hMP_guide_model(fault_model, H, recovered_pure_dec_codeword, r, verified_list):
    """Neural-guided retries on top of the M-hMP decoder."""
    return guided_retry(
        EhMP.MhMP,
        fault_model, H, recovered_pure_dec_codeword, r, verified_list,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _error_dict_to_str(error_dict):
    """Render an {index: value} error map as a CSV-safe string."""
    return str({int(key): int(value) for key, value in error_dict.items()})


def _seed_everything(seed):
    """Seed numpy, the standard library, and torch from a single value."""
    np.random.seed(seed)
    python_random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flush(rows, csv_path):
    """Append buffered rows to the CSV, writing a header on first use."""
    if not rows:
        return
    write_header = not os.path.exists(csv_path)
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=write_header, index=False)


def run_one_trial(H, n, k, r, SER, fault_model_CNN, fault_model_GNN):
    """Run every decoder on one random received word and return a result row."""
    # The evaluation channel rejects the all-zero information frame but allows
    # a flagged symbol to draw an all-zero error value. The training collector
    # makes the opposite choice on both flags; see README.
    _, V_dec, _, Error, Y, Y_dec, Error_Marker = EhMP.create_received_word(
        n=n, k=k, r=r, H=H, SER=SER, H_type="Q|I",
        nonzero_data=NONZERO, nonzero_error=False,
    )
    codeword = cc.MatrixDec_to_MatrixBinary(V_dec, r)

    # --- E-hMP, plain and neural-guided ---
    recovered_EhMP, verified_list_EhMP = EhMP.EhMP(
        H, Y_dec, r
    )

    best_recovered_E_GNN = best_recovered_E_CNN = recovered_EhMP
    best_verified_E_GNN = best_verified_E_CNN = verified_list_EhMP

    if np.count_nonzero(verified_list_EhMP == 0) != 0:
        best_recovered_E_GNN, _, best_verified_E_GNN, _, _ = E_hMP_guide_model(
            fault_model_GNN, H, recovered_EhMP, r, verified_list_EhMP
        )
        best_recovered_E_CNN, _, best_verified_E_CNN, _, _ = E_hMP_guide_model(
            fault_model_CNN, H, recovered_EhMP, r, verified_list_EhMP
        )

    # --- M-hMP, plain and neural-guided ---
    recovered_MhMP, verified_list_MhMP = EhMP.MhMP(
        H, Y_dec, r
    )

    best_recovered_M_GNN = best_recovered_M_CNN = recovered_MhMP
    best_verified_M_GNN = best_verified_M_CNN = verified_list_MhMP

    if np.count_nonzero(verified_list_MhMP == 0) != 0:
        best_recovered_M_GNN, _, best_verified_M_GNN, _, _ = M_hMP_guide_model(
            fault_model_GNN, H, recovered_MhMP, r, verified_list_MhMP
        )
        best_recovered_M_CNN, _, best_verified_M_CNN, _, _ = M_hMP_guide_model(
            fault_model_CNN, H, recovered_MhMP, r, verified_list_MhMP
        )

    # --- VSD. This corrects Y in place, so it must run after the hMP decoders
    # and before the VSD-guided variants, which reuse Y. ---
    decoded_VSD_PV = vm.VSD_partially_verified(H, Y)
    recovered_VSD_PV = decoded_VSD_PV[0]
    verified_list_VSD_PV = decoded_VSD_PV[2]
    S_Binary = decoded_VSD_PV[3]

    _, prob_GNN = fl.predict_fault_positions_any(
        fault_model_GNN, H, Y, verified_list_VSD_PV, S_Binary, use_onnx=False
    )
    recovered_VSD_GNN, _, verified_make_success_VSD_GNN = vm.VSD_model(H, Y, prob_GNN, 1)

    _, prob_CNN = fl.predict_fault_positions_any(
        fault_model_CNN, H, Y, verified_list_VSD_PV, S_Binary, use_onnx=False
    )
    recovered_VSD_CNN, _, verified_make_success_VSD_CNN = vm.VSD_model(H, Y, prob_CNN, 1)

    # --- Analysis ---
    Error_dec = cc.MatrixBinary_to_MatrixDec(Error.T)
    codeword_dec = cc.MatrixBinary_to_MatrixDec(codeword)

    def analyse(recovered):
        return EhMP.get_decoder_analysis(
            H_sys=H,
            original_codeword_dec=codeword_dec.T,
            error_dec=Error_dec,
            recovered_codeword_dec=recovered,
        )

    init_EhMP, solved_EhMP, unsolved_EhMP, success_EhMP = analyse(recovered_EhMP)
    init_E_CNN, solved_E_CNN, unsolved_E_CNN, success_E_CNN = analyse(best_recovered_E_CNN)
    init_E_GNN, solved_E_GNN, unsolved_E_GNN, success_E_GNN = analyse(best_recovered_E_GNN)
    init_MhMP, solved_MhMP, unsolved_MhMP, success_MhMP = analyse(recovered_MhMP)
    init_M_CNN, solved_M_CNN, unsolved_M_CNN, success_M_CNN = analyse(best_recovered_M_CNN)
    init_M_GNN, solved_M_GNN, unsolved_M_GNN, success_M_GNN = analyse(best_recovered_M_GNN)
    init_VSD_PV, solved_VSD_PV, unsolved_VSD_PV, success_VSD_PV = analyse(recovered_VSD_PV)
    init_VSD_CNN, solved_VSD_CNN, unsolved_VSD_CNN, success_VSD_CNN = analyse(recovered_VSD_CNN)
    init_VSD_GNN, solved_VSD_GNN, unsolved_VSD_GNN, success_VSD_GNN = analyse(recovered_VSD_GNN)

    return {
        "n": n,
        "k": k,
        "symbol_size": r,
        "prop_error": SER,
        "Number of Error Symbol": int(np.sum(Error_Marker)),
        "Number of Error Bits": int(np.sum(Error)),

        "success_EhMP": bool(success_EhMP),
        "success_E_hMP_guid_CNN": bool(success_E_CNN),
        "success_E_hMP_guid_GNN": bool(success_E_GNN),
        "success_MhMP": bool(success_MhMP),
        "success_M_hMP_guid_CNN": bool(success_M_CNN),
        "success_M_hMP_guid_GNN": bool(success_M_GNN),
        "success_VSD_PV": bool(success_VSD_PV),
        "success_VSD_guid_CNN": bool(success_VSD_CNN),
        "success_VSD_guid_GNN": bool(success_VSD_GNN),

        "initial_errors_EhMP": _error_dict_to_str(init_EhMP),
        "solved_errors_EhMP": _error_dict_to_str(solved_EhMP),
        "unsolved_errors_EhMP": _error_dict_to_str(unsolved_EhMP),
        "verified_list_EhMP": str(verified_list_EhMP),

        "initial_errors_E_hMP_guid_CNN": _error_dict_to_str(init_E_CNN),
        "solved_errors_E_hMP_guid_CNN": _error_dict_to_str(solved_E_CNN),
        "unsolved_errors_E_hMP_guid_CNN": _error_dict_to_str(unsolved_E_CNN),
        "verified_list_E_hMP_guid_CNN": str(best_verified_E_CNN),

        "initial_errors_E_hMP_guid_GNN": _error_dict_to_str(init_E_GNN),
        "solved_errors_E_hMP_guid_GNN": _error_dict_to_str(solved_E_GNN),
        "unsolved_errors_E_hMP_guid_GNN": _error_dict_to_str(unsolved_E_GNN),
        "verified_list_E_hMP_guid_GNN": str(best_verified_E_GNN),

        "initial_errors_MhMP": _error_dict_to_str(init_MhMP),
        "solved_errors_MhMP": _error_dict_to_str(solved_MhMP),
        "unsolved_errors_MhMP": _error_dict_to_str(unsolved_MhMP),
        "verified_list_MhMP": str(verified_list_MhMP),

        "initial_errors_M_hMP_guid_CNN": _error_dict_to_str(init_M_CNN),
        "solved_errors_M_hMP_guid_CNN": _error_dict_to_str(solved_M_CNN),
        "unsolved_errors_M_hMP_guid_CNN": _error_dict_to_str(unsolved_M_CNN),
        "verified_list_M_hMP_guid_CNN": str(best_verified_M_CNN),

        "initial_errors_M_hMP_guid_GNN": _error_dict_to_str(init_M_GNN),
        "solved_errors_M_hMP_guid_GNN": _error_dict_to_str(solved_M_GNN),
        "unsolved_errors_M_hMP_guid_GNN": _error_dict_to_str(unsolved_M_GNN),
        "verified_list_M_hMP_guid_GNN": str(best_verified_M_GNN),

        "initial_errors_VSD_PV": _error_dict_to_str(init_VSD_PV),
        "solved_errors_VSD_PV": _error_dict_to_str(solved_VSD_PV),
        "unsolved_errors_VSD_PV": _error_dict_to_str(unsolved_VSD_PV),
        "verified_list_VSD_PV": str(verified_list_VSD_PV),

        "initial_errors_VSD_guid_CNN": _error_dict_to_str(init_VSD_CNN),
        "solved_errors_VSD_guid_CNN": _error_dict_to_str(solved_VSD_CNN),
        "unsolved_errors_VSD_guid_CNN": _error_dict_to_str(unsolved_VSD_CNN),
        "verified_list_VSD_guid_CNN": str(verified_make_success_VSD_CNN),

        "initial_errors_VSD_guid_GNN": _error_dict_to_str(init_VSD_GNN),
        "solved_errors_VSD_guid_GNN": _error_dict_to_str(solved_VSD_GNN),
        "unsolved_errors_VSD_guid_GNN": _error_dict_to_str(unsolved_VSD_GNN),
        "verified_list_VSD_guid_GNN": str(verified_make_success_VSD_GNN),
    }


def main():
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n, k, r = CODE_N, CODE_K, BITS_PER_SYMBOL
    G, H = EhMP.BCH_sys(n, k)
    print("G shape:", G.shape, "H shape:", H.shape)

    fault_model_CNN = fl.load_fault_model(
        os.path.join(MODEL_DIR, MODEL_FILE_CNN), r, "cnn"
    )
    fault_model_GNN = fl.load_fault_model(
        os.path.join(MODEL_DIR, MODEL_FILE_GNN), r, "gnn"
    )

    print("NONZERO CODEWORD" if NONZERO else "ZERO CODEWORD")

    for SER in tqdm(SER_VALUES, desc="Overall SER Progress", position=0, leave=True):
        # Derive a per-SER seed so each operating point is independently
        # reproducible regardless of the order they are run in.
        ser_seed = int(
            np.random.SeedSequence(
                [SIMULATION_SEED, int(round(float(SER) * 1_000_000))]
            ).generate_state(1)[0]
        )
        _seed_everything(ser_seed)

        csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_TEMPLATE.format(ser=SER))
        # Start fresh: never append to a file left behind by a crashed run.
        if os.path.exists(csv_path):
            os.remove(csv_path)

        raw_data_list = []
        for _ in tqdm(
            range(NUM_TRIALS),
            miniters=10_000,
            mininterval=10.0,
            desc=f"Processing SER: {SER:.2f}",
            position=1,
            leave=False,
        ):
            raw_data_list.append(
                run_one_trial(H, n, k, r, SER, fault_model_CNN, fault_model_GNN)
            )

            if len(raw_data_list) >= CHUNK_SIZE:
                _flush(raw_data_list, csv_path)
                raw_data_list = []  # release the buffer back to the allocator

        _flush(raw_data_list, csv_path)
        print(f"\nSaved raw data for SER={SER:.2f} to: {csv_path}")

    print(f"run time = {time.time() - start_time:.4f} sec")


if __name__ == "__main__":
    main()
