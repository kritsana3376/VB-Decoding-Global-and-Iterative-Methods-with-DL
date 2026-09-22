"""Verified Symbol Decoding (VSD) driven by a neural fault-probability vector.

Classic VSD computes an error-locating vector from the syndrome and can only
decode when the syndrome rank equals the number of located error positions.
When it does not, `VSD_model` asks a neural fault locator which of the
suspected positions are least likely to be in error, optimistically clears
that many of them, and retries. The retry is only accepted if the resulting
syndrome is zero, so a wrong guess fails closed rather than returning a
silently corrupted word.

`VSD_model` does not modify its `Y` argument.

Requires `Channel_Coding` (imported as `cc`) on the import path; the syndrome
reduction, error-locating vector, and binary/decimal conversion all come from
there rather than being duplicated here.
"""

import numpy as np

import Channel_Coding as cc


def gf2_inverse(A):
    """Invert a square binary matrix over GF(2) by Gauss-Jordan elimination.

    Unlike Channel_Coding.Compute_Inverse_Binary_Matrix, this returns without
    raising on a singular input, so callers must only pass full-rank matrices.
    """
    n = A.shape[0]
    A = A.copy() % 2
    I = np.eye(n, dtype=int)

    for i in range(n):
        # Find a pivot for column i.
        if A[i, i] == 0:
            for j in range(i + 1, n):
                if A[j, i] == 1:
                    A[[i, j]] = A[[j, i]]
                    I[[i, j]] = I[[j, i]]
                    break

        # Eliminate column i from every other row.
        for j in range(n):
            if j != i and A[j, i] == 1:
                A[j] = (A[j] + A[i]) % 2
                I[j] = (I[j] + I[i]) % 2

    return I


def flip_top_false_Verified(verified_list, ranked_collect, K):
    """Flip the K most-likely-correct entries of the error-locating vector.

    Callers pass positions the vector currently marks as erroneous (0), so a
    flip promotes them to "assumed correct" (1). The input is not modified.

    NOTE: a negative K silently flips all but the last |K| entries, because
    `ranked_collect[:K]` slices from the end. PreModel can produce a negative
    K when the syndrome rank exceeds the located error count; see PreModel.
    """
    verified_list_flipped = verified_list.copy()
    for i, _ in ranked_collect[:K]:
        verified_list_flipped[i] ^= 1
    return verified_list_flipped


def PreModel(prob, verified_list, Number_Error, S_rank, Matrix_H_col):
    """Propose a corrected error-locating vector using the locator scores.

    VSD needs the number of located error positions to equal the syndrome
    rank. When there are too many located positions, the excess
    `Number_Error - S_rank` of them with the lowest fault probability are
    cleared, which is the minimum change that can make the rank condition
    hold.

    KNOWN ISSUE (behaviour preserved from the original): when
    `S_rank > Number_Error` the count goes negative and
    flip_top_false_Verified then flips nearly every candidate instead of
    none. Guard the call site or clamp to zero if you hit this; it is left
    as-is here so results stay comparable with the published runs.
    """
    ranked_correct = sorted(
        [(i, prob[i]) for i in range(len(prob)) if verified_list[i] == 0],
        key=lambda x: x[1],
    )
    number_predict = int(Number_Error - S_rank)
    return flip_top_false_Verified(verified_list, ranked_correct, number_predict)


def VSD_sub(S_Binary, Index_Rows, Error_Locating_Vector, Y_Binary, H):
    """Solve for the error values at the located positions and apply them.

    Input:
        S_Binary: binary syndrome matrix
        Index_Rows: independent syndrome rows from the Gauss-Jordan reduction
        Error_Locating_Vector: 0 marks a suspected error position
        Y_Binary: received word in binary, corrected in place
        H: parity-check matrix
    Output:
        Y_decode: the corrected word in decimal
        Y_Binary_out: a copy of the corrected word in binary
    """
    S_Sub = S_Binary[Index_Rows, :]
    Position_Error = np.where(Error_Locating_Vector == 0)[0]
    H_Sub = H[np.ix_(Index_Rows, Position_Error)]

    H_Sub_inv = gf2_inverse(H_Sub)
    Error_Binary = np.floor(np.mod(np.dot(H_Sub_inv, S_Sub), 2)).astype(int)

    for index in range(len(Position_Error)):
        Y_Binary[Position_Error[index]] = (
            Y_Binary[Position_Error[index]] ^ Error_Binary[index]
        )

    return cc.MatrixBinary_to_MatrixDec(Y_Binary), Y_Binary.copy()


def VSD_accept_false_verified(H_matrix, Y_Binary):
    """VSD that returns its error-locating vector even when decoding fails.

    Unlike all-or-nothing VSD, when the rank check fails this still reports
    the error-locating vector as the verified list, so downstream neural
    guidance has something to work with, at the cost of possible false
    verifications.

    NOTE: Y_Binary is corrected IN PLACE when decoding succeeds, unlike
    VSD_model which copies. The evaluation loop depends on this: the
    VSD-guided decoders run afterwards and deliberately see the corrected
    word. Do not add a defensive copy without re-running the benchmarks.

    Output:
        decoded word in decimal, flag (0 failed / 1 succeeded),
        verified list, binary syndrome
    """
    S_Binary = np.floor(np.mod(np.dot(H_matrix, Y_Binary), 2)).astype(int)
    S_Gauss, S_rank, Index_Rows = cc.Compute_Gauss_Jordan_Reduction(S_Binary)

    # Always compute the locating vector, and seed verified_list from it.
    Error_Locating_Vector = cc.Compute_Error_Locating_Vector(S_Gauss, Index_Rows, H_matrix)
    verified_list = Error_Locating_Vector.copy()
    Number_Error = np.count_nonzero(Error_Locating_Vector == 0)

    # No errors at all: everything is verified.
    if np.sum(S_Binary) == 0 or Number_Error == 0:
        verified_list = np.ones(np.shape(H_matrix)[1], dtype=int)
        return cc.MatrixBinary_to_MatrixDec(Y_Binary), 1, verified_list, S_Binary

    # Rank check; if it passes, solve for the error values and apply them.
    if S_rank == Number_Error:
        S_Sub = S_Binary[Index_Rows, :]
        Position_Error = np.where(Error_Locating_Vector == 0)[0]
        H_Sub = H_matrix[np.ix_(Index_Rows, Position_Error)]

        H_Sub_inv = cc.Compute_Inverse_Binary_Matrix(H_Sub)
        Error_Binary = np.floor(np.mod(np.dot(H_Sub_inv, S_Sub), 2)).astype(int)

        for index in range(len(Position_Error)):
            Y_Binary[Position_Error[index]] = (
                Y_Binary[Position_Error[index]] ^ Error_Binary[index]
            )

        verified_list = np.ones(np.shape(H_matrix)[1], dtype=int)
        return cc.MatrixBinary_to_MatrixDec(Y_Binary), 1, verified_list, S_Binary

    # Rank check failed: report failure but keep the locating vector, which
    # may contain false verifications.
    return cc.MatrixBinary_to_MatrixDec(Y_Binary), 0, verified_list, S_Binary


def VSD_model(H, Y, prob, dmin):
    """VSD with a neural fallback when the rank condition fails.

    Input:
        H: parity-check matrix
        Y: received word in binary; not modified
        prob: per-symbol fault probability from the CNN/GNN locator
        dmin: minimum distance (accepted for interface compatibility)
    Output:
        Y_decode: decoded codeword in decimal
        flag: 0 = decoding failed, 1 = decoding succeeded
        verified_list_make_success: the error-locating vector that worked,
            or all zeros on failure
    """
    verified_list_make_success = np.zeros(H.shape[1], dtype=np.int8)
    Y_Binary = np.array(Y, dtype=int)

    S_Binary = np.floor(np.mod(np.dot(H, Y_Binary), 2)).astype(int)
    S_Gauss, S_rank, Index_Rows = cc.Compute_Gauss_Jordan_Reduction(S_Binary)
    Error_Locating_Vector = cc.Compute_Error_Locating_Vector(S_Gauss, Index_Rows, H)
    Number_Error = np.count_nonzero(Error_Locating_Vector == 0)

    # Nothing located: the word is already good.
    if Number_Error == 0:
        return cc.MatrixBinary_to_MatrixDec(Y), 1, Error_Locating_Vector

    # Classic VSD: rank matches the located error count, so solve directly.
    if S_rank == Number_Error:
        Y_decode, _ = VSD_sub(S_Binary, Index_Rows, Error_Locating_Vector, Y_Binary, H)
        return Y_decode, 1, Error_Locating_Vector

    # Rank condition failed: let the fault locator prune the candidates.
    New_Error_Locating_Vector = PreModel(
        prob, Error_Locating_Vector, Number_Error, S_rank, np.shape(H)[1]
    )
    Number_Error = np.count_nonzero(New_Error_Locating_Vector == 0)
    if S_rank != Number_Error:
        return cc.MatrixBinary_to_MatrixDec(Y), 0, verified_list_make_success

    Y_decode, Y_Binary_result = VSD_sub(
        S_Binary, Index_Rows, New_Error_Locating_Vector, Y_Binary, H
    )

    # Only accept the guess if it actually zeroes the syndrome.
    S_Binary = np.floor(np.mod(np.dot(H, Y_Binary_result), 2)).astype(int)
    if np.sum(S_Binary) == 0:
        return Y_decode, 1, New_Error_Locating_Vector

    return cc.MatrixBinary_to_MatrixDec(Y), 0, verified_list_make_success
