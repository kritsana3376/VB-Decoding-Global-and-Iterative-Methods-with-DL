"""hMP and E-hMP decoding for systematic BCH codes over a q-ary channel.

This is the single source of truth for the verification-based decoders. It
replaces the former ``EhMP.py`` and ``EhMP_DLh.py``, which were two 1,127-line
copies of the same code differing only in whether the decoders accepted a
starting verified mask. That mask is now the optional ``initial_verified``
argument, so there is one implementation of each decoder.

Terminology used throughout:

    check node       a parity-check equation, i.e. a row of H
    check node value the XOR of the symbols a row of H touches (the syndrome
                     symbol for that row); zero means the equation is satisfied
    verified         a symbol known to be error-free; the mask is a binary
                     vector of length n
    zero-sum group   a set of 3 or 4 check nodes whose values XOR to zero

Requires ``Channel_Coding`` (imported as ``cc``) on the import path.
"""

from itertools import combinations

import galois
import numpy as np

import Channel_Coding as cc


# ---------------------------------------------------------------------------
# Code construction
# ---------------------------------------------------------------------------

def front_I_to_back_I(H_sys):
    """Convert between the [I|P] and [P|I] systematic forms.

    G has shape (k, n) and H has shape (n-k, n).
    """
    n = H_sys.shape[1]
    k = n - H_sys.shape[0]
    identity_size = n - k
    if np.array_equal(H_sys[:, 0:identity_size], np.identity(identity_size)):
        P = H_sys[:, identity_size:]
        G = np.concatenate((P.T, np.identity(k, dtype=int)), axis=1)
    elif np.array_equal(H_sys[:, k:n], np.identity(identity_size)):
        P = H_sys[:, 0:k]
        G = np.concatenate((np.identity(k, dtype=int), P.T), axis=1)
    else:
        raise ValueError("H matrix is not in a recognized systematic form.")
    return G


def BCH_sys(n, k):
    """Return the systematic generator and parity-check matrices of BCH(n, k)."""
    bch = galois.BCH(n, k)
    G = np.array(bch.G, dtype=np.int8)  # galois array -> numpy.ndarray
    H = front_I_to_back_I(G)
    return G, H


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

def create_received_word(n, k, r, H, SER=0.10, H_type="Q|I",
                         nonzero_data=False, nonzero_error=True):
    """Draw a codeword and pass it through the q-ary symbol error channel.

    Two flags control edge cases that the original scripts handled
    inconsistently; pass them explicitly so the choice is visible at the call
    site.

        nonzero_data   resample until the information frame is not all-zero.
                       The evaluation script used True, the collector False.
        nonzero_error  force every symbol flagged by the channel to carry a
                       nonzero r-bit error value. With False, a flagged symbol
                       can draw an all-zero error and so is not actually
                       corrupted, which makes the effective SER lower than the
                       nominal one. The collector used True, the evaluation
                       script False.

    Output:
        Data         : data symbols, shape (k,)
        V_dec        : codeword in decimal, shape (n,)
        V            : codeword in binary, shape (n, r)
        Error        : error matrix, shape (r, n)
        Y            : received word in binary, shape (n, r)
        Y_dec        : received word in decimal, shape (n,)
        Error_Marker : per-symbol channel error flags, shape (n,)
    """
    # 1) Information bits, iid Bernoulli(0.5).
    while True:
        Data_bin = np.random.choice([0, 1], size=(k, r)).astype(np.int8)
        if not nonzero_data or np.sum(Data_bin) > 0:
            break
    Data = cc.MatrixBinary_to_MatrixDec(Data_bin)

    # 2) Systematic encoding.
    V_dec = cc.Encode_Systematic_H(Data, H, H_type)
    V = cc.MatrixDec_to_MatrixBinary(V_dec, r).astype(np.int8)

    # 3) Error generation.
    Error_Marker = cc.q_ary_error_channel(SER, n)
    Error = np.zeros((r, n), dtype=np.int8)
    num_error_symbols = int(np.sum(Error_Marker))
    if num_error_symbols > 0:
        error_values = np.random.choice(
            [0, 1], size=(r, num_error_symbols), p=[0.5, 0.5]
        ).astype(np.int8)

        if nonzero_error:
            zero_columns = np.where(np.sum(error_values, axis=0) == 0)[0]
            while len(zero_columns) > 0:
                error_values[:, zero_columns] = np.random.choice(
                    [0, 1], size=(r, len(zero_columns)), p=[0.5, 0.5]
                ).astype(np.int8)
                zero_columns = np.where(np.sum(error_values, axis=0) == 0)[0]

        Error[:, Error_Marker == 1] = error_values

    # 4) Received word.
    Y = (V + Error.T) % 2
    Y_dec = cc.MatrixBinary_to_MatrixDec(Y)

    return Data, V_dec, V, Error, Y, Y_dec, Error_Marker


# ---------------------------------------------------------------------------
# hMP primitives
# ---------------------------------------------------------------------------

def hMP_verification(H, Y, Previous_Verified=None):
    """Compute check-node values and the verified-symbol mask.

    ``Previous_Verified`` is OR-ed into the result so symbols verified in an
    earlier iteration keep their status.
    """
    Received_Matrix = np.array(Y, dtype=np.uint64)
    Total_Check_Node = cc.Compute_Check_Nodes_Matrix(H, Received_Matrix)
    Current_Verified = cc.Compute_Verified_Symbols_Matrix(H, Total_Check_Node)
    if Previous_Verified is not None and np.any(Previous_Verified):
        Current_Verified = np.bitwise_or(Previous_Verified, Current_Verified)
    return Total_Check_Node, Current_Verified


def gc_verified(H, Total_Check_Node, Verified):
    """Verify extra symbols from check nodes that share an identical value.

    This is the fallback used when the ordinary hMP loop stalls: among rows
    holding the same non-zero check-node value, the symbols outside their
    common support cannot be the ones carrying that error, so they are
    verified.

    Returns the updated mask and whether anything changed.
    """
    output_Verified = Verified.copy()
    unique_vals, counts = np.unique(Total_Check_Node, return_counts=True)
    valid_error_vals = unique_vals[(counts >= 2) & (unique_vals != 0)]
    prev_sum = np.sum(Verified)
    for error_val in valid_error_vals:
        rows_with_val = np.where(Total_Check_Node == error_val)[0]
        common_cols = np.bitwise_and.reduce(H[rows_with_val], axis=0)
        if (common_cols > output_Verified).any():
            union_all_cols = np.bitwise_or.reduce(H[rows_with_val], axis=0)
            output_Verified |= union_all_cols ^ common_cols
    is_changed = np.sum(output_Verified) > prev_sum
    return output_Verified, is_changed


# ---------------------------------------------------------------------------
# Zero-sum check-node equations
# ---------------------------------------------------------------------------

def find_null_ZS34(matrix):
    """Find 3- and 4-row zero-sum groups of check-node equations.

    A zero-sum group is a set of rows whose check-node values XOR to zero.

    Input:
        matrix: binary check-node value matrix
    Output:
        null_ind_ZS3: row-index triples that XOR to zero
        null_ind_ZS4: row-index quadruples that XOR to zero

    This is O(pairs^2) in the number of non-zero check-node rows and dominates
    decoding time at larger n.
    """
    original_weights = matrix.sum(axis=1)
    non_zero_rows = [i for i, non_zero in enumerate(np.any(matrix, axis=1)) if non_zero]
    pairs = list(combinations(non_zero_rows, 2))

    null_ind_ZS3 = []
    null_ind_ZS4 = []
    if not pairs:
        return null_ind_ZS3, null_ind_ZS4

    pair_xors = [matrix[i] ^ matrix[j] for i, j in pairs]
    pair_weights = [int(np.sum(xor)) for xor in pair_xors]

    for n, (i, j) in enumerate(pairs):
        weight = pair_weights[n]
        if weight == 0:
            continue

        # ZS3: a third row, of index greater than both, equal to the pair XOR.
        for row in range(matrix.shape[0]):
            if (original_weights[row] == weight
                    and row > i and row > j
                    and np.sum(pair_xors[n] ^ matrix[row]) == 0):
                null_ind_ZS3.append([i, j, row])

        # ZS4: a second pair, both indices greater than both of ours, with the
        # same XOR. That ordering implies the partner pair comes later in the
        # combination order, so only later pairs need to be scanned.
        for m in range(n + 1, len(pairs)):
            if pair_weights[m] != weight:
                continue
            p, q = pairs[m]
            if min(p, q) > max(i, j) and np.sum(pair_xors[n] ^ pair_xors[m]) == 0:
                null_ind_ZS4.append([i, j, p, q])

    return null_ind_ZS3, null_ind_ZS4


def Verified_by_34null(H, null_ind_ZS3, null_ind_ZS4, input_verified_list):
    """Verify symbols using the 3/4-row zero-sum property.

    Within a group of check-node equations whose values XOR to zero, any
    symbol appearing an odd number of times must itself be error-free, so it
    can be verified.

    Input:
        H: check-node equation matrix
        null_ind_ZS3: 3-row zero-sum groups from find_null_ZS34
        null_ind_ZS4: 4-row zero-sum groups from find_null_ZS34
        input_verified_list: verified mask from the previous step
    Output:
        the verified mask after applying the zero-sum property
    """
    verified_after_nullcom3 = np.zeros(H.shape[1], dtype=np.int8)
    verified_after_nullcom34 = np.zeros(H.shape[1], dtype=np.int8)

    for row_1, row_2, row_3 in null_ind_ZS3:
        # Occurrence count of each symbol across the three equations.
        occurrences = H[row_1] + H[row_2] + H[row_3]
        # Symbols appearing 1 or 3 times (odd) are verified.
        verified_after_nullcom3 |= ((occurrences != 2) & (occurrences != 0)).astype(int)

    for row_1, row_2, row_3, row_4 in null_ind_ZS4:
        occurrences = H[row_1] + H[row_2] + H[row_3] + H[row_4]
        verified_after_nullcom34 |= (
            (occurrences != 4) & (occurrences != 2) & (occurrences != 0)
        ).astype(int)

    verified_after_nullcom3 |= input_verified_list
    verified_after_nullcom34 |= verified_after_nullcom3
    return verified_after_nullcom34


# ---------------------------------------------------------------------------
# Symbol solvers
# ---------------------------------------------------------------------------

def Backsubstitution(H, Y, verified_list, bin_total_check_node):
    """Repeatedly correct check-node equations holding a single unverified symbol.

    This is the classic hMP correction step: it resolves one unverified symbol
    per equation and cannot handle equations with two or more unknowns. The
    check-node values are kept current by XOR-ing out every symbol as it is
    corrected.

    Input:
        H: check-node equation matrix
        Y: binary codeword to correct
        verified_list: verified mask entering this step
        bin_total_check_node: binary check-node values, updated in place
    Output:
        Y_fix: codeword XOR-ed with every error value that was recovered
        solved_index: verified mask including every newly corrected symbol
        bin_total_check_node: the updated check-node values
    """
    Y_fix_space = np.zeros(Y.shape, dtype=object)
    solved_index = np.zeros(len(verified_list), dtype=int)
    solved_index |= verified_list

    while True:
        H_unverified = H * (solved_index == 0)
        row_indices, col_indices = np.where(H_unverified == 1)
        unique_rows, counts = np.unique(row_indices, return_counts=True)
        rows_with_single_one = unique_rows[counts == 1]
        mask = np.isin(row_indices, rows_with_single_one)
        single_col_in_row = np.stack([col_indices[mask], row_indices[mask]], axis=1)
        if len(single_col_in_row) == 0:
            break

        # Keep one (column, row) pair per column, lowest row index first.
        sort_indices = np.lexsort((single_col_in_row[:, 1], single_col_in_row[:, 0]))
        sorted_pairs = single_col_in_row[sort_indices]
        _, first_indices = np.unique(sorted_pairs[:, 0], return_index=True)

        for y_index, solve_row in sorted_pairs[first_indices]:
            error_value = bin_total_check_node[solve_row]
            if sum(Y_fix_space[y_index]) == 0:
                Y_fix_space[y_index] = error_value
            bin_total_check_node[np.where(H[:, y_index] == 1)[0]] ^= error_value
            solved_index[y_index] = 1

    Y_fix = (Y ^ Y_fix_space).astype(object)
    return Y_fix, solved_index, bin_total_check_node


def compress_matrix(matrix_input):
    """Drop all-zero rows and columns.

    Returns the compressed matrix plus the original row and column indices
    that were kept.
    """
    keep_row_indices = np.where(np.any(matrix_input != 0, axis=1))[0]
    keep_col_indices = np.where(np.any(matrix_input != 0, axis=0))[0]
    compressed_matrix = matrix_input[np.ix_(keep_row_indices, keep_col_indices)]
    return compressed_matrix, keep_row_indices, keep_col_indices


def get_independent_matrix_and_indices(matrix_in, kept_rows):
    """Select a maximal set of GF(2)-independent rows by Gaussian elimination."""
    rows, cols = matrix_in.shape
    basis_ints = []
    relative_indices = []
    original_indices = []

    row_ints = []
    for row in range(rows):
        val = 0
        for bit in matrix_in[row]:
            val = (val << 1) | int(bit)
        row_ints.append(val)

    for i, r_val in enumerate(row_ints):
        temp_val = r_val
        for b_val in basis_ints:
            temp_val = min(temp_val, temp_val ^ b_val)
        if temp_val > 0:
            basis_ints.append(temp_val)
            basis_ints.sort(reverse=True)
            relative_indices.append(i)
            original_indices.append(kept_rows[i] if kept_rows is not None else i)
            if len(basis_ints) == cols:
                break

    independent_matrix = matrix_in[relative_indices]
    return independent_matrix, np.array(original_indices, dtype=int)


def find_group_sym_val(Sub_Matrix, Check_Node_Value, kept_cols):
    """Recover error values from equation pairs whose supports differ by one symbol.

    Two verified check-node equations differing in exactly one unverified
    column pin down that column's error value. (A false unverified symbol also
    gets solved here, yielding error value 0, which costs complexity but is
    harmless.)

    Input:
        Sub_Matrix: submatrix built from the verified check-node equations
        Check_Node_Value: binary check-node values for those same rows
        kept_cols: original column index of each submatrix column
    Output:
        {original column index: recovered error value}
    """
    solved_values = {}
    diffs_A = np.bitwise_xor(Sub_Matrix[:, None, :], Sub_Matrix[None, :, :])
    dists = diffs_A.sum(axis=2)
    pairs = np.argwhere(np.triu(dists == 1))
    for r1, r2 in pairs:
        local_var_idx = np.where(diffs_A[r1, r2] == 1)[0][0]
        calculated_vals = np.bitwise_xor(Check_Node_Value[r1], Check_Node_Value[r2])
        original_idx = kept_cols[local_var_idx]
        if original_idx not in solved_values:
            solved_values[original_idx] = calculated_vals
    return solved_values


def update_system_with_results(Y, np_matrix, C, results):
    """Substitute recovered error values back into the system.

    Corrects the affected symbols and refreshes the check-node equation
    matrix, the check-node values, and the verified mask.

    Input:
        Y: codeword to correct
        np_matrix: coefficient matrix (LHS)
        C: check-node value matrix (RHS)
        results: {column index: value array} from find_group_sym_val
    Output:
        solve_matrix: LHS with the solved columns zeroed
        C_update: RHS XOR-ed with the known values
        Y_update: corrected codeword
        solved_list: mask of the columns that were solved
    """
    solve_matrix = np_matrix.copy()
    C_update = C.copy()
    Y_update = Y.copy()
    solved_list = np.zeros(np_matrix.shape[1], dtype=int)
    for col_idx, val_vector in results.items():
        rows_with_var = (np_matrix[:, col_idx] == 1)
        C_update[rows_with_var] ^= val_vector
        solve_matrix[:, col_idx] = 0
        Y_update[col_idx] ^= val_vector
        solved_list[col_idx] = 1
    return solve_matrix, C_update, Y_update, solved_list


def Common_Backsubstitution(H_unverified, Y, C):
    """Solve symbols that a single-unknown pass cannot reach.

    Chains compression, independent-row selection, the differ-by-one-symbol
    search, and the substitution step into one call.

    Input:
        H_unverified: check-node equations restricted to unverified symbols
        Y: codeword to correct
        C: binary check-node values
    Output:
        Y_update: corrected codeword
        solved_list: mask of the symbols solved here
        C_update: refreshed check-node values
    """
    non_solved_list = np.zeros(H_unverified.shape[1], dtype=int)
    comp_matrix, kept_rows, kept_cols = compress_matrix(H_unverified)
    sub_matrix, orig_indices = get_independent_matrix_and_indices(comp_matrix, kept_rows)
    if len(orig_indices) == 0:
        return Y, non_solved_list, C
    results = find_group_sym_val(sub_matrix, C[orig_indices], kept_cols)
    _, C_update, Y_update, solved_list = update_system_with_results(
        Y, H_unverified, C, results
    )
    return Y_update, solved_list, C_update


def solve_loop(H, Y_bin_insolve, verified_list_insolve, bin_total_check_node):
    """Alternate Backsubstitution and Common_Backsubstitution until the codeword settles."""
    Y_bin_before_solve = Y_bin_insolve.copy()
    verified_list_before_solve = verified_list_insolve.copy()
    bin_total_check_node_before_solve = bin_total_check_node.copy()

    while True:
        Y_fix_single, single_solve_verified_list, update_bin_total_check_node = Backsubstitution(
            H, Y_bin_before_solve, verified_list_before_solve,
            bin_total_check_node_before_solve
        )
        single_solve_verified_list |= verified_list_before_solve

        H_unverified = H * (single_solve_verified_list == 0)
        Y_fix_multiple2, solve_multiple2_list, update_C_from_multiple2 = Common_Backsubstitution(
            H_unverified, Y_fix_single, update_bin_total_check_node
        )
        solve_multiple2_list |= single_solve_verified_list

        settled = np.array_equal(Y_fix_single, Y_fix_multiple2)

        Y_bin_before_solve = Y_fix_multiple2
        verified_list_before_solve = solve_multiple2_list
        bin_total_check_node_before_solve = update_C_from_multiple2

        if settled:
            break

    return Y_fix_multiple2, solve_multiple2_list, bin_total_check_node_before_solve.copy()


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def EhMP(H, Y_dec, r, initial_verified=None):
    """E-hMP decoder: verification, single-symbol solving, gc, then zero-sum.

    Each stage restarts the loop as soon as it verifies something new; the
    decoder stops when a full pass adds nothing.

    ``initial_verified`` seeds the verified mask. Passing a mask produced by a
    neural fault locator is what the guided retries in the evaluation script
    do; leaving it None starts from all-unverified, which is the plain
    algebraic decoder. (This argument is what the former
    ``EhMP_DLh`` provided.)
    """
    Y_local = Y_dec.copy()
    k = H.shape[1] - H.shape[0]
    Verified = (np.zeros(H.shape[1], dtype=int) if initial_verified is None
                else initial_verified.copy())

    while True:
        global_prev_verified = np.sum(Verified)

        # Step 1: verification.
        Total_Check_Node, Verified = hMP_verification(H, Y_local, Verified)
        if len(np.where(Verified[:k] == 0)[0]) == 0:
            return Y_local, Verified

        # Step 2: correct symbols with Backsubstitution.
        Y_bin = cc.MatrixDec_to_MatrixBinary(Y_local, r)
        bin_total_check_node = cc.MatrixDec_to_MatrixBinary(Total_Check_Node, r)

        Y_fix, Verified, bin_total_check_node = Backsubstitution(
            H, Y_bin, Verified, bin_total_check_node
        )

        # Convert both the codeword and the check-node values back to decimal.
        Y_local = cc.MatrixBinary_to_MatrixDec(Y_fix)
        Total_Check_Node = cc.MatrixBinary_to_MatrixDec(bin_total_check_node)

        if len(np.where(Verified[:k] == 0)[0]) == 0:
            return Y_local, Verified
        if np.sum(Verified) > global_prev_verified:
            continue

        # Step 3: gc_verified, now fed an up-to-date Total_Check_Node.
        Verified, is_gc_changed = gc_verified(H, Total_Check_Node, Verified)
        if len(np.where(Verified[:k] == 0)[0]) == 0:
            return Y_local, Verified
        if is_gc_changed:
            continue

        # Step 4: outermost zero-sum pass. bin_total_check_node is already
        # current, so no reconversion is needed.
        prev_34_verified = np.sum(Verified)
        group_of_3, group_of_4 = find_null_ZS34(bin_total_check_node)
        Verified = Verified_by_34null(H, group_of_3, group_of_4, Verified)
        if len(np.where(Verified[:k] == 0)[0]) == 0:
            return Y_local, Verified
        if np.sum(Verified) > prev_34_verified:
            continue
        if np.sum(Verified) == global_prev_verified:
            break

    return Y_local, Verified


def MhMP(H, Y_dec, r, initial_verified=None):
    """M-hMP decoder: like the E-hMP loop but solving with the full solve_loop.

    ``initial_verified`` behaves as in EhMP. (This
    argument is what the former ``MhMP_DLh`` provided.)
    """
    Y_local = Y_dec.copy()
    k = H.shape[1] - H.shape[0]
    Verified = (np.zeros(H.shape[1], dtype=int) if initial_verified is None
                else initial_verified.copy())
    bin_total_check_node = None

    while True:
        while True:
            while True:
                prev_verified_count = np.sum(Verified)
                Total_Check_Node, Verified = hMP_verification(H, Y_local, Verified)
                if len(np.where(Verified[:k] == 0)[0]) == 0:
                    return Y_local, Verified

                Y_bin = cc.MatrixDec_to_MatrixBinary(Y_local, r)
                bin_total_check_node = cc.MatrixDec_to_MatrixBinary(Total_Check_Node, r)
                Y_fix, Verified, bin_total_check_node = solve_loop(
                    H, Y_bin, Verified, bin_total_check_node
                )
                Y_local = cc.MatrixBinary_to_MatrixDec(Y_fix)
                Total_Check_Node = cc.MatrixBinary_to_MatrixDec(bin_total_check_node)

                if len(np.where(Verified[:k] == 0)[0]) == 0:
                    return Y_local, Verified
                if np.sum(Verified) == prev_verified_count:
                    break

            Verified, is_changed = gc_verified(H, Total_Check_Node, Verified)
            if len(np.where(Verified[:k] == 0)[0]) == 0:
                return Y_local, Verified
            if is_changed:
                continue
            break

        prev_outer_verified_count = np.sum(Verified)
        group_of_3, group_of_4 = find_null_ZS34(bin_total_check_node)
        Verified = Verified_by_34null(H, group_of_3, group_of_4, Verified)
        if len(np.where(Verified[:k] == 0)[0]) == 0:
            return Y_local, Verified
        if np.sum(Verified) == prev_outer_verified_count:
            break

    return Y_local, Verified



def hMP_gc_solve_easy(H, Y_dec, r):
    """Single-pass E-hMP: group-common, zero-sum verification, one back substitution.

    This is the cheap baseline whose *residual* errors the neural fault
    locators are trained to predict. The training data collector must keep
    using exactly this decoder: switching it to the full `EhMP` loop would
    change every label and invalidate the existing checkpoints.
    """
    GC_result = cc.hMP_Group_Common(H, Y_dec)
    Y_dec_after_GC = GC_result[0]
    Verified = GC_result[2]

    Total_Check_Node = cc.Compute_Check_Nodes_Matrix(H, Y_dec_after_GC)
    bin_total_check_node = cc.MatrixDec_to_MatrixBinary(Total_Check_Node, r)

    group_of_3, group_of_4 = find_null_ZS34(bin_total_check_node)
    verified_after_34 = Verified_by_34null(H, group_of_3, group_of_4, Verified)

    Y_bin_after_GC = cc.MatrixDec_to_MatrixBinary(Y_dec_after_GC, r)
    Y_fix, verified_after_34, bin_total_check_node = Backsubstitution(
        H, Y_bin_after_GC, verified_after_34, bin_total_check_node
    )
    return cc.MatrixBinary_to_MatrixDec(Y_fix), verified_after_34


# ---------------------------------------------------------------------------
# Result analysis
# ---------------------------------------------------------------------------

def get_decoder_analysis(H_sys, original_codeword_dec, error_dec, recovered_codeword_dec):
    """Compare a decoded codeword against the transmitted one.

    Returns the initial errors, the errors the decoder solved, the errors it
    left unsolved, and whether the *data* portion came out clean.
    """
    H_sys = np.array(H_sys)
    n = H_sys.shape[1]
    k = n - H_sys.shape[0]  # number of data symbols
    identity_size = n - k

    original = np.array(original_codeword_dec).flatten()
    error = np.array(error_dec).flatten()
    recovered = np.array(recovered_codeword_dec).flatten()

    # 1. All errors introduced by the channel.
    initial_errors = {idx: error[idx] for idx in np.where(error != 0)[0]}

    # 2. Errors still present after decoding.
    unsolved_errors = {
        idx: recovered[idx]
        for idx in np.where(original != recovered)[0]
    }

    # 3. Success means no residual error inside the data symbols.
    error_left = 0
    if np.array_equal(H_sys[:, 0:identity_size], np.identity(identity_size)):
        for key in unsolved_errors:
            if key >= (n - k):
                error_left += 1
    elif np.array_equal(H_sys[:, k:n], np.identity(identity_size)):
        for key in unsolved_errors:
            if key < k:
                error_left += 1
    else:
        raise ValueError("H matrix is not in a recognized systematic form ([P|I] or [I|P]).")
    decode_success = error_left == 0

    # 4. Errors the decoder repaired.
    solved_error_indices = set(initial_errors.keys()) - set(unsolved_errors.keys())
    solved_errors = {idx: initial_errors[idx] for idx in solved_error_indices}

    return initial_errors, solved_errors, unsolved_errors, decode_success
