# Neural-Guided Decoders for BCH Codes over a q-ary Symbol Error Channel

Research code for a journal paper on using a learned fault locator to rescue
frames that verification-based algebraic decoding (hMP / E-hMP / VSD) cannot
resolve on its own.

The idea: an algebraic decoder stalls when too many symbol positions remain
unverified. A small CNN or GNN, trained to predict which positions are *still*
wrong after a baseline E-hMP pass, ranks the unverified positions. The decoder
then optimistically accepts the least-suspicious ones as correct and retries.
A retry is kept only if the syndrome goes to zero, so a bad guess fails closed
rather than returning a silently corrupted word.

## Pipeline

```
collect_training_data.py  ->  *_train.pkl, *_validation.pkl, *_metadata.pkl
        |
        v
training_CNN_...ipynb  /  training_GNN_...ipynb  ->  *.pth checkpoints
        |
        v
run_all_for_journal.py  ->  per-trial CSV results
```

## Layout

| File | Role |
|---|---|
| `Channel_Coding.py` | **Not included — you must supply this.** Symbol/matrix conversion, syndrome and check-node computation, Gauss-Jordan reduction, systematic encoding, the q-ary channel. Everything else imports it as `cc`. |
| `EhMP.py` | The decoder library: hMP primitives, zero-sum verification, symbol solvers, the E-hMP and M-hMP decoders, BCH construction, the channel model, and result analysis. |
| `VSD.py` | Verified Symbol Decoding: the classic decoder, the false-verification-accepting variant, and the neural-fallback variant. |
| `fault_locator.py` | The CNN and GNN architectures, checkpoint loading, feature construction, and the inference wrappers. |
| `collect_training_data.py` | Generates the training and validation sets. |
| `run_all_for_journal.py` | The Monte Carlo evaluation driver. |
| `training_CNN_*.ipynb`, `training_GNN_*.ipynb` | Model training with evaluation evidence. |

## Decoders compared by the evaluation script

| Column prefix | Decoder |
|---|---|
| `EhMP_lfns` | E-hMP, loop-fast "new single" variant (purely algebraic) |
| `E_hMP_guid_CNN` / `E_hMP_guid_GNN` | E-hMP plus CNN/GNN-guided verified-flag flipping |
| `EhMP_lfcb` | M-hMP, loop-fast "common back" variant (purely algebraic) |
| `M_hMP_guid_CNN` / `M_hMP_guid_GNN` | M-hMP plus CNN/GNN-guided verified-flag flipping |
| `VSD_AFV` | VSD accepting false verifications |
| `VSD_guid_CNN` / `VSD_guid_GNN` | VSD driven by CNN/GNN fault probabilities |

Guided promotions are restricted to the identity block of H = [Q|I], i.e. the
last n-k columns, and capped at `MAX_ERROR` per frame.

## Requirements

```bash
pip install -r requirements.txt
```

`onnxruntime` is optional and deliberately not imported anywhere. It is only
needed if you build an `InferenceSession` yourself and hand it to
`fault_locator.predict_fault_positions_onnx`.

## Running

Every path is configurable through the environment, so no file needs editing:

| Variable | Default | Used by |
|---|---|---|
| `AI_DECODER_DATA_DIR` | `data` | collector, notebooks |
| `AI_DECODER_DATA_TIMESTAMP` | a fixed default | notebooks |
| `AI_DECODER_DATA_PREFIX` | derived from the two above | notebooks |
| `AI_DECODER_MODEL_DIR` | `models` | evaluation |
| `AI_DECODER_OUTPUT_DIR` | `results` | evaluation |

**1. Collect the dataset.**

```bash
python collect_training_data.py
```

Prints the `DATA_PREFIX` the notebooks need. Defaults to 100,000 training and
25,000 validation frames per SER across 16 operating points, so it is not
quick. Frames are hashed so no realisation appears in both splits.

**2. Train the locators.** Run either notebook top to bottom after pointing
`AI_DECODER_DATA_PREFIX` at the collector output. Each writes a `.pth`
checkpoint plus evaluation evidence into `review_evidence_cnn/` or
`review_evidence_gnn/`. Move the checkpoints into `models/`.

**3. Evaluate.**

```bash
python run_all_for_journal.py
```

The default is 10,000,000 trials at a single SER. Lower `NUM_TRIALS` for a
smoke test before committing to a full run.

## Output

One CSV per symbol error rate, `p{SER}_raw_data_add_EMhMPVSD_2.csv`. An
existing file at that path is deleted at the start of a run rather than
appended to, so a crashed run never contaminates the next one. Rows are
buffered and flushed every `CHUNK_SIZE` trials.

Each row holds the code parameters, the number of symbol and bit errors the
channel introduced, a boolean `success_*` per decoder, and per decoder the
initial / solved / unsolved error maps and the final verified list. Success
means no residual error in the **data** symbols.

## Reproducibility

Three disjoint seeds: 2026 (training split), 2027 (validation split), 2028
(evaluation). Each SER derives its own seed from
`SeedSequence([SIMULATION_SEED, round(SER * 1e6)])`, so operating points are
independently reproducible in any order, and numpy, `random` and torch are all
seeded from it.

The notebooks pin `SEED = 42`, disable cuDNN autotuning, and request
deterministic algorithms.

## Known issues and gotchas

These are preserved rather than fixed, because changing them would change
published numbers. Read before extending the code.

- **The collector and the evaluator do not use the same channel.**
  `EhMP.create_received_word` takes two flags. The collector runs with
  `nonzero_data=False, nonzero_error=True`; the evaluator runs with
  `nonzero_data=True, nonzero_error=False`. So during evaluation a symbol the
  channel flagged can draw an all-zero error value and not actually be
  corrupted, making the effective SER lower than the nominal one, while during
  training every flagged symbol really is corrupted. Worth confirming this
  asymmetry is intended before publication.
- **`VSD.VSD_accept_false_verified` corrects its `Y` argument in place.** The
  evaluation loop depends on it: the VSD-guided decoders run afterwards and
  deliberately see the corrected word. Do not reorder those calls or add a
  defensive copy without re-running the benchmarks. `VSD.VSD_model` copies and
  does not have this behaviour.
- **`VSD.PreModel` can compute a negative flip count** when the syndrome rank
  exceeds the located error count. `flip_top_false_Verified` then slices from
  the end and flips nearly every candidate instead of none. Clamp to zero if
  you hit this.
- **`EhMP.find_null_ZS34` is O(pairs squared)** in the number of non-zero
  check-node rows and dominates runtime at larger n. First place to look for
  speed.
- **`VSD.gf2_inverse` does not raise on a singular matrix.** Callers must only
  pass full-rank submatrices.

## Notes on the code consolidation

The decoder library previously existed as three byte-identical copies:
`EhMP.py`, `EhMP_DLh.py`, and a third pasted inside the evaluation script —
44 functions duplicated three ways. `EhMP_DLh.py` differed from `EhMP.py` in
exactly four lines, namely whether two decoders accepted a starting verified
mask.

That mask is now the optional `initial_verified` argument on
`EhMP_loop_fast_newsingle_fix` and `EhMP_loop_fast_commonback`, and
`EhMP_DLh.py` is gone. Any code still doing this:

```python
import EhMP_DLh
EhMP_DLh.EhMP_loop_fast_newsingle_fix_DLh(H, Y, r, DL_Verified)
EhMP_DLh.EhMP_loop_fast_commonback_DLh(H, Y, r, DL_Verified)
```

should become:

```python
import EhMP
EhMP.EhMP_loop_fast_newsingle_fix(H, Y, r, initial_verified=DL_Verified)
EhMP.EhMP_loop_fast_commonback(H, Y, r, initial_verified=DL_Verified)
```

The collector's private `group_of_34_ZeroRow_new`, `Verified_by_34_smart` and
`Solve34_single` were separately-written versions of the same three
algorithms; they were replaced by the library versions after checking the
outputs agree on 1,000 randomized inputs.

`VSD.py` used to carry its own copies of `Compute_Gauss_Jordan_Reduction`,
`Compute_Error_Locating_Vector` and `MatrixBinary_to_MatrixDec`. These now
come from `Channel_Coding`, which is where the rest of the code already got
them. **Verify those three behave identically to the deleted copies**, since
`Channel_Coding.py` was not available when this consolidation was made.
