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
collect_training_data.py   ->  *_train.pkl, *_validation.pkl, *_metadata.pkl
        |
        v
training_CNN_*.ipynb / training_GNN_*.ipynb  ->  *.pth checkpoints
        |
        v
run_all_for_Simulation.py  ->  per-trial raw CSVs
        |
        v
plot_tab_*.py              ->  summary tables
        |
        v
replot_*.py                ->  figures
```

## Layout

| File | Role |
|---|---|
| `Channel_Coding.py` | Third-party module (V. Manthamkarn, Kasetsart University). Symbol/matrix conversion, syndrome and check-node computation, Gauss-Jordan reduction, systematic encoding, the q-ary channel. Kept close to upstream on purpose; see "Third-party code" below. |
| `EhMP.py` | The decoder library: hMP primitives, zero-sum verification, back substitution, the `EhMP` / `MhMP` / `hMP_gc_solve_easy` decoders, BCH construction, the channel model, result analysis. |
| `VSD.py` | Verified Symbol Decoding: classic, false-verification-accepting, and neural-fallback variants. |
| `fault_locator.py` | CNN and GNN architectures, checkpoint loading, feature construction, inference wrappers. |
| `collect_training_data.py` | Generates the training and validation sets. |
| `run_all_for_Simulation.py` | The Monte Carlo evaluation driver. |
| `csv_tools.py` | Shared helpers for reading the raw CSVs: header validation, file discovery, code-parameter lookup, summary writing. |
| `plot_style.py` | Shared figure styling and the decoder colour table. |
| `plot_tab_Pfail.py` | Raw CSVs -> `analysis_summary.csv` (probability of decoding failure). |
| `plot_tab_output_SER_USR.py` | Raw CSVs -> `SER_plot_summary.csv` (D-SER and D-USR). |
| `plot_tab_FV_analysis.py` | Raw CSVs -> verified / false verified / false unverified counts, `.csv` or `.xlsx`. |
| `replot_Propfail_Eng.py` | `analysis_summary.csv` -> Pfail figure. |
| `replot_SER_USR_ENG.py` | `SER_plot_summary.csv` -> D-SER / D-USR figure. |
| `training_CNN_*.ipynb`, `training_GNN_*.ipynb` | Model training with evaluation evidence. |

## Decoders

| Column name | Decoder |
|---|---|
| `EhMP` | E-hMP, verification + back substitution + group-common + zero-sum (algebraic) |
| `E_hMP_guid_CNN` / `E_hMP_guid_GNN` | E-hMP plus CNN/GNN-guided verified-flag flipping |
| `MhMP` | M-hMP, as above but solving with the full common back-substitution loop |
| `M_hMP_guid_CNN` / `M_hMP_guid_GNN` | M-hMP plus CNN/GNN-guided verified-flag flipping |
| `VSD_AFV` | VSD accepting false verifications |
| `VSD_guid_CNN` / `VSD_guid_GNN` | VSD driven by CNN/GNN fault probabilities |

Guided promotions are restricted to the identity block of H = [Q|I], i.e. the
last n-k columns, and capped at `MAX_ERROR` per frame.

`hMP_gc_solve_easy` is a fourth, single-pass decoder. It is not benchmarked;
it is the baseline whose residual errors the locators are trained against.

## Requirements

```bash
pip install -r requirements.txt
```

`onnxruntime` is optional and deliberately not imported anywhere. It is only
needed if you build an `InferenceSession` yourself and hand it to
`fault_locator.predict_fault_positions_onnx`.

The figures request the Tahoma font. If it is not installed, matplotlib warns
and falls back; change `TARGET_FONT` in `plot_style.py` to silence it.

## Running

Every path is configurable through the environment or the command line, so no
file needs editing:

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
`review_evidence_gnn/`.

The notebooks save `*_BCH_31_21_r32.pth`, but `run_all_for_Simulation.py`
loads `models/*_BCH_31_21_r32_Update.pth`. Copy and rename the checkpoints
into `models/`, or change `MODEL_FILE_CNN` / `MODEL_FILE_GNN` in the
simulator.

**3. Evaluate.**

```bash
python run_all_for_Simulation.py
```

The committed configuration is one SER (0.03) at 10,000,000 trials. Edit
`SER_VALUES` and `NUM_TRIALS` for other operating points, and lower
`NUM_TRIALS` for a smoke test first.

**4. Summarise and plot.**

```bash
python plot_tab_Pfail.py results/
python plot_tab_output_SER_USR.py results/
python plot_tab_FV_analysis.py results/ -o FV_analysis_result.xlsx --pct

python replot_Propfail_Eng.py results/analysis_summary.csv
python replot_SER_USR_ENG.py results/SER_plot_summary.csv
```

All five take the path as an argument. The analysis scripts read `n` and `k`
from the CSV, so BCH(31,21) and BCH(31,16) runs both work unchanged. They
skip any file whose first six columns do not match the raw format, which
includes the summary tables they write into the same folder.

## Output

Raw: one CSV per symbol error rate, `p{SER}_raw_data_add_EMhMPVSD_2.csv`. An
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

There is no resume support: a run that dies partway through must be restarted
from the beginning.

## Known issues and gotchas

Preserved rather than fixed, because changing them would change published
numbers. Read before extending the code.

- **The collector and the evaluator do not use the same channel.**
  `EhMP.create_received_word` takes two flags. The collector runs with
  `nonzero_data=False, nonzero_error=True`; the evaluator with
  `nonzero_data=True, nonzero_error=False`. During evaluation a flagged symbol
  can therefore draw an all-zero error and not actually be corrupted. At
  r = 32 that is probability 2^-32 per flagged symbol, numerically
  negligible, but it is a different channel model from the one described for
  the training data.
- **The collector trains against `hMP_gc_solve_easy`, not `EhMP`.** That is a
  single-pass variant, not the full E-hMP loop that gets benchmarked. The
  labels mean "still wrong after one cheap pass". Switching the collector to
  `EhMP.EhMP` would change every label and invalidate the existing
  checkpoints, so it must be a deliberate decision, not a cleanup.
- **`VSD.VSD_accept_false_verified` corrects its `Y` argument in place.** The
  evaluation loop depends on it: the VSD-guided decoders run afterwards and
  deliberately see the corrected word. Do not reorder those calls or add a
  defensive copy without re-running the benchmarks. `VSD.VSD_model` copies and
  does not have this behaviour.
- **`VSD.PreModel` can compute a negative flip count** when the syndrome rank
  exceeds the located error count, which makes `flip_top_false_Verified` slice
  from the end. The subsequent rank check appears to reject the result either
  way, so the outcome should match clamping to zero, but this has not been
  confirmed by execution.
- **D-USR for the VSD-guided decoders is not a final verified mask.** The
  simulator writes their masks as
  `verified_list_VSD_AFV_make_success_VSD_guid_*`, which is the locating
  vector that made VSD succeed. `csv_tools.resolve_verified` picks it up by
  suffix so the column resolves, but it means something different from the
  other decoders' D-USR.
- **`EhMP.find_null_ZS34` is O(pairs squared)** in the number of non-zero
  check-node rows and dominates runtime at larger n. First place to look for
  speed.
- **`VSD.gf2_inverse` does not raise on a singular matrix**, unlike
  `Channel_Coding.Compute_Inverse_Binary_Matrix`. Callers must only pass
  full-rank submatrices; the zero-syndrome check is what catches a bad guess.
- **`hMP_verification` casts Y to uint64**, so it breaks silently for r > 64.
- **`np.identity(k, dtype=int)`** in `front_I_to_back_I` gives int32 on
  Windows with numpy < 2 and int64 otherwise, so H's dtype depends on platform
  and numpy version.

## Third-party code

`Channel_Coding.py` is by Visuttha Manthamkarn (Department of Electrical
Engineering, Kasetsart University) and is shared with other projects. It has
been left structurally intact, including functions this repository never
calls, so it stays comparable with upstream. Only three things were changed:
line endings and indentation normalised, a stray `F` prefix removed from the
module docstring, and two `print(...); exit()` pairs replaced with exceptions
so a singular matrix cannot kill a ten-million-trial run from inside a
library call.

Confirm the licence and attribution terms with the author before publishing
this repository.

## Notes on the code consolidation

The decoder library previously existed as three byte-identical copies:
`EhMP.py`, `EhMP_DLh.py`, and a third pasted inside the evaluation script —
44 functions duplicated three ways. `EhMP_DLh.py` differed from `EhMP.py` in
exactly four lines, namely whether two decoders accepted a starting verified
mask. That mask is now the optional `initial_verified` argument on
`EhMP.EhMP` and `EhMP.MhMP`, and `EhMP_DLh.py` is gone:

```python
# before
EhMP_DLh.EhMP_loop_fast_newsingle_fix_DLh(H, Y, r, DL_Verified)
EhMP_DLh.EhMP_loop_fast_commonback_DLh(H, Y, r, DL_Verified)

# after
EhMP.EhMP(H, Y, r, initial_verified=DL_Verified)
EhMP.MhMP(H, Y, r, initial_verified=DL_Verified)
```

The collector's private `group_of_34_ZeroRow_new`, `Verified_by_34_smart` and
`Solve34_single` were separately-written versions of the same three
algorithms; they were replaced by the library versions after checking the
outputs agree on 1,000 randomized inputs. The notebooks' inlined `BCH_sys`
was replaced by an import from `EhMP`.

`VSD.py` used to carry its own copies of `Compute_Gauss_Jordan_Reduction`,
`Compute_Error_Locating_Vector` and `MatrixBinary_to_MatrixDec`; these were
confirmed AST-identical to `Channel_Coding`'s and now come from there.

The five analysis and plot scripts each had their own copy of the CSV header
checks, the decoder ordering table, the summary writer and the figure styling
block; those now live in `csv_tools.py` and `plot_style.py`.
