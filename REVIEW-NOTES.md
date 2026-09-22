# REVIEW-NOTES (private; gitignored; never committed)

Draft from Phase 0. File:line refers to the files as found (commit zero).
"Paper" = manuscript revised 2026-09-17. Nothing here has been fixed.

## Code vs manuscript

### High
- Sec 4.2.3 says runs store checkpoints (completed counts and RNG states) so a run can resume. The code has no resume; it deletes any existing CSV at start (run_all_for_journal.py:402-404).
- Sec 4.2.3 says the channel uses a separate unseeded generator for error positions. Error positions come from `Channel_Coding.q_ary_error_channel` (EhMP.py:102), which is missing, so this cannot be checked. If true, the "seeds 2026/2027" claim in Sec 4.2.1 is only partly true for the datasets as well.
- Fig 5(b) reports D-USR for CNN/GNN-assisted VSD. The current analysis script looks for `verified_list_VSD_guid_CNN` (plot_tab_output_SER_USR.py:157,174), but the simulator writes `verified_list_VSD_AFV_make_success_VSD_guid_CNN` (run_all_for_journal.py:364,369). The current scripts cannot produce those D-USR values.

### Medium
- Sec 4.2.1 says training frames are "processed by E-hMP". The collector uses `hMP_gc_solve_easy` (collect_training_data.py:88; EhMP.py:588-609), a single-pass variant, not `EhMP_loop_fast_newsingle_fix`, which is the E-hMP evaluated in Sec 5.
- Algorithm 3 lines 10-17 consider all pairs of check-node equations. `Solve34_multiple2` first keeps only a greedily chosen GF(2)-independent subset of rows (EhMP.py:428), so pairs involving a dropped row are never tested.
- Eq (43) and Sec 4.2.3: error value uniform on nonzero r-bit vectors. The evaluation calls the channel with `nonzero_error=False` (run_all_for_journal.py:231-234), so a flagged symbol can draw the zero error. Probability 2^-32 per flagged symbol at r=32, so numerically negligible, but it is a different model.
- Sec 4.2.3: SER 0.02-0.15, 1e7 trials for 0.02-0.05 and 1e6 for 0.06-0.15. The committed configuration is one SER (0.03) at 1e7 trials (run_all_for_journal.py:48-49); the paper runs required editing the file.
- Data Availability promises checkpoints, seeds/configs and raw Monte Carlo results in the repository. None are in the folder, and the user's instructions gitignore them.
- AI declaration says AI tools were not used to generate the simulation software. README.md:149-184 describes a later consolidation (files merged, functions replaced, three VSD helpers deleted). If a tool did that, and after this reorganization, the declaration may need updating.

### Low
- Eq (24) thresholds with `>`; notebooks use `>=` (CNN/GNN notebook cells `evaluate`, `classifier_row`, frame inspection). No effect with the reported score distribution.
- Algorithm 2 caps the loop at n passes and orders verification zero-check, ZS2, ZS3, ZS4 with correction after each pass. The code has no cap (EhMP.py:490, 549) and orders zero-check, back substitution, ZS2, then ZS3/ZS4.
- Remark 2 / Algorithm 2 verify odd(J) for every ZS2 set. `gc_verified` applies it only when the common support contains an unverified column (EhMP.py:162). Conservative, not wrong.
- Sec 2.5 cites O((n-k)^2 log(n-k)) per zero sum for ZS3/ZS4 [61]. `find_null_ZS34` scans all later pairs for each pair (EhMP.py:199-219), about O(m^4) in m nonzero rows.
- Sec 3.2.1 says the model is consulted only after the algebraic decoder fails. The code runs guided E-hMP/M-hMP whenever any position, parity included, is unverified (run_all_for_journal.py:245,261), and runs VSD-branch inference on every trial (276-284). Outputs are unaffected (retries return early or are rejected); only cost differs.

## Suspected bugs

### High
- Both notebooks call `galois.__version__` in the last cell (CNN .ipynb JSON line 1444, GNN line 1732) but never `import galois`. NameError after the checkpoint is saved, so `run_configuration.json` and `H_used.npy` are never written.

### Medium
- `replot_Propfail_Eng.py:72-76` expects `success_*_sum` / `success_*_count` columns. `plot_tab_Pfail.py:145-167` writes `pfail_*`, `count_success_*`, `count_all_*`. The Pfail figure cannot be regenerated from the current scripts' output.
- `plot_tab_FV_analysis.discover_decoders` (plot_tab_FV_analysis.py:105-125) pairs `verified_list_VSD_AFV_make_success_VSD_guid_*` with no unsolved/initial column. For those two decoders false-verified is always 0 and false-unverified counts every zero.
- `plot_tab_FV_analysis.py:39-40` hardcodes n=31, k=21. Wrong for the (31,16) results in Figs 3(b), 4(b).
- Notebooks save `*_BCH_31_21_r32.pth` in review_evidence_*/; the simulator loads `models/*_BCH_31_21_r32_Update.pth` (run_all_for_journal.py:60-61). An undocumented manual rename is required.

### Low
- `VSD.PreModel` can pass a negative count (VSD.py:86). By my reading the rank check at VSD.py:206 then always fails, so the outcome equals clamping to zero. The existing README overstates this as "flips nearly every candidate". Unverified by execution.
- `get_decoder_analysis` stores the recovered symbol value, not the residual error value, in `unsolved_errors` (EhMP.py:635-638). Downstream scripts use keys only.
- `front_I_to_back_I` is named for H but is called with G (EhMP.py:56); its local `k` is then n-k. Works because the map is symmetric.
- `SER_FROM_FILENAME` expects `p0_03` (plot_tab_FV_analysis.py:43); the simulator writes `p0.03_...` (run_all_for_journal.py:63). Fallback only, since `prop_error` is present.
- Module docstring usage says `python decoder_stats.py` (plot_tab_FV_analysis.py:19-21); argparse prints this as `--help`.

## Numerically questionable steps
- `np.identity(k, dtype=int)` (EhMP.py:43,46) gives int32 on Windows with numpy<2 and int64 otherwise, so H's dtype depends on platform and numpy version. Mixed int32/uint64 arithmetic promotes to float64 in numpy 1.x. Whether Channel_Coding hits this is unknown.
- `hMP_verification` casts Y to uint64 (EhMP.py:137); breaks silently for r > 64.
- `gf2_inverse` returns silently on singular input (VSD.py:23-48). After `PreModel` a wrong guess can make H_sub singular; the zero-syndrome check (VSD.py:214-216) should reject the result.
- `VSD_accept_false_verified` uses `cc.Compute_Inverse_Binary_Matrix` (VSD.py:154) while `VSD_sub` uses the local `gf2_inverse` (VSD.py:107) for the same step.
- `predict_fault_positions_onnx` computes a raw sigmoid (fault_locator.py:213); overflow warnings for large negative logits.
- Float `prop_error` is used as a groupby key across files (plot_tab_Pfail.py:128, plot_tab_output_SER_USR.py:183).

## Duplicated or dead code
- Unused: `predict_fault_positions_onnx`, `predict_fault_positions_rand_gauss` (fault_locator.py:196-240); `VSD_model` arg `dmin`, `PreModel` arg `Matrix_H_col`; CNN notebook `NUM_ITERS`; GNN notebook `H_LOADED`; `train_size, val_size`; `INCLUDE_RAW_SUMS` path.
- `ResidualBlock1D`, `FaultCNN`, `FaultGNN` exist in fault_locator.py and again in each notebook, with different forward signatures (unbatched vs batched).
- Notebook helper functions are duplicated between the CNN and GNN notebooks.
- `read_header`, `check_header`, `write_csv`, `REQUIRED_HEADER_PREFIX` duplicated in plot_tab_Pfail.py and plot_tab_output_SER_USR.py; a third `check_header` in plot_tab_FV_analysis.py.
- Ranking-by-probability comprehension repeated in fault_locator._rank_unverified, VSD.PreModel, run_all.guided_retry, predict_fault_positions_rand_gauss.
- `VSD_accept_false_verified` inlines the body of `VSD_sub`.
- Syndrome expression `np.floor(np.mod(np.dot(H, Y), 2)).astype(int)` repeated in VSD.py.
- Plot styling blocks duplicated across the two replot scripts; several series commented out.

## Performance
- `find_null_ZS34` quadratic in pairs, run every outer iteration (EhMP.py:173-221).
- `Solve34_single` builds object-dtype arrays (EhMP.py:281,307).
- Repeated decimal/binary conversions inside decoder loops (EhMP.py:499-508, 557-563).
- `predict_fault_positions_any` calls `str(model)` (full module repr) on every call (fault_locator.py:246).
- Two CNN/GNN forward passes per trial for VSD even when VSD succeeds (run_all_for_journal.py:276-284).

## Environment and reproducibility
- `Channel_Coding.py` is missing; nothing except fault_locator.py and the plot scripts can be imported.
- requirements.txt is unpinned and lacks `seaborn` (replot_*.py) and `openpyxl` (xlsx output in plot_tab_FV_analysis.py:334).
- Notebooks record Python 3.9.21; no record of numpy/torch/galois versions used for Section 5.
- `torch.load(..., weights_only=False)` (fault_locator.py:145) executes arbitrary code from an untrusted checkpoint.
- No Python environment on this machine except another app's private runtime and WSL Python 3.12 without galois/torch.

## Identifying strings found in code (not edited)
- `C:\Users\user\Desktop\ProjectY4\pre_master\new_journal_run\Add_3121` (plot_tab_FV_analysis.py:351,353; plot_tab_output_SER_USR.py:245; plot_tab_Pfail.py:182).
- `C:\Users\user\Desktop\ProjectY4\pre_master\new_MhMP_patent_3116\SER_plot_summary_updated_3116.csv` (replot_SER_USR_ENG.py:107). "patent" matches the petty-patent references [24], [62].
- `pre_master/new_journal_run/Add_3121/...` (replot_Propfail_Eng.py:133).
- Thai-language comments, docstrings and printed messages in plot_tab_FV_analysis.py, plot_tab_output_SER_USR.py, plot_tab_Pfail.py, replot_Propfail_Eng.py, replot_SER_USR_ENG.py.
- Global git identity `BenzThitikorn <BenzThitikorn@users.noreply.github.com>` would be stamped on every commit.

## Docstring uncertainties
(to be filled during Phase 3)

## Wanted to fix but did not
(to be filled during Phase 3)
