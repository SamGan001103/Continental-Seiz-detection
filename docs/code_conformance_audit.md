# Code conformance audit — our inference path against the authors' code and the paper

*Performed 2026-08-11 against the published PDF (Yang et al., **Continental generalization of a
human-in-the-loop AI system for clinical seizure recognition**, Expert Syst. Appl. 207:118083,
2022) and the original source files in this repository. No original file was modified; the audit
is read-only and every claim below cites a line.*

## Scope and method

Three things were compared, in this order:

1. **Our code against the authors' code.** `models/convlstm_tf2.py` and `gui/io/infer.py` against
   `models/deep_conv_lstm.py` and `utils/ICA_load_data_elec.py`.
2. **Our code against the paper.** §2.3 (pre-processing) and §2.4.1 (model structure).
3. **The authors' code against their own paper**, where the two disagree — because several of our
   "deviations from the paper" turn out to be faithful reproductions of the authors' code, which
   is the correct thing to reproduce: the released weights were fitted by that code, not by the
   paper's prose.

`models/deep_conv_lstm.py` and `utils/ICA_load_data_elec.py` each have exactly one commit in this
repository — the original import. They have never been edited here.

---

## 1. Model — exact match

`models/convlstm_tf2.py::build_convlstm` against `models/deep_conv_lstm.py::ConvLstmNet.setup`:

| layer | authors' code | ours | same |
|---|---|---|---|
| BatchNormalization | `axis=2, name='normal1'` | identical | yes |
| convlstm1 | 16 filters, `kernel_size=(X_train_shape[2], 3)`, `padding='valid'`, `strides=(1,2)`, `tanh`, `dropout=0.0`, `recurrent_dropout=0.0`, `return_sequences=True` | identical, with `kernel_size=(n_electrodes, 3)` | yes |
| convlstm2 | 32 filters, `(1,3)`, valid, `(1,2)`, tanh, 0.0/0.0, `return_sequences=True` | identical | yes |
| convlstm3 | 64 filters, `(1,3)`, valid, `(1,2)`, tanh, 0.0/0.0, **`return_sequences=False`** | identical | yes |
| Flatten → Dropout | `Dropout(0.5)` | identical | yes |
| dens1 | `Dense(256, activation='sigmoid')` | identical | yes |
| Dropout → dens2 | `Dropout(0.5)`, `Dense(nb_classes)` | identical | yes |
| temperature | `Lambda(lambda x: x / temp)` with `temp = 1.0` | `Lambda(lambda t: t / 1.0)` | yes |
| output | `Activation('softmax')` | identical | yes |

`X_train_shape[2]` is the electrode axis of `(None, 23, 19, 125, 1)`, i.e. 19 — so the first
kernel is `(19, 3)` in both.

**Parameter count 384,846** on all three platforms, and the bundled weights hash matches
`eval_config.WEIGHTS_SHA256`. A wrong layer order or size would fail `load_weights` outright;
a wrong *activation* would not, which is why the table above is checked field by field rather
than inferred from the parameter count.

`ConvLstmNetDeep` in the same file is a different, deeper network (4 ConvLSTM blocks, ELU, 1024/256
dense) and is **not** what the released weights belong to. It is not used.

## 2. Short-time Fourier transform — exact match

`gui/io/infer.py::_calc_stft` against `utils/ICA_load_data_elec.py:26-64::calc_stft`:

| step | authors' code | ours |
|---|---|---|
| orientation | `s = s_.transpose()` | identical |
| transform | `stft.spectrogram(s, framelength=250, centered=False)` | identical |
| 2-D guard | `if stft_data.ndim == 2: expand_dims(-1)` | identical |
| axis order | `np.transpose(…, (1, 2, 0))` | identical |
| magnitude | `np.abs(…) + 1e-6` | identical |
| **DC removal** | `stft_data[:,:,1:]` — **before** the log | identical |
| log | `np.log10(…)` | identical |
| floor | `indices = np.where(<= 0); stft_data[indices] = 0` | `d[d <= 0] = 0` (equivalent) |
| shape | `reshape(-1, s0, s1, s2)` | identical |

The ordering matters and is easy to get wrong: taking the log before dropping the DC bin, or
flooring before the log, both produce a plausible spectrogram with different numbers. Neither
happens.

## 3. Channel selection — exact match

`utils/ICA_load_data_elec.py:114` fixes the montage as

```
['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
```

`gui/io/edf.py::CHANNELS_19` is the same list **in the same order**, verified by comparison rather
than by eye. Order is load-bearing: the ConvLSTM's first kernel spans the whole electrode axis, so
a permuted montage would silently feed the trained weights a transposed spatial map.

## 4. Per-window guards — same, in the same order

| | authors' code | ours |
|---|---|---|
| interruption check before ICA | `detect_interupted_data(s.transpose(), 250)` at :120 | `detect_interupted_data(seg.transpose(), fs)` |
| skip on failed ICA | `if ica_filt_s is None: continue` at :133 | `skip_code[i] = SKIP_ICA_FAILED` |
| ICA call | `ica_arti_remove(s, 250, chs)` at :131 | `ica_arti_remove(seg, fs, CHANNELS_19)` |

We pass `fs` where the authors hard-code `250`. `gui/io/edf.py::load_edf_19ch` resamples every
recording to 250 Hz before this point, so the two are the same value; the difference is defensive,
not behavioural.

## 5. Against the paper

| paper (§2.3, §2.4.1) | status |
|---|---|
| "split the EEG signals into 12-second segments" | matches — `SEGMENT_S = 12` |
| "decompose the signal into 19 independent components" | matches — `ICA(n_components=19)` |
| eye-movement sources identified "from two EEG channels, namely 'FP1' and 'FP2'" | matches — `find_bads_eog(ch_name='Fp1' / 'Fp2')` |
| "STFT … window length of 250 (or 1 s) and 50 % overlapping" | matches — 23 frames from 3000 samples is only possible at hop 125 |
| "remove the DC component" | matches |
| "shape will become (n × 23 × 125)" | matches — `(1, 23, 19, 125)` |
| three ConvLSTM blocks, 16/32/64 filters, `(1×2)` strides, `(1×3)` kernels after the first | matches |
| dense layers "with sigmoid activation and output sizes of 256 and 2" | matches |
| dropout 0.5 on the fully connected layers | matches |
| "Python 3.6 … MNE v0.20" | **deviation** — 0.19.2; measured 0.000000 difference, disclosed |
| "Keras 2.0 and Tensorflow 1.4.0" | **deviation** — measured 6.8 × 10⁻⁹, disclosed; see `docs/portability.md` for why Keras **2** must be selected on modern TensorFlow |

## 6. Findings

### 6.1 The authors' code settles the channel question the paper leaves open

`utils/ICA_load_data_elec.py` asserts two different shapes:

```
:157   assert prep_s.shape == (1, 2*segement-1, 19, 125)     # get_ref_train_df_TUH
:285   assert prep_s.shape == (1, 2*segement-1, 20, 125)     # get_ref_df_RPA
```

and the RPAH path reaches 20 by concatenating a separately-transformed ECG channel after the STFT
(`:277-279`). So **TUH ran 19 channels and RPAH ran 20** — two different models.

The paper never says this. §2.3 says 19 components, Fig. 4 shows `23 × 19 × 125`, and ECG appears
only in the EPILEPSIAE table caption and one patient anecdote. Earlier revisions of
`docs/paper_comparison.md` asserted "20-channel model" as though it were published; it is not, and
that has been corrected.

The consequence is favourable and worth stating plainly: **the released 19-channel weights are the
TUH model, so our reproduction is the right comparison for TUH**, and the RPAH headline figures
belong to a model that is not in this repository.

### 6.2 Two montage files the loader needs are missing

`:97` reads TUH through `params_TUH_ECG.txt` and `:200` reads RPAH through
`params_RPA_addECG.txt`. **Neither file exists in this repository.** The only montage files
shipped are `params_04_19.txt`, `params_common_electrodes.txt` (what our GUI uses) and
`params_final_2_channels.txt`.

`utils/ICA_load_data_elec.py` therefore cannot be executed as-is. This does not affect inference —
nothing in the GUI path imports it — but it does mean the training-feature generation is not
reproducible from this repository alone, which is worth knowing before anyone plans to retrain.

### 6.3 The resample condition differs, with no effect on this corpus

| | condition | behaviour below 250 Hz |
|---|---|---|
| authors | `if fsamp > 250` (`:108`) | left at the native rate |
| ours | `if fs != TARGET_FS` | resampled up to 250 |

The authors' `window_len = 250 * segement` is a fixed sample count, so a sub-250 Hz recording
would yield a window of the wrong *duration* under their condition. Measured across 120
recordings of this corpus: **250 Hz (2 files) and 256 Hz (118 files)** — nothing below 250, so the
branch never fires and the two behave identically here. Recorded because it would matter for a
corpus that included lower rates.

### 6.4 The paper contradicts itself on the dense-layer count

§2.4.1 says "three ConvLSTM blocks … followed by three fully connected layers", then two sentences
later "**Two** fully connected layers follow the three ConvLSTM blocks … output sizes of 256 and
2". Fig. 4 shows `896 → 256 → 2`. We follow Fig. 4 and the authors' code, which agree with each
other: two dense layers.

## 7. Deviations already known, and why they are not defects

These are pinned as deviations by `tests/test_paper_conformance.py`, with the paper's wording in
the assertion messages, so that changing one is deliberate rather than accidental:

1. **Only the top EOG component per channel is removed.** The paper says "We removed those
   independent sources" — plural. The authors' code appends `eog_indices1[0]` and
   `eog_indices2[0]` only, and **our code matches the authors' code**. The deviation is theirs,
   and reproducing it is correct: the released weights were fitted to features generated this way.
2. **A 0.1 Hz high-pass before ICA** that the paper never mentions.
3. **EOG threshold 2.0**, where the paper specifies none and MNE's default is 3.0.
4. **Raw pass-through when no EOG component is found** — `return data`, the unfiltered input.
5. MNE 0.19.2 against the paper's 0.20.
6. Keras/TensorFlow versions, measured at 6.8 × 10⁻⁹.

Items 1–4 are all in `utils/preprocessing.py::ica_arti_remove`, which is the authors' file and is
imported unchanged by both their training loader and our inference path. That shared import is
what makes the reproduction argument hold, and `tests/test_paper_conformance.py` asserts it
directly.

## 8. Evaluation protocol

The authors' development setting steps by 12 seconds — non-overlapping — and anchors each window
at the labelled interval's own start (`:138-147`, `:117`). Our GUI uses `STEP_S = 6` for review,
which is the right choice for a reviewer but is not the paper's evaluation grid.

This is already handled: `docs/RESULTS.md` carries the non-overlapping variant, anchored the
authors' way, at **AUC 0.88** — reported separately and never mixed with the overlapping figure.

## Conclusion

Every computation on the inference path — montage, interruption guard, ICA, STFT, model graph,
softmax — is a faithful reproduction of the authors' code, and the authors' code matches their
paper everywhere except the four pre-processing details listed in §7, which are theirs and are
disclosed as such.

Nothing found in this audit changes a reported number.
