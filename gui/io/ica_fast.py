"""Find the eye-artefact components once instead of twice.

The finding
-----------
Profiling one 12-second window of the detector's ICA stage:

    find_bads_eog, called twice     68.4 ms
    FastICA fit                     40.7 ms
    0.1 Hz high-pass                20.1 ms
    building the MNE Raw object      0.8 ms

The decomposition everyone assumes is the bottleneck is not the biggest cost.
Identifying the eye-artefact component is, and it costs that much because the
authors' `ica_arti_remove` calls `find_bads_eog` twice — once with Fp1 as the
EOG proxy and once with Fp2.

Inside MNE each of those calls does the same three things: unmix the window
into 19 source time courses, band-pass those sources 1–10 Hz, and correlate
them against the proxy channel. Only the third differs between the two calls.
The unmixing and the source filtering are byte-for-byte the same work, done
twice, 14,400 times per day-long recording. The filter is 4096 taps applied to
19 channels, which is where the milliseconds are.

So this computes the sources and filters them ONCE, then scores them against
each proxy in turn.

Why this is allowed to be faster
--------------------------------
It is not an approximation and not a reimplementation. It calls MNE's own
`_transform_raw`, `filter_data`, `_find_sources` and `_find_outliers`, with
MNE's own parameters, in MNE's own order — simply not repeating the half that
does not depend on which proxy channel is being used. The arithmetic is
identical because it is literally the same functions on the same inputs.

`tests/test_ica_fast.py` holds that to a bitwise standard on real windows, the
same way `gui/io/edf_stream.py` was held. If MNE's internals move, that test
fails rather than the numbers drifting quietly — which is the failure this
project has been bitten by before (docs/known_issues.md section 1).

The authors' `utils/preprocessing.ica_arti_remove` is untouched and remains the
reference. This is an alternative route to the same answer, used where speed
matters, and never a different answer.
"""
import numpy as np

# MNE's own defaults for EOG component scoring, from
# mne/preprocessing/ica.py: find_bads_eog -> _find_bads_ch -> score_sources.
EOG_L_FREQ = 1.0
EOG_H_FREQ = 10.0
# _band_pass_filter's keywords, copied verbatim. If any of these drift from
# MNE's, the results stop matching, which is what the test is for.
_FILTER_KW = dict(
    phase='zero-double',
    filter_length='10s',
    fir_window='hann',
    l_trans_bandwidth=0.5,
    h_trans_bandwidth=0.5,
    fir_design='firwin2',
)


def find_bads_eog_shared(ica, raw, ch_names, threshold=2.0):
    """`ica.find_bads_eog` for several proxy channels, sources computed once.

    Returns ``[(indices, scores), ...]``, one entry per name in `ch_names`, in
    the same order and the same form `find_bads_eog` returns them.
    """
    from mne.filter import filter_data
    from mne.preprocessing.ica import _find_sources
    from mne.preprocessing.bads import _find_outliers

    # The shared half: unmix once, filter once.
    sources = ica._transform_raw(raw, None, None, True)
    sfreq = raw.info['sfreq']
    sources_f = filter_data(sources, sfreq, EOG_L_FREQ, EOG_H_FREQ,
                            verbose=False, **_FILTER_KW)

    out = []
    for ch in ch_names:
        target = _target_for(ica, raw, ch)
        target_f = filter_data(target, sfreq, EOG_L_FREQ, EOG_H_FREQ,
                               verbose=False, **_FILTER_KW)
        scores = _find_sources(sources_f, target_f, 'pearsonr')
        idx = _find_outliers(scores, threshold=threshold)
        out.append((_ranked(idx, scores), scores))
    return out


def _ranked(idx, scores):
    """Order components the way `_find_bads_ch` does: strongest first.

    This is not cosmetic and it is where the first version of this module was
    wrong. `_find_outliers` returns indices in ASCENDING INDEX order;
    `find_bads_eog` re-sorts them by descending absolute correlation before
    returning. On a real window that is [4, 7, 14] against [7, 14, 4] — and
    since `ica_arti_remove` excludes only `eog_indices[0]`, the two orders
    remove different components and the cleaned signal differs by up to 182 uV.

    A reordering that silently changes which artefact is removed is exactly
    the class of defect this project keeps finding, so the sort is reproduced
    here explicitly, including the stable de-duplication that follows it.
    """
    idx = np.asarray(idx)
    if idx.size == 0:
        return []
    order = np.abs(np.asarray(scores)[idx]).argsort()[::-1]
    ranked = idx[order]
    remaining = list(np.unique(ranked))
    out = []
    for i in ranked:
        if i in remaining:
            out.append(i)
            remaining.remove(i)
    return out


def _target_for(ica, raw, ch):
    """The proxy channel's data, exactly as `_check_target` would fetch it."""
    return ica._check_target(ch, raw, None, None, True)


def ica_arti_remove_fast(data, sfreq, chs=None):
    """`utils.preprocessing.ica_arti_remove` with the duplicated work removed.

    Same steps in the same order — 0.1 Hz high-pass, FastICA with
    n_components=19 and random_state=13, the top-1 flagged component from each
    of Fp1 and Fp2 excluded, reconstruct — but the eye-artefact search shares
    its sources between the two proxies.

    Returns the cleaned window in microvolts, or the raw input unchanged when
    no component was flagged, or None if the decomposition raised. Identical
    return contract to the authors' function, deliberately including the
    branch where nothing is flagged and the ORIGINAL, unfiltered data comes
    back.
    """
    from mne.preprocessing import ICA
    from utils.preprocessing import create_mne_raw

    raw = create_mne_raw(data, sfreq, chs)
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=0.1, h_freq=None, verbose=False)

    ica = ICA(n_components=19, random_state=13)
    try:
        ica.fit(filt_raw, verbose=False)
    except Exception:                                        # noqa: BLE001
        return None

    filt_raw.load_data()
    ica.exclude = []
    found = find_bads_eog_shared(ica, filt_raw, ['Fp1', 'Fp2'], threshold=2.0)
    for idx, _scores in found:
        if len(idx) > 0:
            ica.exclude.append(idx[0])

    if len(ica.exclude) > 0:
        reconst_raw = filt_raw.copy()
        reconst_raw.load_data()
        ica.apply(reconst_raw, verbose=False)
        return reconst_raw.get_data() * 1e6
    return data


def _unused():                                               # pragma: no cover
    """Keeps numpy imported for type clarity in the docstrings above."""
    return np
