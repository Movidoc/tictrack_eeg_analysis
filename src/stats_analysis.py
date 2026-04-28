# ------------------------------------------ #
# Function: Statistical analysis 
# Author: Martyna
# ----------------------------------------- #

import numpy as np
import mne 
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
from mne.stats import fdr_correction


"""
Correcting for ROI with Bonferonni correction or FDR 
"""

def cluster_stats(X, threshold = None, n_permutations=1024, tail=0, correction = "FDR"):
    """
    Run permutation cluster 1-sample test for each ROI.
    X: dict of {roi_name: (n_epochs, n_freqs, n_times)}
    Returns dict of {roi_name: (T_obs, clusters, cluster_pv, H0)}
    """
    cluster_results = {}
    n_rois = len(X) 
    alpha = 0.05 
    print(f"Length of n_rois {n_rois}")
    #alpha_corrected = 0.05 / n_rois  # Bonferroni correction
    for roi_name, X_roi in X.items():
        print(f"  Running cluster test for {roi_name}...")
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X_roi,
            threshold=threshold,
            n_permutations=n_permutations,
            tail=tail,
            verbose=False
        )
        
        # ------ Correcting for multiple comparisons -----
        if correction == 'FDR':
            if len(cluster_pv) > 0:
                reject, pvals_corrected = fdr_correction(cluster_pv, alpha = alpha, method = 'indep')
            else:
                reject, pvals_corrected = [], []

        if correction == None:
            reject = cluster_pv < alpha
            pvals_corrected = cluster_pv
                
        cluster_results[roi_name] = {
            "T_obs": T_obs,
            "clusters": clusters,
            "cluster_pv": cluster_pv,
            "H0": H0,
            "reject": reject,
            "pvals_corrected" : pvals_corrected,
        }


        # Print significant clusters
        for cluster, pval_corr, signif in zip(clusters, pvals_corrected, reject ):
            if signif:
                print(f"  [{roi_name}] Significant cluster: p={pval_corr:.4f} < alpha")

    return cluster_results

def plot_cluster_results_og(roi_tfr, cluster_results, freqs, times, n, phase, tic, sub):
    """
    Plot TFR with significant clusters highlighted.
    """
    roi_names = list(roi_tfr.keys())
    n_rois = len(roi_names)
    alpha_corrected = 0.05 / n_rois
    alpha = 0.05

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"{sub} | {phase} | {tic} | n={n}", fontsize=18, fontweight="bold")

    for i, roi_name in enumerate(roi_names):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        T_obs      = cluster_results[roi_name]["T_obs"]
        clusters   = cluster_results[roi_name]["clusters"]
        cluster_pv = cluster_results[roi_name]["cluster_pv"]

        # NaN where not significant
        reject = cluster_results[roi_name]["reject"]
        T_obs_plot = np.full_like(T_obs, np.nan)
        for cluster, signif  in zip(clusters, reject):
            if signif:
                T_obs_plot[cluster] = T_obs[cluster]

        vmax = np.max(np.abs(T_obs))
        vmin = -vmax
        extent = [times[0], times[-1], freqs[0], freqs[-1]]

        # Gray background: full T_obs
        ax.imshow(T_obs, cmap=plt.cm.gray, extent=extent,
                  aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

        # Color overlay: significant clusters only
        im = ax.imshow(T_obs_plot, cmap=plt.cm.RdBu_r, extent=extent,
                       aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

        ax.set_title(f"{roi_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)

    if n_rois < 6:
        for j in range(n_rois, 6):
            fig.delaxes(axs.flatten()[j])

    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.04,
                 label="T statistic")

    return fig



def plot_cluster_results_before(roi_tfr, cluster_results, freqs, times, n, phase, tic, sub):
    """
    Plot actual TFR signal with significant clusters circled/contoured.
    
    roi_tfr: dict of {roi_name: array}
        Expected shape per ROI:
        - (n_epochs, n_freqs, n_times), or
        - (n_freqs, n_times)
    cluster_results: output from cluster_stats() or between_cluster_stats()
    """

    roi_names = list(roi_tfr.keys())
    n_rois = len(roi_names)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"{sub} | {phase} | {tic} | n={n}", fontsize=18, fontweight="bold")

    im = None

    for i, roi_name in enumerate(roi_names):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        # -------------------------------
        # 1. Get actual TFR signal
        # -------------------------------
        signal = roi_tfr[roi_name]

        # If epochs are present, average across epochs
        if signal.ndim == 3:
            signal_plot = signal.mean(axis=0)
        elif signal.ndim == 2:
            signal_plot = signal
        else:
            raise ValueError(
                f"{roi_name}: expected 2D or 3D array, got shape {signal.shape}"
            )

        # Safety check: expected shape is freqs x times
        if signal_plot.shape != (len(freqs), len(times)):
            raise ValueError(
                f"{roi_name}: signal shape {signal_plot.shape} does not match "
                f"(n_freqs, n_times)=({len(freqs)}, {len(times)})"
            )

        # Symmetric color scale around 0
        # ===== CHANGE THE SCALE ==== #
        vmin = -1
        vmax = 1
        # vmax = np.nanmax(np.abs(signal_plot))
        # vmin = -vmax

        extent = [times[0], times[-1], freqs[0], freqs[-1]]

        # -------------------------------
        # 2. Plot actual TFR signal
        # -------------------------------
        im = ax.imshow(
            signal_plot,
            cmap="RdBu_r",
            extent=extent,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )

        # -------------------------------
        # 3. Build significant cluster mask
        # -------------------------------
        clusters = cluster_results[roi_name]["clusters"]
        reject = cluster_results[roi_name]["reject"]

        sig_mask = np.zeros_like(signal_plot, dtype=bool)

        for cluster, signif in zip(clusters, reject):
            if signif:
                sig_mask[cluster] = True

        # -------------------------------
        # 4. Circle/contour significant areas
        # -------------------------------
        if sig_mask.any():
            ax.contour(
                times,
                freqs,
                sig_mask,
                levels=[0.5],
                colors="black",
                linewidths=2
            )

        # Optional: show cluster p-values in title
        n_sig = np.sum(reject)
        ax.set_title(f"{roi_name} | sig clusters: {n_sig}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)

    # Remove empty subplots if fewer than 6 ROIs
    if n_rois < 6:
        for j in range(n_rois, 6):
            fig.delaxes(axs.flatten()[j])

    fig.colorbar(
        im,
        ax=axs,
        orientation="vertical",
        fraction=0.02,
        pad=0.04,
        label="Actual TFR signal"
    )

    return fig

def between_cluster_stats(X1, X2, n_permutations=1024, threshold = None, tail=0, correction = 'FDR'):
    """
    """
    alpha = 0.05
    cluster_results = {}
    for roi in X1.keys():
        if roi not in X2:
            continue 

        X1_roi = X1[roi]
        X2_roi = X2[roi]

        F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(X = [X1_roi, X2_roi], threshold = threshold, n_permutations = n_permutations, tail = tail)

        # ------ Correcting for multiple comparisons -----
        
        if correction == 'FDR':
            if len(cluster_pv) > 0:
                reject, pvals_corrected = fdr_correction(cluster_pv, alpha = alpha, method = 'indep')
            else:
                reject, pvals_corrected = [], []
        if correction == None:
            reject = cluster_pv < alpha
            pvals_corrected = cluster_pv
                
        cluster_results[roi] = {
            "T_obs": F_obs,
            "clusters": clusters,
            "cluster_pv": cluster_pv,
            "H0": H0,
            "reject": reject,
            "pvals_corrected" : pvals_corrected,
        }

         # Print significant clusters
        for cluster, pval_corr, signif in zip(clusters, pvals_corrected, reject ):
            if signif:
                print(f"  [{roi}] Significant cluster: p={pval_corr:.4f} < alpha")

    return cluster_results

# ======== PLOTTING ERP ======== #
def plot_cluster_results(roi_tfr, cluster_results, freqs, times, n, phase, tic, sub,
                         roi_erp=None):
    """
    Plot actual TFR signal with significant clusters contoured.
    If roi_erp is provided, an ERP (mean ± SEM) is plotted beneath each TFR panel.

    Parameters
    ----------
    roi_tfr : dict of {roi_name: array (n_epochs, n_freqs, n_times) or (n_freqs, n_times)}
    cluster_results : output of cluster_stats() or between_cluster_stats()
    freqs : array-like
    times : array-like
    n : int
    phase : str
    tic : str
    sub : str
    roi_erp : dict of {roi_name: array (n_epochs, n_times) or (n_times,)}, optional
        Raw (or baseline-corrected) epoch data averaged across channels within each ROI.
        Shape per ROI: (n_epochs, n_times) → mean ± SEM plotted.
                       (n_times,)          → plotted as-is (no SEM).
    """

    roi_names = list(roi_tfr.keys())
    n_rois = len(roi_names)
    n_cols = 3
    n_tfr_rows = 2  # up to 6 ROIs → 2 rows of TFR panels

    if roi_erp is not None:
        fig, axs = plt.subplots(
            n_tfr_rows * 2, n_cols,
            figsize=(18, 14),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [3, 1, 3, 1]}
        )
    else:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    fig.suptitle(f"{sub} | {phase} | {tic} | n={n}", fontsize=18, fontweight="bold")

    im = None

    for i, roi_name in enumerate(roi_names):
        col = i % n_cols
        tfr_row_block = (i // n_cols) * (2 if roi_erp is not None else 1)

        ax_tfr = axs[tfr_row_block, col]

        # ── TFR signal ─────────────────────────────────────────────────────────
        signal = roi_tfr[roi_name]
        if signal.ndim == 3:
            signal_plot = signal.mean(axis=0)
        elif signal.ndim == 2:
            signal_plot = signal
        else:
            raise ValueError(
                f"{roi_name}: expected 2D or 3D array, got shape {signal.shape}"
            )

        if signal_plot.shape != (len(freqs), len(times)):
            raise ValueError(
                f"{roi_name}: signal shape {signal_plot.shape} does not match "
                f"(n_freqs, n_times)=({len(freqs)}, {len(times)})"
            )
        # ===== CHANGE THE SCALE ==== #
        vmin = -1
        vmax = 1
        # vmax = np.nanmax(np.abs(signal_plot))
        # vmin = -vmax
        extent = [times[0], times[-1], freqs[0], freqs[-1]]

        im = ax_tfr.imshow(
            signal_plot,
            cmap="RdBu_r",
            extent=extent,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )

        # ── Significant cluster contours ───────────────────────────────────────
        clusters = cluster_results[roi_name]["clusters"]
        reject   = cluster_results[roi_name]["reject"]

        sig_mask = np.zeros_like(signal_plot, dtype=bool)
        for cluster, signif in zip(clusters, reject):
            if signif:
                sig_mask[cluster] = True

        if sig_mask.any():
            ax_tfr.contour(
                times, freqs, sig_mask,
                levels=[0.5], colors="black", linewidths=2
            )

        n_sig = np.sum(reject)
        ax_tfr.set_title(f"{roi_name} | sig clusters: {n_sig}")
        ax_tfr.set_xlabel("Time (s)")
        ax_tfr.set_ylabel("Frequency (Hz)")
        ax_tfr.axvline(0, color="k", linestyle="--", linewidth=0.8)

        # ── ERP panel ──────────────────────────────────────────────────────────
        if roi_erp is not None and roi_name in roi_erp:
            ax_erp = axs[tfr_row_block + 1, col]
            _plot_erp_panel(ax_erp, roi_erp[roi_name], times, roi_name)

    # Hide unused axes
    all_axes = axs.flatten()
    n_axes_per_roi = 2 if roi_erp is not None else 1
    used_indices = set()
    for i in range(n_rois):
        col = i % n_cols
        tfr_row_block = (i // n_cols) * n_axes_per_roi
        used_indices.add(tfr_row_block * n_cols + col)
        if roi_erp is not None:
            used_indices.add((tfr_row_block + 1) * n_cols + col)

    for j, ax in enumerate(all_axes):
        if j not in used_indices:
            fig.delaxes(ax)

    if im is not None:
        fig.colorbar(
            im,
            ax=axs,
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
            label="Actual TFR signal"
        )

    return fig


def _plot_erp_panel(ax, erp_data, times, roi_name):
    """
    Plot mean ERP ± SEM on *ax*.

    Parameters
    ----------
    ax       : matplotlib Axes
    erp_data : ndarray, shape (n_epochs, n_times) or (n_times,)
    times    : array-like
    roi_name : str
    """
    if erp_data.ndim == 2:
        erp_mean = erp_data.mean(axis=0)
        erp_sem  = erp_data.std(axis=0) / np.sqrt(erp_data.shape[0])
        ax.fill_between(times, erp_mean - erp_sem, erp_mean + erp_sem,
                        alpha=0.25, color="steelblue", label="±SEM")
    elif erp_data.ndim == 1:
        erp_mean = erp_data
    else:
        raise ValueError(
            f"{roi_name} ERP: expected 1D or 2D array, got shape {erp_data.shape}"
        )

    ax.plot(times, erp_mean, color="steelblue", linewidth=1.2)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{roi_name} – ERP", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

def plot_power_spectrum_per_roi(spectra, freqs, sub_ids, sig_bands=None):
    """
    Plot normalized power spectrum (1-40 Hz) for each ROI on one figure.
    Each line = one subject. Grand average line added on top.

    Parameters
    ----------
    spectra  : dict {roi_name: (n_subjects, n_freqs)}
    freqs    : array-like (n_freqs,)
    sub_ids  : list of str — subject labels for the legend
    """
    BANDS = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 40),
    }
    BAND_COLORS = {
        "delta": "#d0e8ff",
        "theta": "#d0ffe8",
        "alpha": "#fff5d0",
        "beta":  "#ffd0d0",
        "gamma": "#ead0ff",
    }

    roi_names = list(spectra.keys())
    n_rois    = len(roi_names)
    n_cols    = 3
    n_rows    = int(np.ceil(n_rois / n_cols))

    # One color per subject
    cmap      = plt.cm.get_cmap("tab10", len(sub_ids))
    sub_colors = {sub: cmap(i) for i, sub in enumerate(sub_ids)}

    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(6 * n_cols, 4 * n_rows),
                            constrained_layout=True)
    fig.suptitle("Pre-tic power spectrum | PHASE_FREE | expressed",
                 fontsize=16, fontweight="bold")

    axs_flat = axs.flatten() if n_rois > 1 else [axs]

    for i, roi_name in enumerate(roi_names):
        ax = axs_flat[i]
        data = spectra[roi_name]  # (n_subjects, n_freqs)

        # Shade frequency bands
        for band_name, (fmin, fmax) in BANDS.items():
            ax.axvspan(fmin, fmax, color=BAND_COLORS[band_name],
                       alpha=0.4, label=band_name)

        # One line per subject
        for j, sub_id in enumerate(sub_ids):
            ax.plot(freqs, data[j], color=sub_colors[sub_id],
                    linewidth=1.2, alpha=0.8, label=sub_id)



        if sig_bands is not None and roi_name in sig_bands:
            for j, sub_id in enumerate(sub_ids):
                if sub_id not in sig_bands[roi_name]:
                    continue
                for band_name, (fmin, fmax) in BANDS.items():
                    if sig_bands[roi_name][sub_id].get(band_name, False):
                        # Place a small marker at the top of the band
                        band_center = (fmin + fmax) / 2
                        y_pos = 0.92 - (j * 0.06)  # stagger vertically per subject
                        ax.plot(band_center, y_pos,
                                marker="*", color=sub_colors[sub_id],
                                markersize=8, transform=ax.get_xaxis_transform(),
                                clip_on=False)

        # Grand average line
        grand_mean = data.mean(axis=0)
        grand_sem  = data.std(axis=0) / np.sqrt(data.shape[0])
        ax.plot(freqs, grand_mean, color="black",
                linewidth=2.5, label="grand avg", zorder=5)
        ax.fill_between(freqs, grand_mean - grand_sem, grand_mean + grand_sem,
                        color="black", alpha=0.15)

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(roi_name, fontsize=12)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized power (z-score)")
        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_ylim(-1, 1)
        ax.spines[["top", "right"]].set_visible(False)

    # Single legend for subjects + bands on last axis
    handles, labels = axs_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right",
               fontsize=8, ncol=2, framealpha=0.7)

    # Hide unused axes
    for j in range(n_rois, len(axs_flat)):
        fig.delaxes(axs_flat[j])

    return fig