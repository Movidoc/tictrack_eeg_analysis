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
                print(f"  [{roi_name}] Significant cluster: p={pval_corr:.4f} < alpha (FDR)")

    return cluster_results

def plot_cluster_results(roi_tfr, cluster_results, freqs, times, n, phase, tic, sub):
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

def between_cluster_stats(X1, X2, n_permutations=1024, threshold = None, tail=0, correction = 'FDR'):
    """
    The analysis is done for each person and each ROI, so for each test there are actually 5 comparisons. We add multiple comparison correction to correct for it.
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
                print(f"  [{roi}] Significant cluster: p={pval_corr:.4f} < alpha (FDR)")

    return cluster_results

