from collections import Counter
from typing import Dict, List, Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import AMINO_ACIDS

def plot_aspect_and_top_terms(terms_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """
    plot 1 annotation distribution across GO subontologies (BPO, CCO, MFO)
    and 2 most frequent GO terms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    aspect_counts = terms_df["aspect"].value_counts()
    sns.barplot(
        x=aspect_counts.index,
        y=aspect_counts.values,
        hue=aspect_counts.index,
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("GO Annotations by Sub-Ontology (Aspect)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sub-Ontology Aspect")
    axes[0].set_ylabel("Number of Annotations")
    for idx, val in enumerate(aspect_counts.values):
        axes[0].text(idx, val, f"{val:,}", ha="center", va="bottom", fontsize=10)

    top_terms = terms_df["term"].value_counts().head(top_n)
    sns.barplot(
        x=top_terms.values,
        y=top_terms.index,
        hue=top_terms.index,
        palette="magma",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title(f"Top {top_n} Most Frequent GO Terms", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Annotated Proteins")
    axes[1].set_ylabel("GO Term ID")

    plt.tight_layout()
    return fig


def plot_long_tail_and_labels_per_protein(terms_df: pd.DataFrame) -> plt.Figure:
    """
    plot 1 long tail distribution of GO terms (log-log scale)
    and 2 number of GO labels assigned per protein.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    term_freqs = terms_df["term"].value_counts().values
    axes[0].plot(np.arange(1, len(term_freqs) + 1), term_freqs, color="#1f77b4", lw=2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Long-Tail Distribution of GO Term Frequencies (Log-Log)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("GO Term Rank (log scale)")
    axes[0].set_ylabel("Protein Frequency (log scale)")
    axes[0].grid(True, which="both", ls="--", alpha=0.4)

    labels_per_prot = terms_df.groupby("EntryID")["term"].count()
    clipped = np.clip(labels_per_prot.values, 1, np.percentile(labels_per_prot.values, 99))
    sns.histplot(clipped, bins=40, kde=True, color="#2ca02c", ax=axes[1])
    axes[1].set_title("Distribution of GO Terms per Protein (up to 99th pct)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Assigned GO Terms")
    axes[1].set_ylabel("Number of Proteins")

    plt.tight_layout()
    return fig


def plot_sequence_length_and_aa_composition(
    train_seq_df: pd.DataFrame,
    test_seq_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    plot 1 sequence length distribution comparing train vs test sets
    and 2 amino acid frequency distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    train_lens = np.clip(train_seq_df["length"].values, 0, 2500)
    sns.kdeplot(train_lens, fill=True, label="Train Sequences", color="#1f77b4", ax=axes[0])
    if test_seq_df is not None and not test_seq_df.empty:
        test_lens = np.clip(test_seq_df["length"].values, 0, 2500)
        sns.kdeplot(test_lens, fill=True, label="Test Sequences", color="#ff7f0e", ax=axes[0])
    axes[0].set_title("Protein Sequence Length Distribution (Clipped at 2500 aa)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sequence Length (Amino Acids)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    sample_seqs = train_seq_df["sequence"].sample(
        n=min(10000, len(train_seq_df)), random_state=42
    )
    counts: Counter = Counter()
    for s in sample_seqs:
        counts.update(s)
    total_aa = max(sum(counts[aa] for aa in AMINO_ACIDS), 1)
    aa_freqs = [counts[aa] / total_aa * 100.0 for aa in AMINO_ACIDS]

    sns.barplot(
        x=AMINO_ACIDS,
        y=aa_freqs,
        hue=AMINO_ACIDS,
        palette="coolwarm",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Canonical Amino Acid Composition (%) in Training Set", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Amino Acid (1-Letter Code)")
    axes[1].set_ylabel("Percentage (%)")

    plt.tight_layout()
    return fig


def plot_taxonomy_and_ia(
    tax_df: Optional[pd.DataFrame],
    ia_df: Optional[pd.DataFrame],
    top_n_taxa: int = 15,
) -> plt.Figure:
    """
    plot 1 n most frequent species/taxon IDs in the training set
    and 2 distribution of information accretion (IA) weights.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if tax_df is not None and not tax_df.empty:
        top_taxa = tax_df["taxon_id"].astype(str).value_counts().head(top_n_taxa)
        sns.barplot(
            x=top_taxa.values,
            y=top_taxa.index,
            hue=top_taxa.index,
            palette="crest",
            legend=False,
            ax=axes[0],
        )
        axes[0].set_title(f"Top {top_n_taxa} Organism Taxon IDs in Train Set", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Number of Proteins")
        axes[0].set_ylabel("NCBI Taxon ID")
    else:
        axes[0].text(0.5, 0.5, "Taxonomy file not loaded", ha="center", va="center")

    if ia_df is not None and not ia_df.empty:
        sns.histplot(ia_df["ia"].values, bins=50, kde=True, color="#d62728", ax=axes[1])
        axes[1].set_title("Information Accretion (IA) Weight Distribution", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Information Accretion (IA) Weight")
        axes[1].set_ylabel("Number of GO Terms")
    else:
        axes[1].text(0.5, 0.5, "IA weights file not loaded", ha="center", va="center")

    plt.tight_layout()
    return fig


def compute_fmax(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weights: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float | np.ndarray]:
    """
    Compute the official CAFA maximum F measure (F_max) across decision thresholds.
    Supports both unweighted and IA weighted precision/recall.

    Args:
        y_true: Binary ground truth matrix of shape (n_proteins, n_terms)
        y_prob: Predicted probability matrix of shape (n_proteins, n_terms) in [0, 1]
        weights: Optional 1D array of shape (n_terms,) with IA weights
        thresholds: Optional 1D array of thresholds to sweep (default: 0.01 to 0.99)

    Returns:
        Dictionary with keys: 'fmax', 'best_threshold', 'precision', 'recall',
        'thresholds', 'f_scores', 'precisions', 'recalls'.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.95, 48)

    y_true_f = y_true.astype(np.float32)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(1, -1)
        w = np.maximum(w, 1e-6)
    else:
        w = np.ones((1, y_true.shape[1]), dtype=np.float32)

    weighted_true = y_true_f * w
    true_per_protein = weighted_true.sum(axis=1)
    valid_rc_mask = true_per_protein > 0

    f_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    best_f = -1.0
    best_t = 0.5
    best_pr = 0.0
    best_rc = 0.0

    for t in thresholds:
        pred_bin = (y_prob >= t).astype(np.float32)
        weighted_pred = pred_bin * w
        tp_per_protein = (pred_bin * weighted_true).sum(axis=1)
        pred_per_protein = weighted_pred.sum(axis=1)

        # Precision averaged over proteins with at least one predicted term at threshold t
        valid_pr_mask = pred_per_protein > 0
        if np.any(valid_pr_mask):
            pr = float(np.mean(tp_per_protein[valid_pr_mask] / pred_per_protein[valid_pr_mask]))
        else:
            pr = 0.0

        # Recall averaged over proteins with at least one ground-truth term
        if np.any(valid_rc_mask):
            rc = float(np.mean(tp_per_protein[valid_rc_mask] / true_per_protein[valid_rc_mask]))
        else:
            rc = 0.0

        f = (2.0 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0
        f_scores.append(f)
        precisions.append(pr)
        recalls.append(rc)

        if f > best_f:
            best_f = f
            best_t = float(t)
            best_pr = pr
            best_rc = rc

    return {
        "fmax": float(best_f),
        "best_threshold": best_t,
        "precision": best_pr,
        "recall": best_rc,
        "thresholds": thresholds,
        "f_scores": np.array(f_scores),
        "precisions": np.array(precisions),
        "recalls": np.array(recalls),
    }


def compute_ontology_fmax(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected_terms: List[str],
    term_to_aspect: Dict[str, str],
    ia_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Evaluate F_max and IA weighted F_max separately for each Gene Ontology aspect
    as well as their macro average (official CAFA metric).
    """
    aspects = sorted(set(term_to_aspect.values()))
    rows = []

    for asp in aspects:
        idx = [i for i, term in enumerate(selected_terms) if term_to_aspect.get(term) == asp]
        if not idx:
            continue
        sub_true = y_true[:, idx]
        sub_prob = y_prob[:, idx]
        res_unw = compute_fmax(sub_true, sub_prob, weights=None)

        if ia_map is not None:
            w = np.array([ia_map.get(selected_terms[i], 1.0) for i in idx], dtype=np.float32)
            res_ia = compute_fmax(sub_true, sub_prob, weights=w)
            ia_fmax = res_ia["fmax"]
            ia_thresh = res_ia["best_threshold"]
        else:
            ia_fmax = res_unw["fmax"]
            ia_thresh = res_unw["best_threshold"]

        rows.append(
            {
                "Aspect": asp,
                "Num_Terms": len(idx),
                "Fmax_Unweighted": res_unw["fmax"],
                "Best_Threshold": res_unw["best_threshold"],
                "Precision": res_unw["precision"],
                "Recall": res_unw["recall"],
                "Fmax_IA_Weighted": ia_fmax,
                "IA_Best_Threshold": ia_thresh,
            }
        )

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        mean_row = {
            "Aspect": "MEAN (CAFA Official Style)",
            "Num_Terms": int(df_res["Num_Terms"].sum()),
            "Fmax_Unweighted": float(df_res["Fmax_Unweighted"].mean()),
            "Best_Threshold": float(df_res["Best_Threshold"].mean()),
            "Precision": float(df_res["Precision"].mean()),
            "Recall": float(df_res["Recall"].mean()),
            "Fmax_IA_Weighted": float(df_res["Fmax_IA_Weighted"].mean()),
            "IA_Best_Threshold": float(df_res["IA_Best_Threshold"].mean()),
        }
        df_res = pd.concat([df_res, pd.DataFrame([mean_row])], ignore_index=True)

    return df_res