from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



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
