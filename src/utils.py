import matplotlib.pyplot as plt
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
