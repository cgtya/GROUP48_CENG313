"""
Data loading, parsing, and feature extraction utilities
"""

from pathlib import Path
from typing import Dict, List

# Standard 20 canonical amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

def resolve_data_paths(data_dir: str | Path) -> Dict[str, Path]:
    """
    Resolve file paths whether files are inside
    Train/ and Test/ subdirectories or directly inside `data_dir`.
    """
    base = Path(data_dir)

    def _find(candidates: List[str]) -> Path:
        for rel in candidates:
            p = base / rel
            if p.exists():
                return p
        return base / candidates[0]

    return {
        "train_fasta": _find(["Train/train_sequences.fasta", "train_sequences.fasta"]),
        "train_terms": _find(["Train/train_terms.tsv", "train_terms.tsv"]),
        "train_taxonomy": _find(["Train/train_taxonomy.tsv", "train_taxonomy.tsv"]),
        "go_obo": _find(["Train/go-basic.obo", "go-basic.obo"]),
        "test_fasta": _find(["Test/testsuperset.fasta", "testsuperset.fasta"]),
        "test_taxonomy": _find(["Test/testsuperset-taxon-list.tsv", "testsuperset-taxon-list.tsv"]),
        "ia": _find(["IA.tsv", "IA.txt"]),
        "sample_submission": _find(["sample_submission.tsv"]),
    }
