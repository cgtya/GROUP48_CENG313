"""
Data loading, parsing, and feature extraction utilities
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


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


def normalize_protein_id(header: str) -> str:
    """
    Extract clean UniProt Accession ID from FASTA header.
    Examples:
        '>sp|P20536|UNG_VACCC ...' -> 'P20536'
        '>tr|A0A0A0MRZ9|A0A0A0MRZ9_SPECI ...' -> 'A0A0A0MRZ9'
        '>P20536 description' -> 'P20536'
    """
    token = header.lstrip(">").strip().split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2:
            return parts[1]
        return parts[-1]
    return token


def parse_fasta(
    fasta_path: str | Path,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """
    FASTA parser returning a DataFrame 
    with columns: ['EntryID', 'sequence', 'length'].
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found at: {fasta_path}")

    ids: List[str] = []
    sequences: List[str] = []
    current_id: Optional[str] = None
    current_seq: List[str] = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    ids.append(current_id)
                    sequences.append("".join(current_seq))
                    if max_records is not None and len(ids) >= max_records:
                        current_id = None
                        break
                current_id = normalize_protein_id(line)
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None and (max_records is None or len(ids) < max_records):
            ids.append(current_id)
            sequences.append("".join(current_seq))

    df = pd.DataFrame({"EntryID": ids, "sequence": sequences})
    df["length"] = df["sequence"].str.len()
    return df


def load_train_terms(terms_path: str | Path) -> pd.DataFrame:
    """
    Load ground-truth GO term annotations ('train_terms.tsv').
    Normalizes column names to ['EntryID', 'term', 'aspect'].
    """
    terms_path = Path(terms_path)
    if not terms_path.exists():
        raise FileNotFoundError(f"Terms file not found at: {terms_path}")

    df = pd.read_csv(terms_path, sep="\t")
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if "entry" in low or "protein" in low or low == "id":
            col_map[col] = "EntryID"
        elif "term" in low or "go" in low:
            col_map[col] = "term"
        elif "aspect" in low or "ontology" in low or "namespace" in low:
            col_map[col] = "aspect"
    df = df.rename(columns=col_map)
    if list(df.columns[:3]) != ["EntryID", "term", "aspect"] and len(df.columns) >= 3:
        df.columns = ["EntryID", "term", "aspect"] + list(df.columns[3:])
    return df


def load_taxonomy(taxonomy_path: str | Path) -> pd.DataFrame:
    """
    Load protein taxonomy mapping ('train_taxonomy.tsv' or 'testsuperset-taxon-list.tsv').
    Returns columns: ['EntryID', 'taxon_id'].
    """
    taxonomy_path = Path(taxonomy_path)
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy file not found at: {taxonomy_path}")

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split("\t")

    has_header = any(
         any(k in token.lower() for k in ("entry", "protein", "taxon", "id"))
        for token in first_line
    )
    if has_header:
        df = pd.read_csv(taxonomy_path, sep="\t")
        df = df.iloc[:, :2].copy()
        df.columns = ["EntryID", "taxon_id"]
    else:
        df = pd.read_csv(
            taxonomy_path,
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["EntryID", "taxon_id"],
        )
    df["EntryID"] = df["EntryID"].astype(str)
    return df


def load_ia_weights(ia_path: str | Path) -> pd.DataFrame:
    """
    Load Information Accretion (IA) weights ('IA.tsv' or 'IA.txt').
    Returns columns: ['term', 'ia'].
    """
    ia_path = Path(ia_path)
    if not ia_path.exists():
        raise FileNotFoundError(f"IA file not found at: {ia_path}")

    df = pd.read_csv(
        ia_path,
        sep="\t",
        header=None,
        names=["term", "ia"],
    )
    return df


def parse_go_obo_summary(obo_path: str | Path) -> Dict[str, int]:
    """
    Parse 'go-basic.obo' to count total GO terms, obsolete terms, and 'is_a' relations.
    """
    obo_path = Path(obo_path)
    if not obo_path.exists():
        return {"total_terms": 0, "active_terms": 0, "is_a_edges": 0}

    total_terms = 0
    obsolete_terms = 0
    is_a_edges = 0
    in_term = False

    with open(obo_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "[Term]":
                in_term = True
                total_terms += 1
            elif line.startswith("[") and line.endswith("]"):
                in_term = False
            elif in_term:
                if line.startswith("is_obsolete: true"):
                    obsolete_terms += 1
                elif line.startswith("is_a:"):
                    is_a_edges += 1

    return {
        "total_terms": total_terms,
        "active_terms": total_terms - obsolete_terms,
        "obsolete_terms": obsolete_terms,
        "is_a_edges": is_a_edges,
    }
