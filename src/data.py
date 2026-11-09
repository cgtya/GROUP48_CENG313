"""
Data loading, parsing, and feature extraction utilities
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# Standard 20 canonical amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

CANONICAL_AA_SET = set(AMINO_ACIDS)
# IUPAC-IUBMB extended / ambiguous amino acid codes
NON_CANONICAL_IUPAC = {
    "U": "Selenocysteine (Rare 21st AA)",
    "O": "Pyrrolysine (Rare 22nd AA)",
    "B": "Ambiguous Asx (Asp D or Asn N)",
    "Z": "Ambiguous Glx (Glu E or Gln Q)",
    "J": "Ambiguous Xle (Leu L or Ile I)",
    "X": "Unknown / Unspecified AA",
}


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

def inspect_non_canonical_amino_acids(
    datasets: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan protein sequences across one or more splits to check for
    noncanonical or invalid/corrupted amino acid characters.

    Returns:
        overview_df: High-level summary per dataset split (total sequences, affected sequences, invalid symbols).
        residue_df : Detailed breakdown of each non-canonical character found, its IUPAC meaning,
                     total occurrences, and number of affected proteins per split.
    """
    overview_rows = []
    residue_rows = []

    for split_name, df in datasets.items():
        total_seqs = len(df)
        total_residues = int(df["length"].sum()) if "length" in df.columns else 0
        affected_seqs = 0
        invalid_seqs = 0
        char_occurrence: Counter = Counter()
        char_seq_count: Counter = Counter()

        for seq in df["sequence"]:
            counts = Counter(seq)
            non_can = [ch for ch in counts if ch not in CANONICAL_AA_SET]
            if non_can:
                affected_seqs += 1
                if any(ch not in NON_CANONICAL_IUPAC for ch in non_can):
                    invalid_seqs += 1
                for ch in non_can:
                    char_occurrence[ch] += counts[ch]
                    char_seq_count[ch] += 1

        overview_rows.append(
            {
                "Split": split_name,
                "Total_Sequences": total_seqs,
                "Sequences_with_NonCanonical_AA": affected_seqs,
                "Affected_Sequence_Pct (%)": round(affected_seqs / max(total_seqs, 1) * 100.0, 4),
                "Corrupted_or_Invalid_Char_Sequences": invalid_seqs,
                "Total_NonCanonical_Residues": sum(char_occurrence.values()),
                "Total_Residues": total_residues,
            }
        )

        for ch in sorted(char_occurrence.keys()):
            residue_rows.append(
                {
                    "Split": split_name,
                    "Residue_Code": ch,
                    "Category": "Valid IUPAC Extended/Ambiguous"
                    if ch in NON_CANONICAL_IUPAC
                    else "Invalid / Corrupted Character",
                    "Biological_Meaning": NON_CANONICAL_IUPAC.get(ch, "Unrecognized Symbol"),
                    "Total_Occurrences": char_occurrence[ch],
                    "Affected_Proteins": char_seq_count[ch],
                    "Affected_Proteins_Pct (%)": round(
                        char_seq_count[ch] / max(total_seqs, 1) * 100.0, 4
                    ),
                }
            )

    overview_df = pd.DataFrame(overview_rows)
    residue_df = pd.DataFrame(residue_rows)
    return overview_df, residue_df


def extract_aac_features(sequences: Iterable[str]) -> np.ndarray:
    """
    extract 22 dimensional amino acid composition + sequence length features
    """
    seq_list = list(sequences)
    n = len(seq_list)
    feats = np.zeros((n, 22), dtype=np.float32)

    for i, seq in enumerate(seq_list):
        length = max(len(seq), 1)
        counts = Counter(seq)
        canonical_total = 0
        for aa, idx in AA_TO_IDX.items():
            c = counts.get(aa, 0)
            feats[i, idx] = c / length
            canonical_total += c
        feats[i, 20] = (length - canonical_total) / length
        feats[i, 21] = np.log1p(length)

    return feats


class ProteinFeatureExtractor:
    """
    modular feature pipeline combining:
    1. amino acid composition  + length features (22 dense dims)
    2. character n-gram (k-mer) TF-IDF features (default: dipeptides/tripeptides)
    3. optional top N organism taxon id one hot encoding
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 3),
        max_kmer_features: int = 1500,
        top_taxa: int = 50,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_kmer_features = max_kmer_features
        self.top_taxa = top_taxa
        self.tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=max_kmer_features,
            lowercase=False,
            dtype=np.float32,
        )
        self.top_taxon_ids_: List[str] = []

    def fit(
        self,
        sequences: Iterable[str],
        taxon_ids: Optional[Iterable[str | int]] = None,
    ) -> "ProteinFeatureExtractor":
        seq_list = list(sequences)
        self.tfidf.fit(seq_list)
        if taxon_ids is not None and self.top_taxa > 0:
            counts = Counter(str(t) for t in taxon_ids)
            self.top_taxon_ids_ = [t for t, _ in counts.most_common(self.top_taxa)]
        return self

    def transform(
        self,
        sequences: Iterable[str],
        taxon_ids: Optional[Iterable[str | int]] = None,
    ) -> sparse.csr_matrix:
        seq_list = list(sequences)
        aac_dense = sparse.csr_matrix(extract_aac_features(seq_list))
        kmer_sparse = self.tfidf.transform(seq_list)

        blocks = [aac_dense, kmer_sparse]

        if self.top_taxon_ids_ and taxon_ids is not None:
            tax_list = [str(t) for t in taxon_ids]
            tax_to_col = {t: idx for idx, t in enumerate(self.top_taxon_ids_)}
            rows, cols = [], []
            for r, t in enumerate(tax_list):
                if t in tax_to_col:
                    rows.append(r)
                    cols.append(tax_to_col[t])
            data = np.ones(len(rows), dtype=np.float32)
            tax_sparse = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(seq_list), len(self.top_taxon_ids_)),
            )
            blocks.append(tax_sparse)

        return sparse.hstack(blocks, format="csr", dtype=np.float32)

    def fit_transform(
        self,
        sequences: Iterable[str],
        taxon_ids: Optional[Iterable[str | int]] = None,
    ) -> sparse.csr_matrix:
        return self.fit(sequences, taxon_ids).transform(sequences, taxon_ids)


def build_multilabel_targets(
    protein_ids: Iterable[str],
    terms_df: pd.DataFrame,
    top_k: int = 250,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """
    build binary multi label target matrix Y of shape (n_proteins, top_k) for the
    most frequent `top_k` GO terms in `terms_df`.

    Returns:
        Y: np.ndarray of shape (n_proteins, top_k) with {0, 1}
        selected_terms: list of GO term strings ordered by frequency
        term_to_aspect: dict mapping each selected GO term to its aspect (BPO/CCO/MFO)
    """
    protein_list = list(protein_ids)
    prot_to_row = {pid: idx for idx, pid in enumerate(protein_list)}

    term_counts = terms_df["term"].value_counts()
    selected_terms = term_counts.head(top_k).index.tolist()
    term_to_col = {term: idx for idx, term in enumerate(selected_terms)}

    subset = terms_df[
        terms_df["term"].isin(term_to_col) & terms_df["EntryID"].isin(prot_to_row)
    ]

    Y = np.zeros((len(protein_list), len(selected_terms)), dtype=np.float32)
    rows = subset["EntryID"].map(prot_to_row).to_numpy()
    cols = subset["term"].map(term_to_col).to_numpy()
    Y[rows, cols] = 1.0

    aspect_map = (
        subset[["term", "aspect"]]
        .drop_duplicates(subset=["term"])
        .set_index("term")["aspect"]
        .to_dict()
    )
    return Y, selected_terms, aspect_map
