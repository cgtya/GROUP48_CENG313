# CENG313 Introduction to Data Science - Project
## Competition: [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)

---

## 1. Group Members

**Group 48**

| Member Name | Student ID | GitHub Username |
| :--- | :--- | :--- |
| `Çağatay Aslan` | `23118080003` | `@cgtya` |
| `Mehmet Samet Sümer` | `24118080005` | `@Keynekei` |

---

## 2. Project Overview & Chosen Competition

The Critical Assessment of Functional Annotation 6 (CAFA 6) challenge on Kaggle focuses on predicting the biological functions of proteins from their amino acid sequences. Protein function is represented through Gene Ontology (GO) terms organized as a Directed Acyclic Graph (DAG) across three independent sub-ontologies (aspects):

1. **MFO (Molecular Function Ontology):** Molecular-level activities performed by a gene product (e.g., catalytic or binding activity).
2. **BPO (Biological Process Ontology):** Larger biological programs accomplished by multiple molecular activities (e.g., DNA repair, signal transduction).
3. **CCO (Cellular Component Ontology):** Cellular structures and locations where a protein performs its function (e.g., nucleus, plasma membrane, mitochondrion).

### Evaluation Metric
Predictions are evaluated using the weighted maximum F-measure ($F_{\max}$) computed over a range of decision thresholds $t \in [0, 1]$, weighted by Information Accretion ($IA$) values so that predicting rare, highly specific functional terms carries greater weight than predicting broad root terms.

---

## 3. Dataset Description

| File | Description |
| :--- | :--- |
| `Train/train_sequences.fasta` | Amino acid sequences of training proteins in standard UniProt FASTA format (`sp\|` and `tr\|` headers). |
| `Train/train_terms.tsv` | Ground-truth annotations mapping `EntryID` to `term` (`GO:XXXXXXX`) and `aspect` (`BPO`, `CCO`, `MFO`). |
| `Train/train_taxonomy.tsv` | Mapping of each training protein to its source organism's NCBI Taxonomy ID (`taxon_id`). |
| `Train/go-basic.obo` | Gene Ontology graph structure defining `is_a` and `part_of` parent-child hierarchy relationships. |
| `IA.tsv` | Information Accretion ($IA$) specificity weights for each GO term. |
| `Test/testsuperset.fasta` | Target protein sequences for which GO term probabilities must be predicted. |
| `Test/testsuperset-taxon-list.tsv` | NCBI Taxonomy IDs for the test set proteins. |

---

## 4. Methodology

### Assignment 1: Exploratory Data Analysis (EDA) & Baselines
1. **Exploratory Data Analysis (`assignment1_eda_baseline.ipynb`):**
   - Inspected protein sequence length distributions across training and test sets, missing values, and canonical 20 amino acid compositions.
   - Analyzed annotation balance across `BPO`, `CCO`, and `MFO` sub-ontologies.
   - Visualized the severe **long-tail label distribution** (log-log frequency curve) and label cardinality per protein.
   - Examined organism taxonomy diversity (`train_taxonomy.tsv`) and Information Accretion ($IA$) weight distributions (`IA.tsv`).
2. **Feature Engineering (`src/data.py` -> `ProteinFeatureExtractor`):**
   - **Amino Acid Composition (AAC):** 20 canonical amino acid relative frequencies + non-standard residue fraction + $\log(1 + L)$ sequence length.
   - **Sequence $k$-mers:** Character $n$-gram ($k \in \{2, 3\}$ dipeptide and tripeptide) TF-IDF features.
   - **Taxonomic Indicators:** One-hot encoding of the top-50 most frequent NCBI Taxonomy IDs.
3. **Baseline Models (`src/models.py`):**
   - **Baseline 0 (`NaiveFrequencyBaseline`):** Lower-bound reference model assigning empirical training prior probabilities of the top-$K$ GO terms to every protein.
   - **Baseline 1 (`RidgeMultilabelBaseline`):** Fast L2-regularized multi-output linear classifier trained on AAC + $k$-mer TF-IDF + Taxonomy features.
   - **Baseline 2 (`TorchMLPBaseline`):** GPU accelerated 3-layer Multi-Layer Perceptron (`Linear -> BatchNorm -> GELU -> Dropout`) trained with `BCEWithLogitsLoss`.

---

## 5. Validation Results Summary

> **Evaluation Status:** *Competition closed - evaluated on a held-out split.* (Late submissions are not accepted and test ground-truth labels are withheld; scores below are measured on a 20% held-out split of the training set, `16,481` proteins, `random_state=42`, across the Top-300 GO terms.)

| Model | Features Used | Mean Validation $F_{\max}$ (Unweighted) | Mean Validation $F_{\max}$ (IA-Weighted) | Optimal Threshold $t^*$ |
| :--- | :--- | :---: | :---: | :---: |
| **0. Naive Frequency Prior** | None (Empirical Priors) | 0.242246 | 0.238117 | 0.05 |
| **1. Ridge Linear Baseline** | AAC + (2,3)-mer TF-IDF + Taxon | 0.336527 | 0.328197 | 0.17 |
| **2. PyTorch MLP Baseline** | AAC + (2,3)-mer TF-IDF + Taxon | 0.369472 | 0.360862 | 0.23 |

- **Leaderboard & Public Notebook Comparison:** The 1st-place leaderboard score on CAFA 6 is `0.52430` (ZhuLab GOAlpha), while public baseline notebooks typically range between `~0.16` and `~0.37`. Our best Assignment 1 baseline (`0.360862` IA-weighted $F_{\max}$ on the Top-300 term subset) is `0.163438` below the 1st-place score.

---

## 6. How to Run the Notebooks

### Local Execution
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Get the compatible Pytorch for your system at [here](https://pytorch.org/get-started/locally/) and install
2. Download the competition dataset via Kaggle CLI (or extract the zip into `data/cafa-6-protein-function-prediction/`):
   ```bash
   kaggle competitions download -c cafa-6-protein-function-prediction -p data/cafa-6-protein-function-prediction
   unzip data/cafa-6-protein-function-prediction/cafa-6-protein-function-prediction.zip -d data/cafa-6-protein-function-prediction
   ```
3. Launch Jupyter / VS Code and run `assignment1_eda_baseline.ipynb`.

### Google Colab / Kaggle Execution
- Upload the repository (or clone it inside Colab using `!git clone ...`) and run `assignment1_eda_baseline.ipynb` from top to bottom.

---
