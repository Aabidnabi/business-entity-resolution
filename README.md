# Business Entity Resolution

A machine-learning pipeline for resolving business entities across multiple data sources by identifying records that refer to the same real-world businesses.

## 1. Project Overview

Business Entity Resolution (BER) determines whether records from different datasets represent the same real-world entity.

In this challenge:

- **Source 1 (S1)** contains reference business records.
- **Source 2 (S2)** contains candidate business records.
- **Source 3 (S3)** contains additional candidate business records.

For every S1 entity, the system identifies the corresponding S2 and/or S3 entity IDs.

The project implements an end-to-end pipeline combining:

- Data preprocessing and normalization
- Candidate generation / blocking
- Pairwise similarity features
- Supervised machine learning
- LightGBM classification
- Threshold optimization
- Test prediction
- Submission generation and validation

The central challenge is reducing the enormous search space of possible record pairs while retaining as many true matches as possible.

---

## 2. Problem Statement

Given three business-record sources, determine which S2 and S3 records refer to the same real-world businesses as each S1 record.

A conceptual view of the system is:

```text
Source 1
   |
   +--------------------+
   |                    |
   v                    v
Source 2             Source 3
   |                    |
   +---------+----------+
             |
             v
     Candidate Generation
             |
             v
      Feature Engineering
             |
             v
        LightGBM Model
             |
             v
        Match Scores
             |
             v
      Threshold Selection
             |
             v
      Final Entity Matches
```

The same business may appear with differences in:

- Name spelling
- Name token order
- Abbreviations
- Address formatting
- Missing fields
- Country representation
- Character-level variations

Comparing every S1 record against every S2/S3 record is computationally expensive, so the pipeline first generates plausible candidate pairs and then applies a machine-learning model.

---

## 3. Project Objectives

1. Normalize heterogeneous business records.
2. Generate high-quality candidate pairs using blocking.
3. Build informative pairwise similarity features.
4. Train a supervised matching model.
5. Score test candidate pairs.
6. Select an appropriate decision threshold.
7. Generate the required competition submission files.
8. Validate the final submission.

---

## 4. Pipeline

The complete workflow is:

```text
Raw TSV Data
    ↓
Preprocessing
    ↓
Normalized Parquet Data
    ↓
Candidate Generation / Blocking
    ↓
Training Pair Construction
    ↓
Pairwise Feature Engineering
    ↓
LightGBM Training
    ↓
Threshold Optimization
    ↓
Test Candidate Generation
    ↓
Test Feature Generation
    ↓
Test Prediction
    ↓
matching_results.tsv
candidate_pairs.tsv
    ↓
Official Validation
```

---

## 5. Repository Structure

```text
business-entity-resolution/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── data/
│   └── processed/
│
├── src/
│   ├── preprocessing
│   ├── candidate generation
│   ├── feature generation
│   ├── training
│   └── prediction
│
├── notebooks/
│
├── output/
│
├── README.md
├── TEAM_GUIDE.md
├── Documentation_template.md
├── requirements.txt
└── .gitignore
```

Generated datasets, caches, virtual environments and temporary artifacts should not be committed unnecessarily.

---

## 6. Dataset

The official challenge dataset is required to run the complete pipeline.

Expected structure:

```text
dataset/
├── train/
│   └── train_ground_truth.tsv
└── test/
    └── ...
```

Processed datasets are stored separately:

```text
data/
└── processed/
    ├── train_source1.parquet
    ├── train_source2.parquet
    ├── train_source3.parquet
    ├── test_source1.parquet
    ├── test_source2.parquet
    └── test_source3.parquet
```

The raw competition dataset is not part of the normal GitHub source-code workflow.

---

## 7. Data Preprocessing

The preprocessing stage converts raw business records into normalized representations.

Important normalized information includes:

- Business names
- Business addresses
- Countries
- Name tokens
- Address numbers / components

Processed data is stored in Parquet format for efficient downstream processing.

---

## 8. Candidate Generation / Blocking

Candidate generation is one of the most important stages of the system.

Rather than comparing every possible S1-S2 and S1-S3 pair, blocking produces a smaller set of plausible candidates.

The project uses information such as:

- `country_norm`
- Normalized business names
- `name_token_string`
- Name tokens
- Character-level signatures

One principal blocking strategy uses:

```text
country_norm + name_token_string
```

Candidate generation determines the maximum possible recall of the downstream model:

```text
If a true match is not generated as a candidate,
the machine-learning model cannot recover it.
```

For this reason, candidate recall is a critical evaluation metric.

---

## 9. Candidate Frequency Control

Very frequent blocking keys can produce extremely large candidate sets.

The project therefore evaluates block-frequency limits.

Example candidate-pair counts observed during development:

```text
Frequency limit       Candidate pairs
-------------------------------------
100                    30,626,755
50                     24,486,665
30                     12,573,808
20                      7,420,263
15                      5,682,856
10                      4,717,343
5                       3,503,343
```

A frequency limit of 20 produced the following test candidate set:

```text
S1 records:              1,732,544
S1 with candidates:      1,305,053
Candidate pairs:         7,420,263
Average candidates/S1:   5.69
Maximum candidates/S1:   40
```

These figures describe the generated candidate set and are not official leaderboard performance metrics.

---

## 10. Pairwise Feature Engineering

Each S1-candidate pair is converted into a feature vector.

The current training feature schema is:

```text
source1_entity_id
matched_entity_id
label
name_exact
name_token_exact
name_edit_sim
name_length_ratio
address_exact
address_edit_sim
country_exact
s1_name_missing
s2_name_missing
s1_address_missing
s2_address_missing
```

### Feature descriptions

| Feature | Description |
|---|---|
| `name_exact` | Exact equality of normalized business names |
| `name_token_exact` | Exact equality of normalized/tokenized names |
| `name_edit_sim` | Name similarity based on edit distance |
| `name_length_ratio` | Relative name-length similarity |
| `address_exact` | Exact equality of normalized addresses |
| `address_edit_sim` | Address similarity based on edit distance |
| `country_exact` | Whether normalized countries match |
| `s1_name_missing` | Missing-name indicator for S1 |
| `s2_name_missing` | Missing-name indicator for candidate |
| `s1_address_missing` | Missing-address indicator for S1 |
| `s2_address_missing` | Missing-address indicator for candidate |

The target variable is:

```text
label = 1  -> true match
label = 0  -> non-match
```

---

## 11. Machine Learning Model

The current matching model is **LightGBM**.

The model receives pairwise similarity features and produces a match score for every candidate pair.

```text
Candidate Pair
      ↓
Similarity Features
      ↓
LightGBM
      ↓
Match Score
      ↓
Threshold
      ↓
Match / Non-match
```

LightGBM is used as the main tabular classification model for the structured entity-matching features.

---

## 12. Training Data

The training pipeline uses the official training ground truth to create labelled candidate pairs.

The resulting training feature file is:

```text
output/training_features.parquet
```

The observed training feature dataset contained:

```text
8,638,365 rows
```

with the feature schema described above.

---

## 13. Threshold Optimization

LightGBM produces continuous scores, so a threshold is required to convert scores into predicted matches.

An example training-side evaluation produced:

```text
Threshold   Precision   Recall   F0.5
----------------------------------------
0.50        0.9734      0.9623   0.9712
0.55        0.9806      0.9542   0.9752
0.60        0.9849      0.9483   0.9774
0.65        0.9886      0.9421   0.9790
0.70        0.9905      0.9381   0.9795
0.75        0.9924      0.9330   0.9799
0.80        0.9941      0.9268   0.9799
0.85        0.9955      0.9201   0.9795
0.90        0.9970      0.9096   0.9782
0.95        0.9989      0.8857   0.9740
0.99        0.9996      0.8614   0.9685
```

For this training evaluation, the highest observed F0.5 was approximately:

```text
F0.5:       0.979896
Threshold:  0.75
Precision:  0.992360
Recall:     0.933019
```

These are training-side metrics and should not be interpreted as the official competition leaderboard score.

---

## 14. Test Prediction

The test prediction workflow is:

1. Load the trained LightGBM model.
2. Load test features.
3. Score all candidate pairs.
4. Apply the selected threshold.
5. Group predicted IDs by Source 1 entity.
6. Generate the final matching-results file.

The test feature dataset generated during development contained:

```text
7,420,263 candidate rows
```

The scored predictions are stored in:

```text
output/test_scored.parquet
```

with:

```text
source1_entity_id
matched_entity_id
score
```

---

## 15. Submission Files

The expected submission package contains:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### `matching_results.tsv`

Contains one row for each Source 1 entity:

```text
source1_entity_id    matched_entity_ids
```

Example:

```text
S1-100001295    S2-334300764,S3-327023247,S3-367738979
```

### `candidate_pairs.tsv`

Contains generated candidate pairs:

```text
source1_entity_id    matched_entity_id
```

Example:

```text
S1-100001295    S2-334300764
S1-100001295    S3-327023247
```

---

## 16. Submission Validation

Before submission, run the official validator supplied by the challenge:

```bash
python3 utils/validate_submission.py     --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv     --test-dir dataset/test     --check-ids
```

The submission should only be finalized after the validator reports:

```text
PASS — no blocking issues found. Safe to submit.
```

---

## 17. Python Environment

The tested development environment uses:

```text
Python 3.11.6
```

Check the version:

```bash
python3.11 --version
```

Expected:

```text
Python 3.11.6
```

Windows PowerShell users can create and activate the environment with:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## 18. Dependencies

Tested versions:

```text
numpy==2.2.3
pandas==2.2.3
pyarrow==25.0.1
duckdb==1.5.5
lightgbm==4.6.0
scikit-learn==1.6.1
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 19. Reproducing the Pipeline

The general execution order is:

```text
1. Prepare the official dataset
2. Run preprocessing
3. Generate training candidates
4. Build training features
5. Train LightGBM
6. Evaluate and optimize threshold
7. Generate test candidates
8. Build test features
9. Score test candidates
10. Generate matching_results.tsv
11. Generate candidate_pairs.tsv
12. Run the official validator
```

The exact script names and commands should follow the current source code and `TEAM_GUIDE.md`.

---

## 20. Candidate Recall

Candidate recall is a key property of the system.

The relationship is:

```text
Ground-truth match
       ↓
Was it generated as a candidate?
       |
   +---+---+
   |       |
  YES      NO
   |       |
   ↓       ↓
ML can    ML cannot
score it  recover it
```

Therefore, improving the classifier alone cannot recover true matches that were removed during blocking.

Candidate-generation experiments should consequently be evaluated against training ground truth before changing the downstream model.

---

## 21. Future Improvements

Potential directions for further development include:

### Candidate Generation

- Multiple complementary blocking strategies
- Token-based blocking
- Character n-gram blocking
- Prefix and suffix signatures
- Address-based blocking
- Country-aware blocking
- Frequency-aware blocking
- Multi-block candidate unions

### Feature Engineering

- Name token Jaccard similarity
- Address token Jaccard similarity
- Address-number agreement
- Character n-gram similarity
- Additional edit-distance features
- Token-overlap counts
- Name/address length differences
- Feature interactions
- More robust missing-value handling

### Model Improvements

- Hard-negative mining
- Better negative sampling
- Class balancing
- Cross-validation
- Hyperparameter tuning
- Probability calibration
- Ranking-based matching
- More systematic threshold selection

Experiments should be conducted one major change at a time so that improvements can be measured against the baseline.

---

## 22. Experimentation Strategy

Recommended workflow:

```text
Baseline
   ↓
Change one component
   ↓
Measure candidate recall
   ↓
Measure validation performance
   ↓
Compare with baseline
   ↓
Keep or reject the change
```

Do not overwrite a working baseline without keeping a backup or recording the previous configuration.

---

## 23. Collaboration Workflow

Before modifying shared production code:

```bash
git checkout main
git pull
```

Create a feature branch:

```bash
git checkout -b feature-name
```

After testing:

```bash
git add .
git commit -m "Describe the change"
git push
```

Avoid committing:

- `.venv/`
- Python caches
- Large generated Parquet files
- Temporary output
- IDE configuration
- Competition datasets unless required
- Credentials or API keys

---

## 24. Challenge Constraints

The implementation should follow the official competition rules.

Do not use:

- External business databases
- Commercial entity-resolution APIs
- Geocoding APIs
- Prohibited internet-based data augmentation
- Other external information sources prohibited by the challenge

Only data and methods permitted by the official challenge specification should be used.

---

## 25. Reproducibility

For reproducible experiments, record:

- Python version
- Dependency versions
- Candidate-generation parameters
- Feature configuration
- Model configuration
- Threshold
- Training-data version
- Test-data version
- Output-generation procedure

Keep `requirements.txt` synchronized with the tested environment.

---

## 26. Final Checklist

Before creating the final submission:

```text
[ ] Dataset correctly prepared
[ ] Candidate generation completed
[ ] Test features generated
[ ] Test candidates scored
[ ] matching_results.tsv generated
[ ] candidate_pairs.tsv generated
[ ] Correct column names used
[ ] Correct entity IDs used
[ ] Required Source 1 rows present
[ ] Candidate pairs valid
[ ] No unnecessary files included
[ ] Official validator executed
[ ] Validator reports PASS
[ ] Final package follows the official structure
```

---

## 27. Key Takeaway

This project implements a complete machine-learning-based Business Entity Resolution pipeline.

The main design principle is:

```text
High-quality candidate generation
            +
Informative pairwise features
            +
Supervised matching model
            +
Careful threshold selection
            =
Entity Resolution Pipeline
```

Candidate generation controls the maximum achievable recall, while feature engineering and the matching model determine how effectively plausible candidates can be distinguished.

The repository is structured to support reproducible experiments, team collaboration and future improvements in entity-resolution methodology.

---

## 28. Team and Acknowledgements

This project was developed as part of a collaborative machine-learning challenge focused on Business Entity Resolution.

The work involved collaboration across:

- Data preprocessing
- Candidate generation
- Feature engineering
- Machine-learning modelling
- Threshold optimization
- Validation
- Submission preparation

The project also provided practical experience in designing and evaluating an end-to-end entity-resolution system under competition constraints.

---

## License

Add the appropriate license for the repository if required by the team or competition organizers.
