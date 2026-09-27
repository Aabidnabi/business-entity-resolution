# Business Entity Resolution

This repository contains our machine learning pipeline for the Business Entity Resolution challenge.

The goal is to identify which Source 2 and Source 3 records refer to the same real-world businesses as Source 1.

The project uses data preprocessing, candidate generation, pairwise similarity features, LightGBM and threshold optimization.

## Project Structure

business-entity-resolution/

├── dataset/
│   ├── train/
│   └── test/
├── data/
│   └── processed/
├── src/
├── notebooks/
├── output/
├── README.md
├── TEAM_GUIDE.md
├── requirements.txt
├── Documentation_template.md
└── .gitignore

## Python Version

The current development environment uses:

Python 3.11.6

Use Python 3.11.6 when possible so that the environment matches the tested setup.

Check your version:

python3.11 --version

Expected:

Python 3.11.6

## Required Libraries

The tested versions are:

numpy==2.2.3
pandas==2.2.3
pyarrow==25.0.1
duckdb==1.5.5
lightgbm==4.6.0
scikit-learn==1.6.1

Install them with:

pip install -r requirements.txt

## Local Setup

Create the environment:

python3.11 -m venv .venv

Activate it:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Dataset Setup

The official challenge dataset is required to run the complete pipeline.

Expected local structure:

dataset/
├── train/
└── test/

The challenge data should be obtained from the official challenge resources.

The raw dataset is not part of the normal GitHub source-code workflow.

## Pipeline

The current pipeline is:

Raw TSV data
    ↓
Preprocessing
    ↓
Candidate Generation
    ↓
Training Pairs
    ↓
Pairwise Features
    ↓
LightGBM Training
    ↓
Threshold Optimization
    ↓
Test Prediction
    ↓
matching_results.tsv
candidate_pairs.tsv
    ↓
Official Validation

## Current Baseline

The baseline uses normalized:

- business names
- business addresses
- countries
- name tokens
- address numbers

Current matching features include:

- exact normalized business name
- exact normalized name tokens
- name Levenshtein similarity
- name length ratio
- exact normalized address
- address Levenshtein similarity
- exact country match
- missing-value indicators

## Current Improvement Goal

The current leaderboard result is approximately 0.49.

The next goal is to improve the system substantially.

The first thing to investigate is candidate recall because candidate generation determines which possible matches the model is allowed to see.

Planned improvements:

1. Measure candidate recall against training ground truth.
2. Improve candidate blocking.
3. Add name token Jaccard similarity.
4. Add address token Jaccard similarity.
5. Add address-number agreement.
6. Add stronger character-level similarity.
7. Improve negative and hard-negative sampling.
8. Retrain the model.
9. Optimize the decision threshold.
10. Validate the final output.

Experiments should be performed one major change at a time so that improvements can be measured.

## Collaboration

Before changing production code:

git checkout main
git pull

Create a separate branch:

git checkout -b <experiment-name>

Keep backups of working code before major changes.

After testing:

git add .
git commit -m "Describe the change"
git push

## Important Files

src/
    Pipeline source code.

data/processed/
    Generated processed Parquet data.

output/
    Generated candidate and prediction files.

TEAM_GUIDE.md
    Detailed setup and collaboration instructions.

requirements.txt
    Exact tested Python dependency versions.

## Submission

The final competition package must follow the official challenge structure.

The final package contains:

output/
├── matching_results.tsv
└── candidate_pairs.tsv

code/
└── business_entity_resolution/
    ├── src/
    ├── README.md
    └── requirements.txt

Documentation_template.md

Do not add unnecessary files to the final submission package.

## Validation

Before submission, run the official validator:

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids

The submission should only be finalized after the validator reports:

PASS — no blocking issues found. Safe to submit.

## Important Rules

Do not use external business databases, commercial entity-resolution APIs, geocoding APIs or internet-based data augmentation where prohibited by the challenge rules.

Do not overwrite a working baseline without keeping a backup.

Do not submit generated caches, virtual environments or unnecessary files.

For the detailed setup and experiment workflow, read TEAM_GUIDE.md.
