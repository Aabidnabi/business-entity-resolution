# Team Guide - Business Entity Resolution

This guide explains how a teammate can clone the repository, prepare the environment, set up the dataset, run the pipeline and prepare the final submission.

## 1. Clone the Repository

git clone <GITHUB_REPO_URL>

cd business-entity-resolution

Check the branch:

git branch

Before starting new work:

git checkout main
git pull

Create a separate experiment branch:

git checkout -b <experiment-name>

## 2. Python Version

The tested local environment uses:

Python 3.11.6

Check:

python3.11 --version

Expected:

Python 3.11.6

Use Python 3.11.6 where possible.

## 3. Create Virtual Environment

macOS/Linux:

python3.11 -m venv .venv

source .venv/bin/activate

Windows:

python -m venv .venv

.venv\Scripts\activate

## 4. Install Dependencies

The repository contains pinned versions in requirements.txt.

Install:

pip install -r requirements.txt

Tested versions:

numpy==2.2.3
pandas==2.2.3
pyarrow==25.0.1
duckdb==1.5.5
lightgbm==4.6.0
scikit-learn==1.6.1

Check installed versions if needed:

pip show numpy pandas pyarrow duckdb lightgbm scikit-learn

## 5. Dataset Setup

The complete pipeline requires the official challenge dataset.

Expected structure:

dataset/
├── train/
└── test/

The code expects the challenge data to be available from these repository-relative paths.

Do not rename files unless the corresponding code is also updated.

The raw challenge dataset should normally remain outside GitHub source control unless the team has confirmed an appropriate Git LFS/data-storage setup.

## 6. Local Directory Structure

After setup, the working directory should look approximately like:

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
└── Documentation_template.md

Generated data under data/processed/ and generated prediction files under output/ should not be treated as source code.

## 7. Current Pipeline

The pipeline is:

1. Preprocess Source 1, Source 2 and Source 3.
2. Generate candidate pairs.
3. Build training pairs.
4. Build pairwise features.
5. Train LightGBM.
6. Optimize the matching threshold.
7. Generate test predictions.
8. Build final matching_results.tsv.
9. Build final candidate_pairs.tsv.
10. Run the official validator.

## 8. Current Baseline Features

### Name

- normalized business name
- normalized name token string
- exact name match
- exact token match
- Levenshtein similarity
- name length ratio

### Address

- normalized address
- exact address match
- Levenshtein similarity

### Country

- exact country match

### Missing values

- Source 1 name missing
- Source 2/3 name missing
- Source 1 address missing
- Source 2/3 address missing

## 9. Current Improvement Plan

The current leaderboard score is approximately 0.49.

The first investigation should determine whether the score is limited by:

A. candidate recall

or

B. incorrect model predictions / false positives.

Candidate recall should be measured against the training ground truth before making major model changes.

After that, investigate:

1. Better candidate blocking.
2. Name token Jaccard.
3. Address token Jaccard.
4. Address number agreement.
5. Postal/address signals.
6. Character-level similarity.
7. Hard-negative sampling.
8. Better pairwise model features.
9. Threshold optimization.
10. Final validation.

Do not change the complete pipeline at once.

## 10. Protect the Baseline

Before changing a production script:

cp src/<file>.py src/<file>_backup.py

For major experiments, prefer versioned scripts such as:

build_test_candidates_v2.py
build_features_v2.py
train_model_v2.py

Keep the old working implementation until the new implementation has been tested.

## 11. Git Workflow

Before work:

git checkout main
git pull

Create branch:

git checkout -b <experiment-name>

After changes:

git status

git add <changed-files>

git commit -m "Describe the experiment"

git push -u origin <experiment-name>

Do not force-push main.

## 12. AWS / SageMaker Setup

The same Python dependencies can be used on AWS.

Recommended approach:

1. Start a Python 3.11 environment on the AWS machine.
2. Clone the repository.
3. Place the official challenge dataset under:

dataset/train
dataset/test

4. Create a virtual environment if supported by the environment.
5. Install:

pip install -r requirements.txt

6. Run the pipeline scripts in order.
7. Keep generated data on the AWS instance/storage rather than committing it to GitHub.
8. Run the official validator before preparing the final submission.

Important:

The dataset is large, so make sure the AWS instance/storage has enough disk space for:

- raw train/test data
- processed Parquet files
- candidate pairs
- feature files
- model outputs

Do not assume a small instance will be sufficient.

## 13. Candidate Recall Audit

Before making major improvements, run the candidate recall audit.

The purpose is to answer:

How many true training matches are actually present in our generated candidate set?

If true matches are missing from candidates, the model can never recover them.

Therefore candidate generation is the first major optimization target if recall is low.

## 14. Model Optimization

Once candidate recall is sufficiently high:

- add stronger similarity features
- improve hard-negative sampling
- retrain LightGBM
- evaluate on a fixed validation split
- optimize threshold
- compare with baseline

Do not choose a threshold only because it looks good on one small sample.

## 15. Submission Files

The final output requires:

output/matching_results.tsv
output/candidate_pairs.tsv

Every test Source 1 entity must be represented.

Predicted IDs must come only from Source 2 or Source 3.

There must be no duplicate matched IDs for an S1.

Every final matching ID must also be present in the candidate set.

## 16. Official Validator

Run:

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids

Only proceed to final submission after the validator passes.

Expected:

PASS — no blocking issues found. Safe to submit.

## 17. Final Submission Structure

The official final package should be:

<team_name>_submission.zip

├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Do not include:

- raw dataset
- .venv
- Python cache
- generated temporary files
- unnecessary notebooks
- unrelated files

## 18. Important Challenge Constraints

Follow the official challenge rules regarding:

- allowed data
- external lookup
- model licensing
- model size
- submission format
- candidate generation
- validation

Do not use external business databases, commercial entity-resolution APIs, geocoding services or prohibited internet augmentation.

## 19. Team Principle

Keep the working baseline safe.

Measure before changing.

Change one major component at a time.

Record validation results.

Never assume that a change improved the system until it has been measured.

The main optimization target is to improve true-match recall while maintaining high precision.
