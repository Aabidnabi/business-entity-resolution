# Business Entity Resolution

## Overview

This project matches business records from Source 2 and Source 3 with the reference businesses in Source 1.

The data contains noisy business names, addresses, and country values, so the pipeline uses normalization, candidate generation, similarity features, and a machine-learning model.

## Pipeline

The solution follows these steps:

1. Preprocess and normalize business names, addresses, and countries.
2. Generate candidate pairs using normalized country and business-name tokens.
3. Build features for each candidate pair.
4. Train a LightGBM binary classification model.
5. Select the matching threshold using validation data and Macro F0.5.
6. Score the test candidate pairs.
7. Generate the final matching and candidate-pair TSV files.

## Model Features

The current model uses:

- Exact business-name match
- Exact normalized name-token match
- Business-name edit similarity
- Name length ratio
- Exact address match
- Address edit similarity
- Exact country match
- Missing-value indicators

## Candidate Generation

Candidate pairs are generated using normalized country and business-name tokens.

This reduces the number of pairs that need to be scored compared with comparing every Source 1 record against every Source 2 and Source 3 record.

## Model

LightGBM is used as the binary classification model.

The matching threshold is selected on validation data using Macro F0.5.

## Output Files

The final submission contains:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Both files contain one row for every Source 1 entity.

## Source Code

All pipeline source code is included under:

`code/business_entity_resolution/src/`

The main scripts cover preprocessing, candidate generation, feature creation, model training, threshold optimization, prediction, and submission validation.

## Dependencies

Required Python packages and versions are listed in:

`code/business_entity_resolution/requirements.txt`
