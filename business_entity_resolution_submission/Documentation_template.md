# Business Entity Resolution

## Approach

This solution matches business records from Source 2 and Source 3 with Source 1.

The main steps are:

1. Normalize business names, addresses, and country values.
2. Generate candidate pairs using country and normalized business-name tokens.
3. Create similarity and exact-match features for candidate pairs.
4. Train a LightGBM binary classification model.
5. Select the matching threshold using validation data and Macro F0.5.
6. Apply the trained model to the test candidate pairs.
7. Generate the final matching and candidate-pair submission files.

## Features

The model uses features based on:

- Exact business-name match
- Normalized name-token match
- Business-name edit similarity
- Name length similarity
- Exact address match
- Address edit similarity
- Exact country match
- Missing-value indicators

## Candidate Generation

Candidate pairs are generated using normalized country and normalized business-name tokens.

This keeps the candidate set smaller than comparing every Source 1 record with every Source 2/Source 3 record.

## Model

LightGBM is used as the binary matching model.

The validation threshold is selected using Macro F0.5, with precision given more importance than recall.

## Output

The final submission contains:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Both files contain one row for every Source 1 entity.

## Reproducibility

Python dependencies are listed in `code/business_entity_resolution/requirements.txt`.

The source code used for preprocessing, candidate generation, feature creation, model training, threshold selection, prediction, and validation is included under:

`code/business_entity_resolution/src/`
