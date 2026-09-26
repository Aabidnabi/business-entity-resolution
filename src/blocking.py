from collections import defaultdict
from typing import Iterable

import pandas as pd


def build_inverted_index(
    df: pd.DataFrame,
    column: str,
    min_token_length: int = 2,
    max_df_ratio: float = 0.01,
) -> dict[str, list[int]]:
    """
    Build token -> row-index inverted index.

    Very common tokens are ignored because they create huge candidate sets.
    """
    n_rows = len(df)
    max_allowed = max(1, int(n_rows * max_df_ratio))

    index: dict[str, list[int]] = defaultdict(list)

    for row_idx, tokens in enumerate(df[column]):
        if not isinstance(tokens, list):
            continue

        unique_tokens = {
            token
            for token in tokens
            if isinstance(token, str)
            and len(token) >= min_token_length
        }

        for token in unique_tokens:
            index[token].append(row_idx)

    # Keep only useful/rare-enough blocking tokens.
    index = {
        token: rows
        for token, rows in index.items()
        if len(rows) <= max_allowed
    }

    return index


def build_exact_name_index(
    df: pd.DataFrame,
) -> dict[str, list[int]]:
    """Exact normalized-name -> row indices."""
    index: dict[str, list[int]] = defaultdict(list)

    for row_idx, value in enumerate(df["name_norm"]):
        if value:
            index[value].append(row_idx)

    return dict(index)


def generate_token_candidates(
    query_tokens: Iterable[str],
    token_index: dict[str, list[int]],
) -> set[int]:
    """Return union of candidate rows sharing useful name tokens."""
    candidates: set[int] = set()

    for token in query_tokens:
        rows = token_index.get(token)

        if rows:
            candidates.update(rows)

    return candidates


def generate_exact_name_candidates(
    name_norm: str,
    exact_name_index: dict[str, list[int]],
) -> set[int]:
    """Return candidates with exact normalized business name."""
    if not name_norm:
        return set()

    return set(exact_name_index.get(name_norm, []))


def generate_candidates_for_record(
    name_norm: str,
    name_tokens: list[str],
    exact_name_index: dict[str, list[int]],
    token_index: dict[str, list[int]],
) -> set[int]:
    """
    Combine exact-name and token blocking for one S1 record.
    """
    candidates = generate_exact_name_candidates(
        name_norm,
        exact_name_index,
    )

    candidates.update(
        generate_token_candidates(
            name_tokens,
            token_index,
        )
    )

    return candidates


def add_country_filter(
    candidate_rows: set[int],
    source_df: pd.DataFrame,
    source_country: str,
) -> set[int]:
    """
    Fast country-aware filtering.

    Precompute the country column once instead of calling
    pandas .iloc for every candidate.
    """
    if not source_country or not candidate_rows:
        return candidate_rows

    countries = source_df["country_norm"].to_numpy()

    return {
        row_idx
        for row_idx in candidate_rows
        if countries[row_idx] == source_country
    }
def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    max_candidates_per_s1: int = 500,
) -> pd.DataFrame:
    """
    Generate candidate pairs between S1 and S2/S3.

    Important:
    - Candidate generation is intentionally broad.
    - Matching happens later.
    - We never do an all-pairs Cartesian product.
    """

    exact2 = build_exact_name_index(source2)
    exact3 = build_exact_name_index(source3)

    token2 = build_inverted_index(source2, "name_tokens")
    token3 = build_inverted_index(source3, "name_tokens")

    rows = []

    for s1_idx, s1 in source1.iterrows():
        candidates2 = generate_candidates_for_record(
            s1["name_norm"],
            s1["name_tokens"],
            exact2,
            token2,
        )

        candidates3 = generate_candidates_for_record(
            s1["name_norm"],
            s1["name_tokens"],
            exact3,
            token3,
        )

        # Prefer candidates from the same country.
        candidates2 = add_country_filter(
            candidates2,
            source2,
            s1["country_norm"],
        )

        candidates3 = add_country_filter(
            candidates3,
            source3,
            s1["country_norm"],
        )

        # Safety cap.
        # Later we will improve ranking instead of blindly truncating.
        candidates2 = list(candidates2)[:max_candidates_per_s1]
        candidates3 = list(candidates3)[:max_candidates_per_s1]

        for idx in candidates2:
            rows.append(
                (
                    s1["entity_id"],
                    source2.iloc[idx]["entity_id"],
                )
            )

        for idx in candidates3:
            rows.append(
                (
                    s1["entity_id"],
                    source3.iloc[idx]["entity_id"],
                )
            )

    return pd.DataFrame(
        rows,
        columns=["source1_entity_id", "candidate_entity_id"],
    )


if __name__ == "__main__":
    print("Blocking module loaded successfully.")