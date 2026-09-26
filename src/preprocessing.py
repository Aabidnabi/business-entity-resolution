import re
import unicodedata
import pandas as pd


LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation",
    "co", "company", "llc", "ltd", "limited",
    "plc", "llp", "lp", "pvt", "private", "priv",
    "pte", "gmbh", "ag", "sa", "sas", "sarl",
    "bv", "nv", "oy", "ab",
}


def normalize_series(series: pd.Series) -> pd.Series:
    """
    Fast vectorized normalization.
    Designed for millions of rows.
    """
    s = series.fillna("").astype("string")

    # Unicode normalization is the one operation that remains
    # Python-level, but everything else is vectorized.
    s = s.map(lambda x: unicodedata.normalize("NFKC", x))

    s = s.str.lower()

    # Replace punctuation with spaces.
    s = s.str.replace(r"[^\w\s]", " ", regex=True)

    # Underscores behave like separators.
    s = s.str.replace("_", " ", regex=False)

    # Collapse whitespace.
    s = s.str.replace(r"\s+", " ", regex=True)

    return s.str.strip()


def normalize_name_series(series: pd.Series) -> pd.Series:
    return normalize_series(series)


def normalize_address_series(series: pd.Series) -> pd.Series:
    return normalize_series(series)


def normalize_country_series(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype("string")
        .str.lower()
        .str.strip()
    )


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns using vectorized pandas operations.
    """
    result = df.copy()

    result["name_norm"] = normalize_name_series(
        result["business_name"]
    )

    result["address_norm"] = normalize_address_series(
        result["business_address"]
    )

    result["country_norm"] = normalize_country_series(
        result["country"]
    )

    # Token representation.
    result["name_tokens"] = (
        result["name_norm"]
        .str.split()
        .map(
            lambda tokens: [
                t for t in tokens
                if len(t) >= 2 and t not in LEGAL_SUFFIXES
            ]
        )
    )

    result["name_token_string"] = (
        result["name_tokens"]
        .map(lambda x: " ".join(sorted(set(x))))
    )

    # Numeric tokens from address.
    result["address_numbers"] = (
        result["address_norm"]
        .str.findall(r"\d+")
    )

    return result