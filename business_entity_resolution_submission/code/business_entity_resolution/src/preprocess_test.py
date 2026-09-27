from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from preprocessing import add_normalized_columns

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "data" / "processed"
CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name in ["test_source1", "test_source2", "test_source3"]:
    input_file = TRAIN_DIR / f"{name}.tsv"
    output_file = OUTPUT_DIR / f"{name}.parquet"

    if output_file.exists():
        output_file.unlink()

    print(f"\n=== {name} ===", flush=True)

    total = 0
    first = True

    for chunk in pd.read_csv(
        input_file,
        sep="\t",
        chunksize=CHUNK_SIZE,
    ):
        chunk = add_normalized_columns(chunk)

        table = pa.Table.from_pandas(
            chunk,
            preserve_index=False,
        )

        if first:
            writer = pq.ParquetWriter(
                output_file,
                table.schema,
            )
            first = False

        writer.write_table(table)
        total += len(chunk)

        print(f"Processed {total:,} rows", flush=True)

    writer.close()

    print(f"Saved: {output_file}", flush=True)
