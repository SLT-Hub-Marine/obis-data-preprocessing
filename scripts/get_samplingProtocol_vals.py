import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import os
import s3fs
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    fpath_prefix = "../" if "scripts" in os.getcwd() else ""

    fs = s3fs.S3FileSystem(anon=True)
    bucket_path = "obis-open-data/occurrence"
    file_paths = fs.glob(f"{bucket_path}/*.parquet")

    # 1. Setup our local directories and tracking logs
    _main_out_dir = Path(f"{fpath_prefix}samplingProtocol_column_data")
    _main_out_dir.mkdir(exist_ok=True)

    out_dir = _main_out_dir/"source_data"
    out_dir.mkdir(out_dir, exist_ok=True)

    log_file = out_dir / "processed_files.txt"
    error_log_file = out_dir / "error_files.txt"

    schema = pa.schema([
        ('filename', pa.string()),
        ('samplingProtocol', pa.string())
    ])

    # 2. Load already processed files into a set for O(1) lookups
    processed_files = set()
    if log_file.exists():
        with open(log_file, 'r') as f:
            processed_files = set(line.strip() for line in f)

    error_files = set()
    if error_log_file.exists():
        with open(error_log_file, 'r') as f:
            error_files = set(line.strip() for line in f)


    print("Fetching file list from S3...")

    print(f"Found {len(file_paths)} parquet files.")
    print(f"Found {len(processed_files)} already processed. Resuming extraction...\n")

    for i, file_path in tqdm(enumerate(file_paths, 1)):
        filename = os.path.basename(file_path)

        # 3. Checkpoint: Skip if already done
        if filename in processed_files:
            continue
        elif filename in error_files:
            continue

        print(f"[{i}/{len(file_paths)}] Processing {filename}...", end=" ", flush=True)

        try:
            with fs.open(file_path, 'rb') as f:
                pf = pq.ParquetFile(f, pre_buffer=True)
                print("File downloaded, processing", flush=True)
                file_schema = pf.schema_arrow

                has_protocol = False
                target_column = None

                if 'source' in file_schema.names:
                    print("Source in file_schema.names found")
                    source_field = file_schema.field('source')
                    if any(field.name == 'samplingProtocol' for field in source_field.type):
                        has_protocol = True
                        target_column = 'source'

                if not has_protocol and 'samplingProtocol' in file_schema.names:
                    print("samplingProtocol in file_schema.names found")
                    has_protocol = True
                    target_column = 'root'

                if has_protocol:
                    # 1. Read the necessary column
                    col_to_read = 'source' if target_column == 'source' else 'samplingProtocol'
                    print("Reading table")
                    table = pf.read(columns=[col_to_read])

                    # Debug: See how many rows we actually started with
                    total_rows = table.num_rows
                    print(f"Table read - rows = {total_rows}.")

                    if target_column == 'source':
                        print("target_column == source")
                        # Flattening 'source' turns a struct column into individual top-level columns
                        # This is safer than pc.struct_field for nested Parquet data
                        flat_table = table.flatten()
                        try:
                            protocols = flat_table.column('source.samplingProtocol')
                        except KeyError:
                            print("Key error detected.")
                            # Sometimes flattening uses different separators or indices
                            protocols = pc.struct_field(table.column('source'), 'samplingProtocol')
                    else:
                        print("target_column != source")
                        protocols = table.column('samplingProtocol')

                    # 2. Ensure protocols is treated as a column (ChunkedArray), not a scalar
                    # We cast to string to ensure consistency
                    protocols = protocols.cast(pa.string())

                    # 3. Create the filename column to match the length of our data
                    # Using table.num_rows ensures we match the input file's height
                    # filenames = pa.array([filename] * len(protocols), type=pa.string())
                    print("Creating out_table created.", flush=True)
                    out_table = pa.Table.from_arrays(
                        [protocols],
                        names=['samplingProtocol']
                    )
                    print("Out_table created.", flush=True)
                    # print(type(protocols))

                    # 4. Write if we have data
                    if out_table.num_rows > 0:
                        print("Writing file...", flush=True)
                        out_path = os.path.join(str(out_dir), filename)
                        pq.write_table(out_table, out_path)
                        print(f"Success! {out_table.num_rows}/{total_rows} rows extracted.", flush=True)
                    else:
                        print(f"Warning: File had {total_rows} rows, but extraction resulted in 0.")

                else:
                    print("File has no protocol")

            # 4. Success! Log it so we never process it again
            with open(log_file, 'a') as log:
                log.write(filename + '\n')

        except Exception as e:
            print(f"ERROR: {e}. Skipping file.")

            with open(error_log_file, 'a') as log:
                log.write(filename + '\n')

    print("\nAll files processed!")