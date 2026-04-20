from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import polars as pl
from tqdm import tqdm
import os
import plotly.graph_objects as go

# pip install plotly kaleido

def get_samplingProtocol_column_data():
    fpath_prefix = "../" if "scripts" in os.getcwd() else ""
    # 1. Setup paths
    input_dir = Path(f"{fpath_prefix}samplingProtocol_column_data")
    output_dir = Path(f"{fpath_prefix}samplingProtocol_column_data/merged_data")
    if output_dir.exists() and len(list(output_dir.glob("*.csv"))):
        print("Existing files found, loading")
        return list(output_dir.glob("*.csv")), output_dir

    output_dir.mkdir(exist_ok=True)

    files = sorted(list(input_dir.glob("*.parquet")))
    num_files = float(len(files))
    num_shards = 10
    chunk_size = math.ceil(num_files / num_shards)
    print(f"Found {num_files} files. Splitting into {num_shards} shards (~{chunk_size} files each).")

    for i in range(num_shards):
        shard_files = files[i*chunk_size:int(min((i+1)*chunk_size, num_files))]

        shard_output = output_dir / f"combined_samplingProtocol_shard_{i:02}.csv"
        print(f"Processing Shard {i + 1}/{num_shards} -> {shard_output.name}")

        shard_dfs = []
        for f in tqdm(shard_files):
            df = pl.read_parquet(f)
            # Rename the first column to "value" and add source
            df = df.rename({df.columns[0]: "value"}).with_columns(pl.lit(f.name).alias("source"))
            shard_dfs.append(df)

        combined_shard = pl.concat(shard_dfs)
        combined_shard.write_csv(shard_output)

    return list(output_dir.glob("*.csv")), output_dir


def generate_master_value_counts(shard_paths, output_dir):

    """Generates a master CSV of unique samplingProtocol values and their overall counts."""
    out_path = output_dir / "samplingProtocol_unique_value_counts.csv"
    if out_path.exists():
        print("Found existing global unique samplingProtocol value counts CSV.")
        return out_path
    print("Generating global unique samplingProtocol value counts CSV...")
    # scan_csv builds a lazy computation graph, great for larger-than-memory datasets
    lf = pl.scan_csv(shard_paths)

    result = (
        lf.group_by("value")
        .agg(pl.len().alias("n_count"))
        .rename({"value": "unique_samplingProtocol_value"})
        .sort("n_count", descending=True)
    ).collect(streaming=True)

    result.write_csv(out_path)
    print(f"Saved master counts to: {out_path}")
    return out_path


def generate_master_source_stats(shard_paths, output_dir):
    """Generates a master CSV detailing value statistics per source parquet file."""
    out_path = output_dir / "samplingProtocol_per_parquet_stats.csv"
    if out_path.exists():
        print("Found existing per-parquet stats CSV.")
        return out_path
    print("Generating per-parquet stats CSV...")
    lf = pl.scan_csv(shard_paths)

    result = (
        lf.group_by("source")
        .agg(
            pl.len().alias("total_rows_incl_null"),
            pl.col("value").count().alias("total_non_null_values"),
            pl.col("value").drop_nulls().n_unique().alias("unique_values_excl_nulls")
        )
        .with_columns(
            # Using total non-null values for the ratio.
            # If you preferred ratio against ALL rows including nulls,
            # replace 'total_non_null_values' with 'total_rows_incl_null'
            (pl.col("unique_values_excl_nulls") / pl.col("total_rows_incl_null"))
            .alias("ratio_unique_to_total_values")
        )
    ).collect(streaming=True)

    result.write_csv(out_path)
    print(f"Saved source stats to: {out_path}")


def make_cumulative_count_chart(df, save_fpath):
    """
        Creates a cumulative count chart from a DataFrame and saves it as a PDF.

        Parameters:
        df (pd.DataFrame): DataFrame containing 'unique_samplingProtocol_value' and 'n_count'
        save_fpath (str): File path to save the generated PDF (e.g., 'output.pdf')
        """

    # 1. Exclude NaN values from the target category column
    df_clean = df.dropna(subset=['unique_samplingProtocol_value']).copy()
    # just in case some null values encoded as "NaN" strings
    df_clean = df_clean[df_clean['unique_samplingProtocol_value'] != 'NaN']

    # 2. Sort by frequency (n_count) in descending order
    df_clean = df_clean.sort_values('n_count', ascending=False).reset_index(drop=True)

    # 3. Create a numeric rank for the x-axis
    df_clean['rank'] = df_clean.index + 1

    # 4. Calculate the cumulative sum of the counts
    df_clean['cumulative_count'] = df_clean['n_count'].cumsum() / df_clean['n_count'].sum()

    total_val = df_clean['n_count'].sum()

    # 5. Determine a good regular interval for x-axis ticks based on dataframe size
    # For ~30,000 rows, this will place a tick roughly every 3000 ranks
    dtick_interval = 3000  # max(1, len(df_clean) // 10)

    # 6. Create the Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_clean['rank'],
        y=df_clean['cumulative_count'],
        mode='lines',
        line=dict(width=3, color='royalblue'),
        name='Cumulative proportion out of total no. of<br>non-null samplingProtocol values'
    ))

    # 7. Update layout for regular interval numeric ticks and formatting
    fig.update_layout(
        xaxis_title="samplingProtocol value's numeric rank<br>sorted by descending frequency",
        yaxis_title="Cumulative proportion out of total no. of<br>non-null samplingProtocol values",
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=dtick_interval,
            showgrid=True
        ),
        yaxis=dict(
            showgrid=True
        ),
    )

    # 8. Save the figure as a PDF
    fig.write_image(save_fpath)
    print(f"Figure successfully saved to {save_fpath}")

def export_top_n_vocab(df, output_dir, n=100):
    top_n_df = df.copy()
    top_n_df["proportion_of_all_values_incl_null"] = top_n_df["n_count"] / top_n_df["n_count"].sum()
    top_n_df.head(n).to_csv(output_dir/f"top_{n}_samplingProtocol_vocab.csv", index=False)
    print(f"Top {n} vocab exported")

def export_cumulative_stats(df, output_dir, top_n_marks=(1, 50, 100, 500, 1000, 5000, 10000)):
    top_n_df = df.copy()
    top_n_df["proportion_of_all_values_incl_null"] = top_n_df["n_count"] / top_n_df["n_count"].sum()

    output_df = []
    null_proportion = top_n_df.head(1)["proportion_of_all_values_incl_null"][0]

    for n in top_n_marks:
        output_df.append({"top_n": n,
                          "cumulative_count": top_n_df.head(n)["n_count"].sum(),
                          "cumulative_proportion_of_all_values_incl_null": top_n_df["proportion_of_all_values_incl_null"].head(n).sum(),
                          "cumulative_proportion_of_all_values_excl_null": (top_n_df["proportion_of_all_values_incl_null"].head(n).sum()-null_proportion)/(1-null_proportion)})

    output_df.append({"top_n": len(top_n_df),
                      "cumulative_count": top_n_df["n_count"].sum(),
                      "cumulative_proportion_of_all_values_incl_null": 1.0,
                      "cumulative_proportion_of_all_values_excl_null": 1.0})

    pd.DataFrame(output_df).to_csv(output_dir/f"samplingProtocol_vocab_cumulative_stats.csv", index=False)
    print("samplingProtocol_vocab_cumulative_stats csv exported")




if __name__ == "__main__":
    df_fpaths, output_dir = get_samplingProtocol_column_data()

    aggregate_analysis_output_dir = output_dir.parent / "aggregate_analysis"
    aggregate_analysis_output_dir.mkdir(exist_ok=True)

    unique_count_csv_fpath = generate_master_value_counts(df_fpaths, aggregate_analysis_output_dir)

    stats_csv_fpath = generate_master_source_stats(df_fpaths, aggregate_analysis_output_dir)

    unique_count_df = pd.read_csv(unique_count_csv_fpath)

    # make_cumulative_count_chart(unique_count_df, aggregate_analysis_output_dir/f"cumulative_samplingProtocol_value_counts.pdf")

    # export_top_n_vocab(unique_count_df, aggregate_analysis_output_dir, n=500)

    export_cumulative_stats(unique_count_df, aggregate_analysis_output_dir)