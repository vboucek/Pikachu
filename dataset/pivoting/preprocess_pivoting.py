#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random

def preprocess_pivoting_data(input_file, output_file, chunksize=10000):
    """
    Process the pivoting dataset with the following structure:
      ,src,dst,p_src,p_dst,b_in,b_out,d,timestamp,pkts_in,pkts_out,proto,is_pivoting

    The output CSV will contain:
      - timestamp       (original timestamp)
      - src_computer    (renamed from src)
      - dst_computer    (renamed from dst)
      - orig_index      (an incremental row index)
      - label           (from is_pivoting)
      - snapshot        (1-hour snapshot, computed as (timestamp - initial_time)//3600)

    The data is processed in chunks to reduce memory usage.
    """
    # Define the column names as they appear in the file.
    # The first (unnamed) column is used as the index.
    col_names = ['src', 'dst', 'p_src', 'p_dst', 'b_in', 'b_out', 'd',
                 'timestamp', 'pkts_in', 'pkts_out', 'proto', 'is_pivoting']

    # Remove output file and any stored initial time if they exist
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(output_file + ".init_time"):
        os.remove(output_file + ".init_time")

    first_chunk = True
    orig_idx = 0

    for chunk in tqdm(pd.read_csv(input_file,
                                  header=0,
                                  names=col_names,
                                  index_col=0,
                                  dtype={'src': str, 'dst': str,
                                         'p_src': str, 'p_dst': str,
                                         'b_in': np.int32, 'b_out': np.int32,
                                         'd': float, 'timestamp': float,
                                         'pkts_in': np.int32, 'pkts_out': np.int32,
                                         'proto': np.int8, 'is_pivoting': np.int8},
                                  chunksize=chunksize),
                      desc="Processing chunks"):
        # Add an incremental original index for bookkeeping
        chunk['orig_index'] = range(orig_idx, orig_idx + len(chunk))
        orig_idx += len(chunk)

        # Rename columns:
        # src -> src_computer, dst -> dst_computer, is_pivoting -> label
        chunk.rename(columns={'src': 'src_computer',
                              'dst': 'dst_computer',
                              'is_pivoting': 'label'}, inplace=True)

        # Keep only the desired columns
        chunk = chunk[['timestamp', 'src_computer', 'dst_computer', 'orig_index', 'label']].copy()

        # For the first chunk, determine and store the global minimum timestamp
        if first_chunk:
            initial_time = int(chunk['timestamp'].min())
            with open(output_file + ".init_time", "w") as f:
                f.write(str(initial_time))
            write_header = True  # Write header for the very first chunk
            first_chunk = False
        else:
            with open(output_file + ".init_time", "r") as f:
                initial_time = int(f.read().strip())
            write_header = False

        # Compute snapshot: number of complete hours elapsed since initial_time
        chunk.loc[:, 'snapshot'] = ((chunk['timestamp'].astype(float) - initial_time) // 3600).astype(int)

        # Reorder columns: timestamp, src_computer, dst_computer, orig_index, label, snapshot
        chunk = chunk[['timestamp', 'src_computer', 'dst_computer', 'orig_index', 'label', 'snapshot']]

        # Append processed chunk to the output CSV
        chunk.to_csv(output_file, mode='a', index=False, header=write_header)

    print(f"Processed data saved to {output_file}")


def filter_pivoting_data(input_csv, output_csv, chunksize=10000, drop_ratio=0.9, filter_host="559"):
    PROCESSED_COLS = ['timestamp', 'src_computer', 'dst_computer', 'orig_index', 'label', 'snapshot']

    if os.path.exists(output_csv):
        os.remove(output_csv)
    first_chunk = True

    for chunk in tqdm(pd.read_csv(input_csv, header=0, names=PROCESSED_COLS, chunksize=chunksize),
                      desc="Filtering dataset"):
        chunk['src_computer'] = chunk['src_computer'].astype(str)
        chunk['dst_computer'] = chunk['dst_computer'].astype(str)

        # Separate rows where src_computer == filter_host and label == 0
        filter_mask = (chunk['src_computer'] == filter_host) & (chunk['label'] == 0)
        to_filter = chunk[filter_mask]
        keep_rows = chunk[~filter_mask]

        # Randomly select a subset to keep from the filtered rows
        if not to_filter.empty:
            keep_sample = to_filter.sample(frac=(1 - drop_ratio), random_state=42)
            filtered_chunk = pd.concat([keep_rows, keep_sample])
        else:
            filtered_chunk = keep_rows

        filtered_chunk.to_csv(output_csv, mode='a', index=False, header=first_chunk)
        first_chunk = False

    print(f"Filtered data saved to {output_csv}")


if __name__ == "__main__":
    # File paths (update as necessary)
    input_file = "dataset_pivoting.csv"               # Original pivoting dataset file
    processed_file = "processed_pivoting.csv"   # Intermediate processed file
    downsampled_file = "pivoting_downsampled_09_1h.csv"  # Final output after rebalancing

    # First, process the raw pivoting file into the desired format.
    #preprocess_pivoting_data(input_file, processed_file, chunksize=10000)
    # Most benign flows are from the host 559, balance the data a little by removing most of the traffic from this host
    filter_pivoting_data(processed_file, downsampled_file, chunksize=10000, drop_ratio=0.9, filter_host="559")
