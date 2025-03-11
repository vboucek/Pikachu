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
      - src_computer    (renamed from p_src)
      - dst_computer    (renamed from p_dst)
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
        # p_src -> src_computer, p_dst -> dst_computer, is_pivoting -> label
        chunk.rename(columns={'p_src': 'src_computer',
                              'p_dst': 'dst_computer',
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


def pivoting_host_subset(input_csv, output_csv, chunksize=10000):
    """
    Downsample (rebalance) the processed pivoting dataset so that the number of benign hosts
    (those that never appear in an anomalous row) is limited to 20× the number of malicious hosts.

    Process:
      1. First pass: Collect unique hosts from both 'src_computer' and 'dst_computer'
         and determine anomalous hosts (those that appear in rows with label==1).
      2. Sample normal hosts (all hosts not anomalous) at a 1:20 ratio relative to anomalous hosts.
      3. Second pass: Write out only rows where either src_computer or dst_computer is in the selected host set.
    """
    PROCESSED_COLS = ['timestamp', 'src_computer', 'dst_computer', 'orig_index', 'label', 'snapshot']
    all_hosts = set()
    anom_hosts = set()

    # Pass 1: Collect unique hosts and anomalous hosts
    for chunk in tqdm(pd.read_csv(input_csv, header=0, names=PROCESSED_COLS, chunksize=chunksize),
                      desc="Collecting unique hosts"):
        # Ensure the host columns are strings
        chunk['src_computer'] = chunk['src_computer'].astype(str)
        chunk['dst_computer'] = chunk['dst_computer'].astype(str)
        hosts = set(chunk['src_computer'].unique()) | set(chunk['dst_computer'].unique())
        all_hosts.update(hosts)
        # Consider a row anomalous if label==1
        anom_chunk = chunk[chunk['label'] == 1]
        a_hosts = set(anom_chunk['src_computer'].unique())
        anom_hosts.update(a_hosts)

    print("Total hosts in dataset:", len(all_hosts))
    print("Anomalous hosts found:", len(anom_hosts))

    # Normal hosts are those not in the anomalous set.
    normal_hosts = list(all_hosts - anom_hosts)
    # Sample 20 times as many normal hosts as there are anomalous hosts.
    if len(anom_hosts) * 20 > len(normal_hosts):
        sample_normal = normal_hosts
    else:
        sample_normal = random.sample(normal_hosts, len(anom_hosts) * 20)
    selected_hosts = set(sample_normal) | anom_hosts
    print("Total hosts after rebalancing:", len(selected_hosts))

    # Pass 2: Filter rows where either src_computer or dst_computer is in the selected host set.
    if os.path.exists(output_csv):
        os.remove(output_csv)
    first_chunk = True
    for chunk in tqdm(pd.read_csv(input_csv, header=0, names=PROCESSED_COLS, chunksize=chunksize),
                      desc="Filtering downsampled data"):
        chunk['src_computer'] = chunk['src_computer'].astype(str)
        chunk['dst_computer'] = chunk['dst_computer'].astype(str)
        filtered = chunk[chunk['src_computer'].isin(selected_hosts)]
        filtered.to_csv(output_csv, mode='a', index=False, header=first_chunk)
        first_chunk = False

    print(f"Downsampled data saved to {output_csv}")


if __name__ == "__main__":
    # File paths (update as necessary)
    input_file = "dataset_pivoting.csv"               # Original pivoting dataset file
    processed_file = "processed_pivoting.csv"   # Intermediate processed file
    downsampled_file = "pivoting_anom_full_100xuser_1hr.csv"  # Final output after rebalancing

    # First, process the raw pivoting file into the desired format.
    preprocess_pivoting_data(input_file, processed_file, chunksize=10000)
    # Then, perform host-based downsampling to rebalance benign vs. malicious hosts (20:1 ratio).
    pivoting_host_subset(processed_file, downsampled_file, chunksize=10000)
