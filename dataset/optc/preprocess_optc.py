#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random


def parse_timestamp(ts):
    """Try parsing a timestamp string with microseconds; if that fails, without."""
    try:
        return pd.to_datetime(ts, format="%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        return pd.to_datetime(ts, format="%Y-%m-%dT%H:%M:%S%z")


def preprocess_optc_data(input_file, output_file, chunksize=10000):
    """
    Process the optc dataset with the following structure:
      index,timestamp,src_ip,dest_ip,label

    The output CSV will contain:
      - timestamp       (original timestamp, as a string)
      - src_ip          (source IP)
      - dest_ip         (destination IP)
      - orig_index      (an incremental row index)
      - label           (anomaly label)
      - snapshot        (1-hour snapshot, computed as the number of complete hours
                         elapsed since the global minimum timestamp)

    Timestamps are parsed as datetime objects (using a helper function) and then
    converted to UTC for consistency.
    """

    # Remove the output file and any stored initial time if they exist
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(output_file + ".init_time"):
        os.remove(output_file + ".init_time")

    first_chunk = True
    orig_idx = 0

    # Read the CSV in chunks
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunksize,
                                  dtype={'timestamp': str, 'src_ip': str, 'dest_ip': str, 'label': np.int8}),
                      desc="Processing optc chunks"):
        # Drop the original 'index' column if present; we'll generate our own
        if 'index' in chunk.columns:
            chunk = chunk.drop(columns=['index'])

        # Add an incremental original index for bookkeeping
        chunk['orig_index'] = range(orig_idx, orig_idx + len(chunk))
        orig_idx += len(chunk)

        # Parse the timestamp column using the helper function and convert to UTC
        chunk['timestamp_dt'] = chunk['timestamp'].apply(parse_timestamp)
        chunk['timestamp_dt'] = chunk['timestamp_dt'].dt.tz_convert('UTC')

        # For the first chunk, determine and store the global minimum timestamp (in UTC)
        if first_chunk:
            initial_time = chunk['timestamp_dt'].min()
            # Store the timestamp as nanoseconds; then reload as tz-aware UTC
            with open(output_file + ".init_time", "w") as f:
                f.write(str(initial_time.value))
            write_header = True  # Write header for the very first chunk
            first_chunk = False
        else:
            with open(output_file + ".init_time", "r") as f:
                initial_time_ns = int(f.read().strip())
                initial_time = pd.to_datetime(initial_time_ns, utc=True)
            write_header = False

        # Compute snapshot: number of complete hours elapsed since initial_time.
        # Both timestamp_dt and initial_time are now tz-aware in UTC.
        chunk['snapshot'] = ((chunk['timestamp_dt'] - initial_time).dt.total_seconds() // 3600).astype(int)

        # Keep only the desired columns: timestamp, src_ip, dest_ip, orig_index, label, snapshot.
        chunk = chunk[['timestamp', 'src_ip', 'dest_ip', 'orig_index', 'label', 'snapshot']].copy()

        # Append the processed chunk to the output CSV.
        chunk.to_csv(output_file, mode='a', index=False, header=write_header)

    print(f"Processed optc data saved to {output_file}")


def optc_host_subset(input_csv, output_csv, chunksize=10000, ratio=20):
    """
    Downsample (rebalance) the processed optc dataset so that the number of benign hosts
    (those that never appear in an anomalous row) is limited to `ratio` times the number
    of anomalous hosts.

    Process:
      1. First pass: Collect unique hosts from both 'src_ip' and 'dest_ip'
         and determine anomalous hosts (rows where label == 1).
      2. Sample normal hosts (all hosts not anomalous) at a 1:ratio ratio relative to anomalous hosts.
      3. Second pass: Write out only rows where either src_ip or dest_ip is in the selected host set.
    """
    PROCESSED_COLS = ['timestamp', 'src_ip', 'dest_ip', 'orig_index', 'label', 'snapshot']
    all_hosts = set()
    anom_hosts = set()

    # Pass 1: Collect unique hosts and anomalous hosts
    for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunksize, names=PROCESSED_COLS, header=0),
                      desc="Collecting unique hosts from optc"):
        chunk['src_ip'] = chunk['src_ip'].astype(str)
        chunk['dest_ip'] = chunk['dest_ip'].astype(str)
        hosts = set(chunk['src_ip'].unique()) | set(chunk['dest_ip'].unique())
        all_hosts.update(hosts)
        # Rows with label==1 are considered anomalous.
        anom_chunk = chunk[chunk['label'] == 1]
        a_hosts = set(anom_chunk['src_ip'].unique())
        anom_hosts.update(a_hosts)

    print("Total hosts in optc dataset:", len(all_hosts))
    print("Anomalous hosts found:", len(anom_hosts))

    # Normal hosts are those not in the anomalous set.
    normal_hosts = list(all_hosts - anom_hosts)
    desired_normal = len(anom_hosts) * ratio
    if desired_normal > len(normal_hosts):
        sample_normal = normal_hosts
    else:
        sample_normal = random.sample(normal_hosts, desired_normal)
    selected_hosts = set(sample_normal) | anom_hosts
    print("Total hosts after rebalancing:", len(selected_hosts))

    # Pass 2: Filter rows where either src_ip or dest_ip is in the selected host set.
    if os.path.exists(output_csv):
        os.remove(output_csv)
    first_chunk_flag = True
    for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunksize, names=PROCESSED_COLS, header=0),
                      desc="Filtering downsampled optc data"):
        chunk['src_ip'] = chunk['src_ip'].astype(str)
        chunk['dest_ip'] = chunk['dest_ip'].astype(str)
        filtered = chunk[(chunk['src_ip'].isin(selected_hosts))]
        filtered.to_csv(output_csv, mode='a', index=False, header=first_chunk_flag)
        first_chunk_flag = False

    print(f"Downsampled optc data saved to {output_csv}")


if __name__ == "__main__":
    # Update these paths as necessary:
    input_file = "optc_sorted_labeled.csv"  # optc dataset file with columns: index,timestamp,src_ip,dest_ip,label
    processed_file = "processed_optc.csv"  # Intermediate processed file
    downsampled_file = "optc_anom_full_20xuser.csv"  # Final output after rebalancing

    # First, process the raw optc file.
    preprocess_optc_data(input_file, processed_file, chunksize=10000)
    # Then, perform host-based downsampling (rebalancing) to limit benign hosts.
    optc_host_subset(processed_file, downsampled_file, chunksize=10000, ratio=20)
