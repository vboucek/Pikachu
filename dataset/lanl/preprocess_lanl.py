#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import random
import os


def preprocess_lanl_data(auth_file, redteam_file, output_file, chunksize=10000):
    """
    Reads the raw auth file in chunks, filters rows, merges with redteam data
    to label anomalies, and writes the processed data incrementally to output_file.
    This avoids loading the entire dataset into memory.

    The processed CSV will contain 8 columns:
      timestamp, src_user, dst_user, src_computer, dst_computer, orig_index, label, snapshot
    """
    col_names = ['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer',
                 'auth_type', 'logon_type', 'log_action', 'log_status']

    # Compile regex patterns:
    # Remove unwanted accounts (ANONYMOUS, LOCAL, NETWORK, ADMIN) and computer accounts ending with '$'
    unwanted_pattern = re.compile(r'^(?:ANONYMOUS|LOCAL|NETWORK|ADMIN)(?!$)')
    computer_account_pattern = re.compile(r'\$$')

    # Load redteam data once
    rt_df = pd.read_csv(redteam_file, header=0)
    rt_df.columns = ['timestamp', 'src_user', 'src_computer', 'dst_computer']
    filter_cols = ['timestamp', 'src_user', 'src_computer', 'dst_computer']

    # Remove output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    orig_idx = 0  # to track original row numbers
    first_chunk = True
    # Process the raw file in chunks
    for chunk in tqdm(pd.read_csv(auth_file,
                                  header=None,
                                  names=col_names,
                                  usecols=['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer',
                                           'auth_type', 'logon_type'],
                                  dtype={'timestamp': np.int32, 'src_user': str, 'dst_user': str,
                                         'src_computer': str, 'dst_computer': str,
                                         'auth_type': 'category', 'logon_type': 'category'},
                                  chunksize=chunksize),
                      desc="Preprocessing chunks"):
        chunk['orig_index'] = range(orig_idx, orig_idx + len(chunk))
        orig_idx += len(chunk)

        # Filter rows with missing auth or logon type
        mask = (chunk['auth_type'] != '?') & (chunk['logon_type'] != '?')
        chunk = chunk.loc[mask]

        # Filter out rows where src_user matches unwanted patterns or ends with '$'
        chunk = chunk[~chunk['src_user'].str.contains(unwanted_pattern)]
        chunk = chunk[~chunk['src_user'].str.contains(computer_account_pattern)]

        # Exclude rows where src_computer equals dst_computer
        chunk = chunk.loc[chunk['src_computer'] != chunk['dst_computer']]

        # **Do not drop dst_user** so we can downsample from both users.
        chunk = chunk.drop(['auth_type', 'logon_type'], axis=1)

        # Merge with redteam data to mark anomalies.
        merged = pd.merge(chunk, rt_df, how='inner', on=filter_cols, suffixes=('', '_rt'))
        anom_indices = merged['orig_index'].tolist()

        # Create label: default 0, set to 1 if row's index (i.e. original index) is in anom_indices
        chunk['label'] = 0
        chunk.loc[chunk.index.isin(anom_indices), 'label'] = 1

        # Compute snapshot: difference from minimum timestamp (use global minimum from first chunk)
        if first_chunk:
            initial_time = chunk['timestamp'].min()
            with open(output_file + ".init_time", "w") as f:
                f.write(str(initial_time))
            first_chunk = False
        else:
            with open(output_file + ".init_time", "r") as f:
                initial_time = int(f.read().strip())
        # Use 1 hour snapshots
        chunk['snapshot'] = (chunk['timestamp'] - initial_time) // 3600

        # Reorder columns to:
        # timestamp, src_user, dst_user, src_computer, dst_computer, orig_index, label, snapshot
        chunk = chunk[
            ['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'orig_index', 'label', 'snapshot']]

        # Append this processed chunk to the output CSV (write header only for the first chunk)
        chunk.to_csv(output_file, mode='a', index=False, header=first_chunk)

    print(f"Preprocessed data saved to {output_file}")


def lanl_user_subset(input_csv, output_csv, chunksize=10000):
    """
    Downsamples the dataset in a memory-efficient way:
      1. First pass: Determine unique users (from both 'src_user' and 'dst_user')
         and anomalous users.
      2. Sample normal users at a 1:20 ratio (20 normal per anomalous).
      3. Second pass: Write out only rows where either src_user or dst_user is in the selected user set.

    """

    # Expected processed columns (8 columns)
    PROCESSED_COLS = ['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'orig_index', 'label',
                      'snapshot']

    all_users = set()
    anom_users = set()

    # Pass 1: Collect unique users from both src_user and dst_user
    for chunk in tqdm(pd.read_csv(input_csv, header=0, names=PROCESSED_COLS, chunksize=chunksize),
                      desc="Collecting unique users"):
        chunk['src_user'] = chunk['src_user'].astype(str)
        chunk['dst_user'] = chunk['dst_user'].astype(str)
        users = set(chunk['src_user'].unique()) | set(chunk['dst_user'].unique())
        all_users.update(users)
        anom_chunk = chunk[chunk['label'] == True]
        a_users = set(anom_chunk['src_user'].unique()) | set(anom_chunk['dst_user'].unique())
        anom_users.update(a_users)
    print("Total users in dataset:", len(all_users))
    print("Anomalous users found:", len(anom_users))

    # Normal users = all_users minus anomalous users
    normal_users = list(all_users - anom_users)
    # Sample 20 times as many normal users as anomalous users
    if len(anom_users) * 20 > len(normal_users):
        sample_normal = normal_users
    else:
        sample_normal = random.sample(normal_users, len(anom_users) * 20)
    selected_users = set(sample_normal) | anom_users
    print("Total users after downsampling:", len(selected_users))

    # Pass 2: Write only rows where either src_user or dst_user is in the selected set.
    if os.path.exists(output_csv):
        os.remove(output_csv)
    first_chunk = True
    for chunk in tqdm(pd.read_csv(input_csv, header=0, names=PROCESSED_COLS, chunksize=chunksize),
                      desc="Filtering downsampled data"):
        chunk['src_user'] = chunk['src_user'].astype(str)
        chunk['dst_user'] = chunk['dst_user'].astype(str)
        filtered = chunk[chunk['src_user'].isin(selected_users) | chunk['dst_user'].isin(selected_users)]
        filtered.to_csv(output_csv, mode='a', index=False, header=first_chunk)
        first_chunk = False
    print(f"Downsampled data saved to {output_csv}")


if __name__ == "__main__":
    # Change paths if necessary
    auth_file = "auth.txt"
    redteam_file = "redteam.txt"

    processed_csv = "auth_all_anom_1hr.csv"
    downsampled_csv = "anom_full_20xuser_1hr.csv"

    preprocess_lanl_data(auth_file, redteam_file, processed_csv, chunksize=10000)

    # Downsample the processed data using a 1:20 ratio, considering both src_user and dst_user.
    lanl_user_subset(processed_csv, downsampled_csv, chunksize=10000)
