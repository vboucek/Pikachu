# ******************************************************************************
# utils.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 3/4/21   Paudel     Initial version,
# ******************************************************************************
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx


class GraphUtils:
    def __init__(self, node_map):
        self.node_map = node_map
        pass

    def embedding_hadamard(self, u, v):
        return u * v

    def embedding_l1(self, u, v):
        return np.abs(u - v)

    def embedding_l2(self, u, v):
        return (u - v) ** 2

    def embedding_avg(self, u, v):
        return (u + v) / 2.0

    def create_graph(self, snapshot_df):
        G = nx.MultiGraph()
        anom_node = []
        for index, row in snapshot_df.iterrows():
            scomp = row.src_computer
            dcomp = row.dst_computer
            # host_name = row.src_user #row.host_name
            time = index #row.timestamp
            gid = row.snapshot
            is_anomaly = False
            if row.label == 1:
                # print(row)
                is_anomaly = True
                if scomp not in anom_node:
                    anom_node.append(scomp)
                if dcomp not in anom_node:
                    anom_node.append(dcomp)

            G.add_node(self.node_map[scomp], anom= (scomp in anom_node))
            G.add_node(self.node_map[dcomp], anom= (dcomp in anom_node))
            G.add_edge(self.node_map[scomp], self.node_map[dcomp], time=time, anom=is_anomaly, snapshot = gid, weight=1)
        # print("Auth N: %d E: %d \n" % (G.number_of_nodes(), G.number_of_edges()))
        return G

class DataUtils:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def get_node_map(self, data_df):
        print("... Generating Node Map ... \n")
        node_map = {}
        node_id = 0
        for index, row in tqdm(data_df.iterrows()):
            scomp = row.src_computer
            dcomp = row.dst_computer
            if scomp not in node_map:
                node_map[scomp] = node_id
                node_id += 1
            if dcomp not in node_map:
                node_map[dcomp] = node_id
                node_id += 1
        return node_map

    def get_data(self):
        data_df = pd.read_csv(self.data_folder, header=0)
        node_df = data_df[['src_computer', 'dst_computer']]
        node_df = node_df.drop_duplicates()
        node_map = self.get_node_map(node_df)
        return data_df, node_map

    def get_node_label(graphs, node_list):
        node_labels = []
        for G in graphs:
            label = np.zeros((len(node_list), 1), dtype=np.float32)
            for n, data in G.nodes(data=True):
                label[node_list.index(str(n))] = data['anom']
            node_labels.append(label)
        node_labels = np.array(node_labels)
        # print("Node Label: ", node_labels.shape)
        return node_labels

    def get_node(node_map, n):
        for k, v in node_map.items():
            if v == n:
                return k
        return None

    def generate_seq_lookback(static_emb, lookback):
        X_train = []
        for sample in range(static_emb.shape[0]):
            for i in range(static_emb.shape[1] - lookback + 1):
                X_train.append(static_emb[sample, i:i + lookback, :])
        return np.array(X_train, dtype=np.float32)

    def generate_seq(static_emb):
        X_train, Y_train, = [], []
        for sample in range(static_emb.shape[0]):
            for i in range(static_emb.shape[1]):
                X_train.append(static_emb[sample, i, :])
        X_train = np.array(X_train, dtype=np.float32)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        return np.array(X_train, dtype=np.float32)

    def rollback_seq(dynamic_emb, total_node, batch_size):
        graph_emb = []
        for node_idx in range(total_node):
            offset = (node_idx + 1) * batch_size
            node_emb = dynamic_emb[node_idx*batch_size:offset,:]
            graph_emb.append(node_emb)
        return np.array(graph_emb, dtype=np.float32)
