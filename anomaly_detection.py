# ******************************************************************************
# anomaly_detection.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/11/21   Paudel     Initial version,
# ******************************************************************************
import os

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, log_loss, classification_report
from sklearn import metrics
import numpy as np
import pandas as pd
import random
from scipy.special import softmax
import pickle
from multiprocessing import Pool

from timeit import default_timer as timer

def aggregate_neighbors(node_emb, node_list, u, n_u):
    CNu = [node_emb[node_list.index(str(n))] for n in n_u]
    Cu = node_emb[node_list.index(str(u))]
    H = 1 / (1 + len(CNu)) * np.add(Cu, sum(Cn for Cn in CNu))
    return np.array(H)


def calculate_edge_probability(w, node_emb, node_list, G):
    """
    Vectorized version:
    - Precomputes a node-to-index mapping.
    - Caches aggregated embeddings for every node in G.
    - Builds arrays for the source and target aggregated embeddings for all edges.
    - Computes the softmax predictions in batch.
    - Computes an edge score for each edge.
    """
    # Precompute node2idx for fast lookup
    node2idx = {str(n): i for i, n in enumerate(node_list)}

    # Cache aggregated embeddings for each node in G
    agg_cache = {}
    for node in tqdm(G.nodes(), desc="Caching aggregated embeddings"):
        node_str = str(node)
        idx = node2idx[node_str]
        Cu = node_emb[idx]
        neighbors = list(G.neighbors(node))
        if neighbors:
            CNu = [node_emb[node2idx[str(n)]] for n in neighbors]
            agg = (Cu + sum(CNu)) / (1 + len(CNu))
        else:
            agg = Cu
        agg_cache[node_str] = agg

    # Get all edges (with data) from the graph
    edges = list(G.edges(data=True))
    num_edges = len(edges)
    if num_edges == 0:
        return []

    # Determine embedding dimension from node_emb
    dim = node_emb.shape[1]

    # Preallocate arrays for aggregated embeddings and index arrays
    agg_u = np.empty((num_edges, dim), dtype=node_emb.dtype)
    agg_v = np.empty((num_edges, dim), dtype=node_emb.dtype)
    u_idx_arr = np.empty(num_edges, dtype=np.int64)
    v_idx_arr = np.empty(num_edges, dtype=np.int64)

    # Also collect additional edge data for later use
    snapshots = []
    times = []
    labels = []
    us = []
    vs = []

    # Build arrays for all edges with progress printing
    for i, (u, v, data) in enumerate(tqdm(edges, desc="Building edge arrays")):
        u_str = str(u)
        v_str = str(v)
        agg_u[i, :] = agg_cache[u_str]
        agg_v[i, :] = agg_cache[v_str]
        u_idx_arr[i] = node2idx[u_str]
        v_idx_arr[i] = node2idx[v_str]
        snapshots.append(data['snapshot'])
        times.append(data['time'])
        labels.append(data['anom'])
        us.append(u)
        vs.append(v)

    # Compute prediction probabilities in batch using the dot product and softmax.
    P_source = softmax(np.dot(agg_u, w.T), axis=1)
    P_target = softmax(np.dot(agg_v, w.T), axis=1)

    # Compute the edge score for each edge.
    scores = ((1 - P_source[np.arange(num_edges), v_idx_arr]) +
              (1 - P_target[np.arange(num_edges), u_idx_arr])) / 2

    # Build and return the list of edge scores with progress printing
    edge_scores = []
    for i in tqdm(range(num_edges), desc="Building final edge score list"):
        edge_scores.append([us[i], vs[i], scores[i], snapshots[i], times[i], labels[i]])
    return edge_scores

class AnomalyDetection:
    def __init__(self, args, node_list, node_map, node_embeddings, idx):
        self.args = args
        self.node_list = node_list
        self.node_map = node_map
        self.node_embeddings = node_embeddings
        self.idx = idx

    def get_ip(self, node):
        for k, v in self.node_map.items():
            if v == node:
                return k
        return None

    def aggregate_neighbors_object(self, u, n_u):
        node_index_u = self.node_list.index(str(u))
        # Get the embedding for node u at time self.idx
        Cu = self.node_embeddings[node_index_u, self.idx, :]
        # Get embeddings for neighbor nodes at time self.idx
        CNu = [self.node_embeddings[self.node_list.index(str(n)), self.idx, :] for n in n_u]
        # Compute the aggregate (here simply the average of u and its neighbors)
        H = 1 / (1 + len(CNu)) * (Cu + sum(CNu))
        return H

    def initialize_parameters(self, k, v):
        w = np.random.randn(v, k) * 0.0001
        return w

    def propagate(self, w, X, Y):
        m = X.shape[1]
        #calculate activation function
        p = softmax(np.dot(X, w.T))
        # find the cost (cross entropy)
        cost = log_loss(Y, p)

        # find gradient (back propagation)
        dw = (1 / m) * np.dot((p - Y).T, X)

        cost = np.squeeze(cost)
        grads = {"dw": dw}
        return grads, cost

    def gradient_descent(self, w, X, Y, iterations, learning_rate):
        costs = []
        for i in range(iterations):
            grads, cost = self.propagate(w, X, Y)
            # update parameters
            w = w - learning_rate * grads["dw"]
            costs.append(cost)
            print("Cost after iteration %i/%i:      %f" % (i, iterations, cost))
        params = {"w": w}
        return params, costs

    def predict(self, w, X):
        return softmax(np.dot(X, w.T))

    def get_train_edges(self, train_graphs, s = 10):
        data_x = []
        data_y = []
        for G in tqdm(train_graphs):
            print("G: ", len(G.nodes()), len(G.edges()))
            for u in G.nodes():
                N = [n for n in G.neighbors(u)]
                for v in N:
                    if len(N) > 1:
                        n_minus_v = [n for n in N if n != v]
                        support_set = random.choices(n_minus_v, k=s)
                    else:
                        support_set = random.choices(N, k=s)
                    H = self.aggregate_neighbors_object(u, support_set)
                    y = np.zeros(len(self.node_list))
                    y[v] = 1
                    data_x.append(H)
                    data_y.append(y)
            self.idx += 1

        return np.array(data_x), np.array(data_y)

    def print_result(self, percentile, threshold, true_label, pred_label):
        print("\n====BEST ANOMALY DETECTION RESULTS====")
        print("Percentile : ", percentile, " Threshold : ", threshold)
        print("---------------------------------------\n")
        print(metrics.classification_report(true_label, pred_label))
        print("Confusion Matrix: \n", confusion_matrix(true_label, pred_label, labels=[False, True]))
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()
        print("(tn, fp, fn, tp): ", tn, fp, fn, tp)

    def calculate_performance_metrics(self, edge_scores, result_file):
        print("\n\nCalculating Performance Metrices....")
        true_label = list(edge_scores['label'])
        scores = list(edge_scores['score'])
        fpr, tpr, thresholds = metrics.roc_curve(true_label, scores, pos_label=1)
        # print("FPR: ", list(fpr))
        # print("TPR: ", list(tpr))
        fw = 0.5
        tw = 1 - fw
        fn = np.abs(tw * tpr - fw * (1 - fpr))
        best = np.argmin(fn, 0)
        print("\n\nOptimal cutoff %0.10f achieves TPR: %0.5f FPR: %0.5f on train data"
              % (thresholds[best], tpr[best], fpr[best]))
        print("Final AUC: ", metrics.auc(fpr, tpr))
        print("AUC: ", metrics.roc_auc_score(true_label, scores))
        edge_scores['pred'] = np.where(edge_scores['score'] >= thresholds[best], True, False)
        true_label = list(edge_scores['label'])
        pred_label = list(edge_scores['pred'])
        print("\n\n======= CLASSIFICATION REPORT =========\n")
        print(classification_report(true_label, pred_label))
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()
        print("Confusion Matrix: \n", confusion_matrix(true_label, pred_label, labels=[False, True]))
        print("FPR: ", fp / (fp + tn))
        print("TPR: ", tp / (tp + fn))

        reported_anom = edge_scores[edge_scores['pred'] == True]
        reported_anom['src'] = reported_anom['src'].apply(self.get_ip)
        reported_anom['dest'] = reported_anom['dest'].apply(self.get_ip)
        reported_anom.to_csv("results/" + result_file)

    def anomaly_detection(self, graphs, param_file):
        print("\n\nEstimating Edge Probability Distribution....")
        learning_rates = [0.001] #[0.1, 0.01, 0.001, 0.0001, 0.00001]
        support_sets = [10]#[2, 5, 15, 20, 25]
        for lr in learning_rates:
            for s in support_sets:
                self.args.alpha = lr
                self.args.support = s
                prob_param_file = param_file + '_' + str(self.args.alpha) + '_' + str(self.args.support) + '.pickle'
                print("++++++++++ Parameters +++++++ ")
                print("Learning Rate: ", lr)
                print("# of Support Set: ", s)
                print("Param File: ", prob_param_file)
                if self.args.train:
                    w = self.initialize_parameters(self.args.dimensions, len(self.node_list))

                    print("\n\nGenerating Training Edges....")
                    self.idx = 0
                    train_x, train_y = self.get_train_edges(graphs[:self.args.trainwin], self.args.support)

                    print("\n\nStarting Gradient Descent....")
                    parameters, costs = self.gradient_descent(w, train_x, train_y, self.args.iter, self.args.alpha)
                    w = parameters["w"]

                    with open(prob_param_file, 'wb') as f:
                        pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

                with open(prob_param_file, 'rb') as f:
                    param = pickle.load(f)
                w = param['w']
                total_cpu = os.cpu_count()
                print("\nNumber of CPU Available: ", total_cpu)

                graph_tuple = [(w, self.node_embeddings[:, self.args.trainwin + idx, :], self.node_list, G)
                               for idx, G in enumerate(graphs[self.args.trainwin:])]
                s_time = timer()
                with Pool(total_cpu) as pool:
                    all_graph_edges = pool.starmap(calculate_edge_probability, graph_tuple)
                pool.close()
                print("\nEdge Probability Estimation Completed...   [%s Sec.]" % (timer() - s_time))
                edge_scores = [edges for g_edges in all_graph_edges for edges in g_edges]
                edge_scores = pd.DataFrame(edge_scores, columns=['src', 'dest',  'score', 'snapshot', 'time', 'label'])
                result_file = self.args.dataset + '_d' + str(self.args.dimensions) + 'all_users.csv'
                self.calculate_performance_metrics(edge_scores, result_file)