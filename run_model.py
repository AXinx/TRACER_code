import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import time

from castle.algorithms import PC
from castle.algorithms import DirectLiNGAM
from castle.algorithms import DAG_GNN

from sknetwork.ranking import PageRank
import logging
logging.getLogger().setLevel(logging.WARNING)

# Ac@k
# Ac@k
def ac_at_k(pre_rc, true_rc, k):
    pre_rc_at_k = pre_rc[:k]
    relevant_count = sum(item in true_rc for item in pre_rc_at_k)
    return relevant_count / min(k, len(true_rc))

# Avg@k
def avg_at_k(pre_rc, true_rc, k):
    if k <= len(pre_rc):
        k_list = list(range(k))
        ac_at_k_values = [ac_at_k(pre_rc, true_rc, i+1) for i in k_list]
        avg_at_k = sum(ac_at_k_values) / len(k_list) if len(k_list) > 0 else 0
    else:
        k_list = list(range(len(pre_rc)))
        ac_at_k_values = [ac_at_k(pre_rc, true_rc, i+1) for i in k_list]
        avg_at_k = sum(ac_at_k_values) / len(k_list) if len(k_list) > 0 else 0
    return avg_at_k

res_ad = []
res_pc = []
res_lingam = []
res_daggnn = []

services = ['cartservice', 'checkoutservice', 'currencyservice', 'paymentservice', 'productcatalogservice']
ad_type = ['cpu', 'delay', 'loss', 'mem']
rcs = ['cpu', 'latency-90', 'latency-90', 'mem']
ssi = 0
aaj = 0
for ts in range(1,6):
    #Read data
    df_data = pd.read_csv('./data/fse-ob/'+services[ssi]+'_'+ad_type[aaj]+'/'+str(ts)+'/simple_data.csv', delimiter=',', encoding='utf-8')
    df_data = df_data.loc[:, ~df_data.columns.str.endswith("_latency-50")]
    df_data.fillna(0, inplace=True)
    
    with open('./data/fse-ob/'+services[ssi]+'_'+ad_type[aaj]+'/'+str(ts)+'/inject_time.txt', 'r') as f:
        inject_time = int(f.read().strip())
    inject_indices = df_data[df_data['time'] == inject_time].index[0]
    
    normal_start = 0
    normal_end = inject_indices
    abn_start = inject_indices
    abn_end = len(df_data)+1-300
    true_rc = [services[ssi]+'_'+rcs[aaj]]
    
    norm_rows = df_data.iloc[normal_start: normal_end]
    abn_rows = df_data.iloc[abn_start: abn_end]
    epos=30
    wds=30
    
    
    #Data preprocessing
    #Select columns with changes
    filtered_cols = df_data.loc[:, (df_data.nunique() > 1)]
    filtered_cols = filtered_cols.drop(columns=['time'], errors='ignore')

    filt_cols = filtered_cols.columns
    
    scaler = StandardScaler()
    scaler.fit(norm_rows[filt_cols])
    
    all_data = pd.concat([norm_rows, abn_rows], axis=0)[filt_cols]
    norm_df = pd.DataFrame(scaler.transform(all_data[filt_cols]), columns=filt_cols, index=all_data.index)
  
    #AE
    def create_sliding_windows(data, window_size, step_size):
        windows = []
        for i in range(0, len(data) - window_size + 1, step_size):
            windows.append(data[i:i+window_size])
        return np.array(windows)
    
    abn_columns = []
    abn_scores = {}
    train_time_total = []
    test_time_total = []
    for i in filtered_cols.columns:
        normal_data = norm_df.iloc[0: len(norm_rows)][i].values
        anomalous_data = norm_df.iloc[len(norm_rows): ][i].values
        window_size = wds
        step_size = 1
    
        # Create sliding windows for both normal and anomalous data
        normal_windows = create_sliding_windows(normal_data, window_size, step_size)
        anomalous_windows = create_sliding_windows(anomalous_data, window_size, step_size)
    
        # Define the autoencoder model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(window_size,)),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(window_size, activation='sigmoid')
        ])
        
        # Set the learning rate in the Adam optimizer
        learning_rate = 0.005  # You can adjust this value as needed
        optimizer = Adam(learning_rate=learning_rate)
    
        # Compile the model with the custom learning rate
        model.compile(optimizer=optimizer, loss='mse')
    
        # Custom callback to save the loss values at each epoch
        class LossHistory(Callback):
            def on_train_begin(self, logs=None):
                self.losses = []
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.losses.append(loss)
    
        # Instantiate the custom callback
        history = LossHistory()
        try:
            # Train the autoencoder on normal data
            train_start_time = time.time()
            model.fit(normal_windows, normal_windows, epochs=epos, batch_size=32, verbose=0, callbacks=[history]) #verbose=1: print loss
            train_end_time = time.time()
            # Test normal data
            test_start_time = time.time()
            normal_reconstructions = model.predict(normal_windows, verbose=0)
            normal_errors = np.mean(np.square(normal_windows - normal_reconstructions), axis=1)
            avg_n_error = np.mean(normal_errors)
            n_variance = np.var(normal_errors)
    
            # Predict reconstruction errors for anomalous data
            anomalous_reconstructions = model.predict(anomalous_windows, verbose=0)
            anomalous_errors = np.mean(np.square(anomalous_windows - anomalous_reconstructions), axis=1)
            avg_abn_error = np.mean(anomalous_errors)
    
            # Define threshold
            lower = avg_n_error - 3 * np.sqrt(n_variance)
            upper = avg_n_error + 3 * np.sqrt(n_variance)

            # Determine abnormal column
            if lower < avg_abn_error < upper:
                print(f"Column {i} is normal")
            else:
                print(f"Column {i} is abnormal")
                abn_columns.append(i)
                abn_scores[i] = avg_abn_error
            ad_end_time = time.time()
            train_time = train_end_time - train_start_time
            test_time = ad_end_time - test_start_time
            train_time_total.append(train_time)
            test_time_total.append(test_time)
        except:
            continue
    abn_data = norm_df.iloc[abn_start: abn_end]
    X_org = abn_data[abn_columns]
    stds = X_org.std()
    X = X_org.loc[:, stds >= 1e-8]
            
    #PC
    try:
        pc_start_time = time.time()
        pc = PC()
        pc.learn(X)
    
        adj = pc.causal_matrix # pc has no weight_causal_matrix
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        mapping = {i: name for i, name in enumerate(X.columns.tolist())}
        G = nx.relabel_nodes(G, mapping)
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        # PageRank
        total = sum(abn_scores.values())
        personalization = {node: abn_scores.get(node, 0.0) / total for node in G.nodes}
        
        scores = nx.pagerank(G, alpha=0.85, personalization=personalization, weight='weight')
        sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print('PC')
        print(sorted_res)
        
        pc_end_time = time.time()
    
        # Evaluation
        pre_rc = [ele[0] for ele in sorted_res]
        
        k = 3
        ac_k_3 = ac_at_k(pre_rc, true_rc, k)
        avg_k_3 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_3:.4f}')
        # print(f'Avg@{k}: {avg_k_3:.4f}')
        
        k = 5
        ac_k_5 = ac_at_k(pre_rc, true_rc, k)
        avg_k_5 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_5:.4f}')
        # print(f'Avg@{k}: {avg_k_5:.4f}')
        
        k = 10
        ac_k_10 = ac_at_k(pre_rc, true_rc, k)
        avg_k_10 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_10:.4f}')
        # print(f'Avg@{k}: {avg_k_10:.4f}')
        # print('--------------------------------')
        
        train_time_avg = np.mean(train_time_total)
        test_time_avg = np.mean(test_time_total) + (pc_end_time - pc_start_time)
        each_res = [ac_k_3, avg_k_3, ac_k_5, avg_k_5, ac_k_10, avg_k_10, train_time_avg, test_time_avg]
        f_each_res = [f"{x:.4f}" for x in each_res]
        res_pc.append(f_each_res)
    except:
        print('The input matrix is empty.')
        res_pc.append([0,0,0,0,0,0,0,0])
        pass

    #LiNGAM
    try:
        lingam_start_time = time.time()
        g = DirectLiNGAM()
        g.learn(X)
        adj = g.weight_causal_matrix # g.weight_causal_matrix
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        mapping = {i: name for i, name in enumerate(X.columns.tolist())}
        G = nx.relabel_nodes(G, mapping)
        
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        # PageRank
        total = sum(abn_scores.values())
        personalization = {node: abn_scores.get(node, 0.0) / total for node in G.nodes}
        
        scores = nx.pagerank(G, alpha=0.85, personalization=personalization, weight='weight')
        sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print('LiNGAM')
        print(sorted_res)
        
        lingam_end_time = time.time()
        
        # Evaluation
        pre_rc = [ele[0] for ele in sorted_res]
        
        k = 3
        ac_k_3 = ac_at_k(pre_rc, true_rc, k)
        avg_k_3 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_3:.4f}')
        # print(f'Avg@{k}: {avg_k_3:.4f}')
        
        k = 5
        ac_k_5 = ac_at_k(pre_rc, true_rc, k)
        avg_k_5 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_5:.4f}')
        # print(f'Avg@{k}: {avg_k_5:.4f}')
        
        k = 10
        ac_k_10 = ac_at_k(pre_rc, true_rc, k)
        avg_k_10 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_10:.4f}')
        # print(f'Avg@{k}: {avg_k_10:.4f}')
        # print('--------------------------------')
        
        train_time_avg = np.mean(train_time_total)
        test_time_avg = np.mean(test_time_total) + (lingam_end_time - lingam_start_time)
        each_res = [ac_k_3, avg_k_3, ac_k_5, avg_k_5, ac_k_10, avg_k_10, train_time_avg, test_time_avg]
        f_each_res = [f"{x:.4f}" for x in each_res]
        res_lingam.append(f_each_res)
    except:
        print('The input matrix is empty.')
        res_lingam.append([0,0,0,0,0,0,0,0])
        pass

    #DAG_GNN
    try:
        gnn_start_time = time.time()
        gnn = DAG_GNN() #gnn = DAG_GNN(lr=1e-4, k_max_iter=50, dag_penalty=10.0)
        gnn.learn(X)
     
        adj = gnn.weight_causal_matrix # gnn.weight_causal_matrix
        np.fill_diagonal(adj, 0) # remove self-loop
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        mapping = {i: name for i, name in enumerate(X.columns.tolist())}
        G = nx.relabel_nodes(G, mapping)
        
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        # PageRank
        total = sum(abn_scores.values())
        personalization = {node: abn_scores.get(node, 0.0) / total for node in G.nodes}
        
        scores = nx.pagerank(G, alpha=0.85, personalization=personalization, weight='weight')
        sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print('DAG-GNN')
        print(sorted_res)
        
        gnn_end_time = time.time()
    
        # Evaluation
        pre_rc = [ele[0] for ele in sorted_res]
        
        k = 3
        ac_k_3 = ac_at_k(pre_rc, true_rc, k)
        avg_k_3 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_3:.4f}')
        # print(f'Avg@{k}: {avg_k_3:.4f}')
        
        k = 5
        ac_k_5 = ac_at_k(pre_rc, true_rc, k)
        avg_k_5 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_5:.4f}')
        # print(f'Avg@{k}: {avg_k_5:.4f}')
        
        k = 10
        ac_k_10 = ac_at_k(pre_rc, true_rc, k)
        avg_k_10 = avg_at_k(pre_rc, true_rc, k)
        # print(f'AC@{k}: {ac_k_10:.4f}')
        # print(f'Avg@{k}: {avg_k_10:.4f}')
        # print('--------------------------------')
        
        train_time_avg = np.mean(train_time_total)
        test_time_avg = np.mean(test_time_total) + gnn_end_time - gnn_start_time
        each_res = [ac_k_3, avg_k_3, ac_k_5, avg_k_5, ac_k_10, avg_k_10, train_time_avg, test_time_avg]
        f_each_res = [f"{x:.4f}" for x in each_res]
        res_daggnn.append(f_each_res)
    except:
        print('The input matrix is empty.')
        res_daggnn.append([0,0,0,0,0,0,0,0])
        pass
excel_res_ad = "\n".join(["\t".join(map(str, row)) for row in res_ad])
excel_res_pc = "\n".join(["\t".join(map(str, row)) for row in res_pc])
excel_res_lingam = "\n".join(["\t".join(map(str, row)) for row in res_lingam])
excel_res_daggnn = "\n".join(["\t".join(map(str, row)) for row in res_daggnn])

print(excel_res_ad)
print('--------------------------------')
print(excel_res_pc)
print('--------------------------------')
print(excel_res_lingam)
print('--------------------------------')
print(excel_res_daggnn)
