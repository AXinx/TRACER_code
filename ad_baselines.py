import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.optimizers import Adam
from scipy.stats import genpareto, norm
import sys

# -------------------------------
# Evaluation
# -------------------------------
def evaluate(pre_rc, true_rc, all_cols, train_time, test_time):
    tp = len(set(pre_rc) & set(true_rc))
    fp = len(pre_rc) - tp
    fn = len(true_rc) - tp
    recall = tp / len(true_rc) if len(true_rc) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    filtering_rate = 1 - len(pre_rc) / len(all_cols)
    return [recall, precision, f1, filtering_rate, train_time, test_time, len(all_cols), len(pre_rc)]


def detect_nsigma(normal_data, abn_data, k=3):
    mean = np.mean(normal_data)
    std = np.std(normal_data)

    outlier_mask = (abn_data > mean + k * std) | (abn_data < mean - k * std)
    num_outliers = np.sum(outlier_mask)
    ratio = num_outliers / len(abn_data)
    
    return ratio > 0.05

# -------------------------------
# SPOT
# -------------------------------
def detect_spot(normal_data, abn_data, init_threshold=0.95, q=1e-4):
    threshold0 = np.quantile(normal_data, init_threshold)
    exceedances = normal_data[normal_data > threshold0] - threshold0
    if len(exceedances) < 2:
        abn_max = np.max(abn_data)
        return abn_max > threshold0

    c, loc, scale = genpareto.fit(exceedances)

    n_exceed = len(exceedances)
    n_total = len(normal_data)
    q_dynamic = 1 - q
    threshold_dyn = threshold0 + (scale / c) * ((n_total / n_exceed * (1 - q_dynamic)) ** (-c) - 1)

    abn_max = np.max(abn_data)
    return abn_max > threshold_dyn

# -------------------------------
# AE
# -------------------------------
def detect_ae(normal_data, abn_data, epochs=30, window=30):
    def create_windows(data, window_size):
        return np.array([data[i:i+window_size] for i in range(len(data)-window_size+1)])
    normal_w = create_windows(normal_data, window)
    abn_w = create_windows(abn_data, window)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(window,)),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(window, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.005), loss='mse')

    start_train = time.time()
    model.fit(normal_w, normal_w, epochs=epochs, batch_size=32, verbose=0)
    end_train = time.time()

    start_test = time.time()
    normal_recon = model.predict(normal_w, verbose=0)
    normal_err = np.mean(np.square(normal_w - normal_recon), axis=1)
    avg_n_err = np.mean(normal_err)
    n_variance = np.var(normal_err)
    
    abn_recon = model.predict(abn_w, verbose=0)
    abn_err = np.mean(np.square(abn_w - abn_recon), axis=1)
    avg_abn_err = np.mean(abn_err)
    
    lower = avg_n_err - 3 * np.sqrt(n_variance)
    upper = avg_n_err + 3 * np.sqrt(n_variance)
    
    if lower <= avg_abn_err <= upper:
        result = False
    else:
        result = True

    end_test = time.time()

    return result, end_train - start_train, end_test - start_test

# -------------------------------
# LSTM-AE
# -------------------------------
def detect_lstm_ae(normal_data, abn_data, window=30, epochs=30):
    def create_windows(data, window_size):
        return np.array([data[i:i+window_size] for i in range(len(data)-window_size+1)])

    normal_w = create_windows(normal_data, window)
    abn_w = create_windows(abn_data, window)
    normal_w = np.expand_dims(normal_w, -1)
    abn_w = np.expand_dims(abn_w, -1)

    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(window, 1)),
        RepeatVector(window),
        LSTM(64, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')

    start_train = time.time()
    model.fit(normal_w, normal_w, epochs=epochs, batch_size=32, verbose=0)
    end_train = time.time()

    start_test = time.time()
    normal_recon = model.predict(normal_w, verbose=0)
    normal_err = np.mean(np.square(normal_w - normal_recon), axis=1)
    avg_n_err = np.mean(normal_err)
    n_variance = np.var(normal_err)
    
    abn_recon = model.predict(abn_w, verbose=0)
    abn_err = np.mean(np.square(abn_w - abn_recon), axis=1)
    avg_abn_err = np.mean(abn_err)
    
    lower = avg_n_err - 3 * np.sqrt(n_variance)
    upper = avg_n_err + 3 * np.sqrt(n_variance)
    
    if lower <= avg_abn_err <= upper:
        result = False
    else:
        result = True

    end_test = time.time()

    return result, end_train - start_train, end_test - start_test

# -------------------------------
# Save to Excel
# -------------------------------
def run_experiment_excel_multi(methods=None, ssi=None, aaj=None):
    if methods is None:
        methods = ['n-sigma', 'spot', 'iforest', 'ocsvm', 'ae', 'lstm-ae']

    filename = f"experiment_results_{ssi}_{aaj}.xlsx"
    print(f"Running experiment: ssi={ssi}, aaj={aaj}, filename={filename}")

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for method in methods:
            services = ['cartservice', 'checkoutservice', 'currencyservice', 'paymentservice', 'productcatalogservice']
            ad_type = ['cpu', 'delay', 'loss', 'mem']
            rcs = ['cpu', 'latency-90', 'latency-90', 'mem']

            all_res = []
            for ite in range(3):
                for ts in range(1, 6):
                    df_data = pd.read_csv(f'../data/fse-ob/{services[ssi]}_{ad_type[aaj]}/{ts}/simple_data.csv')
                    df_data = df_data.loc[:, ~df_data.columns.str.endswith("_latency-50")]
                    df_data.fillna(0, inplace=True)

                    with open(f'../data/fse-ob/{services[ssi]}_{ad_type[aaj]}/{ts}/inject_time.txt', 'r') as f:
                        inject_time = int(f.read().strip())
                    inject_idx = df_data[df_data['time'] == inject_time].index[0]

                    normal_end = inject_idx
                    abn_start = inject_idx
                    abn_end = len(df_data) + 1 - 300
                    true_rc = [services[ssi] + '_' + rcs[aaj]]

                    filtered_cols = df_data.loc[:, (df_data.nunique() > 1)]
                    filtered_cols = filtered_cols.drop(columns=['time'], errors='ignore')
                    filt_cols = filtered_cols.columns

                    scaler = StandardScaler()
                    scaler.fit(df_data.iloc[:normal_end][filt_cols])
                    scaled_df = pd.DataFrame(scaler.transform(df_data[filt_cols]), columns=filt_cols)

                    abn_columns = []
                    train_times, test_times = [], []

                    for col in filt_cols:
                        x = scaled_df[col].values
                        normal_x = x[:normal_end]
                        abn_x = x[abn_start:abn_end]

                        start_time = time.time()
                        if method == 'ae':
                            result, t_train, t_test = detect_ae(normal_x, abn_x)
                        elif method == 'n-sigma':
                            t_train = 0
                            result = detect_nsigma(normal_x, abn_x)
                            t_test = time.time() - start_time
                        elif method == 'spot':
                            t_train = 0
                            result = detect_spot(normal_x, abn_x)
                            t_test = time.time() - start_time
                        elif method == 'bocpd':
                            t_train = 0
                            result = detect_bocpd(normal_x, abn_x)
                            t_test = time.time() - start_time
                        elif method == 'iforest':
                            t_train = 0
                            clf = IsolationForest(contamination=0.01, random_state=0)
                            clf.fit(normal_x.reshape(-1, 1))
                            preds = clf.predict(abn_x.reshape(-1, 1))
                            result = np.sum(preds == -1) > 5
                            t_test = time.time() - start_time
                        elif method == 'ocsvm':
                            t_train = 0
                            clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
                            clf.fit(normal_x.reshape(-1, 1))
                            preds = clf.predict(abn_x.reshape(-1, 1))
                            result = np.sum(preds == -1) > 5
                            t_test = time.time() - start_time
                        elif method == 'lstm-ae':
                            result, t_train, t_test = detect_lstm_ae(normal_x, abn_x)
                        else:
                            raise ValueError(f"Unknown method {method}")

                        train_times.append(t_train)
                        test_times.append(t_test)

                        if result:
                            abn_columns.append(col)

                    all_res.append(evaluate(abn_columns, true_rc, filt_cols, np.mean(train_times), np.mean(test_times)))

            df_res = pd.DataFrame(all_res, columns=['recall','precision','f1','filtering_rate','train_time','test_time','num_metrics','num_detected'])
            df_res.to_excel(writer, sheet_name=method[:31], index=False)
            print(f"==== Method: {method} ====")
            print(f"Results in sheet: {method[:31]}")
            print("--------------------------------")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_ss_ad_bench.py <ssi> <aaj>")
        sys.exit(1)

    ssi = int(sys.argv[1])
    aaj = int(sys.argv[2])
    run_experiment_excel_multi(ssi=ssi, aaj=aaj)
