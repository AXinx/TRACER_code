from RCAEval.e2e import baro
from RCAEval.e2e import circa
from RCAEval.e2e import rcd
from RCAEval.utility import download_data, read_data
import time

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

services = ['cartservice', 'checkoutservice', 'currencyservice', 'paymentservice', 'productcatalogservice']
ad_type = ['cpu', 'delay', 'loss', 'mem']
rcs = ['cpu', 'latency', 'latency', 'mem']
ssi = 0
aaj = 0

res_baro = []
res_circa = []
res_rcd = []
for ts in range(1,6):
    #Read data
    data = read_data('./data/fse-ob/'+services[ssi]+'_'+ad_type[aaj]+'/'+str(ts)+'/simple_data.csv')
   
    with open('./data/fse-ob/'+services[ssi]+'_'+ad_type[aaj]+'/'+str(ts)+'/inject_time.txt', 'r') as f:
        anomaly_detected_timestamp = int(f.read().strip())

    true_rc = [services[ssi]+'_'+rcs[aaj]]

    #BARO
    baro_s_time = time.time()
    root_causes = baro(data, anomaly_detected_timestamp)["ranks"]
    baro_e_time = time.time()
    baro_time = baro_e_time - baro_s_time
    pre_rc = root_causes
    
    k = 3
    ac_k_3 = ac_at_k(pre_rc, true_rc, k)
    avg_k_3 = avg_at_k(pre_rc, true_rc, k)
    
    k = 5
    ac_k_5 = ac_at_k(pre_rc, true_rc, k)
    avg_k_5 = avg_at_k(pre_rc, true_rc, k)
    
    k = 10
    ac_k_10 = ac_at_k(pre_rc, true_rc, k)
    avg_k_10 = avg_at_k(pre_rc, true_rc, k)
    each_res = [ac_k_3, avg_k_3, ac_k_5, avg_k_5, ac_k_10, avg_k_10, baro_time]
    f_each_res = [f"{x:.4f}" for x in each_res]
    res_baro.append(f_each_res)

    #CIRCA
    circa_s_time = time.time()
    root_causes = circa(data, anomaly_detected_timestamp)["ranks"]
    circa_e_time = time.time()
    circa_time = circa_e_time - circa_s_time
    pre_rc = root_causes
    
    k = 3
    ac_k_3 = ac_at_k(pre_rc, true_rc, k)
    avg_k_3 = avg_at_k(pre_rc, true_rc, k)
    
    k = 5
    ac_k_5 = ac_at_k(pre_rc, true_rc, k)
    avg_k_5 = avg_at_k(pre_rc, true_rc, k)
    
    k = 10
    ac_k_10 = ac_at_k(pre_rc, true_rc, k)
    avg_k_10 = avg_at_k(pre_rc, true_rc, k)
    
    each_res = [ac_k_3, avg_k_3, ac_k_5, avg_k_5, ac_k_10, avg_k_10, circa_time]
    f_each_res = [f"{x:.4f}" for x in each_res]
    res_circa.append(f_each_res)

    #RCD
    rcd_s_time = time.time()
    root_causes = rcd.rcd(data, anomaly_detected_timestamp)['ranks']
    rcd_e_time = time.time()
    rcd_time = rcd_e_time - rcd_s_time
    pre_rc = root_causes
    
    k = 3
    ac_k_3 = ac_at_k(pre_rc, true_rc, k)
    avg_k_3 = avg_at_k(pre_rc, true_rc, k)
    
    k = 5
    ac_k_5 = ac_at_k(pre_rc, true_rc, k)
    avg_k_5 = avg_at_k(pre_rc, true_rc, k)
    
    k = 10
    ac_k_10 = ac_at_k(pre_rc, true_rc, k)
    avg_k_10 = avg_at_k(pre_rc, true_rc, k)
    
    each_res = [ac_k_3, avg_k_3, ac_k_5, avg_k_5, ac_k_10, avg_k_10, rcd_time]
    f_each_res = [f"{x:.4f}" for x in each_res]
    res_rcd.append(f_each_res)

excel_res_baro = "\n".join(["\t".join(map(str, row)) for row in res_baro])
excel_res_circa = "\n".join(["\t".join(map(str, row)) for row in res_circa])
excel_res_rcd = "\n".join(["\t".join(map(str, row)) for row in res_rcd])

print(excel_res_baro)
print('--------------------------------')
print(excel_res_circa)
print('--------------------------------')
print(excel_res_rcd)
