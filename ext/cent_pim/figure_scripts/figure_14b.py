import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_QoS_file

# Load the data
df_processed_results = pd.read_csv('cent_simulation/processed_results.csv')

batch = [1, 2, 4, 8, 16, 80]
dict_CENT_70B = {}
dict_CENT_70B["latency"] = []      # minutes
dict_CENT_70B["throughput"] = []   # queries per minute

seqlen = 4096
num_devices = 32

for pp in batch[:-1]:
    df = df_processed_results[(df_processed_results['Model'] == 'Llama2-70B') & (df_processed_results['Device number'] == 32) & (df_processed_results['Seqlen'] == seqlen) & (df_processed_results['Pipeline parallelism'] == pp) & (df_processed_results['Tensor parallelism'] == num_devices // pp) & (df_processed_results['Phase'] == 'end2end')]

    dict_CENT_70B["latency"].append(df['Total Latency (s)'].mean().item() / 60)
    dict_CENT_70B["throughput"].append(df['Throughput (tokens/s)'].mean().item() * 60 / seqlen)

df = df_processed_results[(df_processed_results['Model'] == 'Llama2-70B') & (df_processed_results['Device number'] == 32) & (df_processed_results['Seqlen'] == seqlen) & (df_processed_results['Pipeline parallelism'] == 80) & (df_processed_results['Tensor parallelism'] == 1) & (df_processed_results['Phase'] == 'end2end')]

dict_CENT_70B["latency"].append(df['Total Latency (s)'].mean().item() / 60)
dict_CENT_70B["throughput"].append(df['Throughput (tokens/s)'].mean().item() * 60 / seqlen)

font = 20
dict_GPU_70B = load_QoS_file("data/GPU_70B_4k.csv")
plt.figure(figsize=(10, 8))
    
plt.plot(dict_CENT_70B["throughput"], dict_CENT_70B["latency"], marker='s', linestyle='-', color='Red', label="CENT")
plt.plot(dict_GPU_70B["throughput"], dict_GPU_70B["latency"], marker='o', linestyle='-', color='Blue', label="GPU")

plt.legend(loc="upper left", fontsize=font)
plt.tick_params(axis='x', labelsize=font)
plt.tick_params(axis='y', labelsize=font)

plt.xlabel('Throughput (Query/min)', fontsize=font)
plt.ylabel('Query Latency (min)', fontsize=font)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_14b.pdf')

# print(dict_CENT_70B["latency"])
# print(dict_CENT_70B["throughput"])
# print(dict_GPU_70B["latency"])
# print(dict_GPU_70B["throughput"])

df = pd.DataFrame(columns=['CENT Throughput (Query/min)', 'CENT Latency (min)', 'GPU Throughput (Query/min)', 'GPU Latency (min)'])

for i in range(len(dict_GPU_70B['latency'])):
    if i < len(dict_CENT_70B['latency']):
        new_row = {
            'CENT Throughput (Query/min)': dict_CENT_70B['throughput'][i],
            'CENT Latency (min)': dict_CENT_70B['latency'][i],
            'GPU Throughput (Query/min)': dict_GPU_70B['throughput'][i],
            'GPU Latency (min)': dict_GPU_70B['latency'][i]
        }
    else:
        new_row = {
            'GPU Throughput (Query/min)': dict_GPU_70B['throughput'][i],
            'GPU Latency (min)': dict_GPU_70B['latency'][i]
        }
    df_new = pd.DataFrame(new_row, index=[0])    
    df = pd.concat([df, df_new], ignore_index=True)

if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df.to_csv('figure_source_data/figure_14b.csv', index=False)