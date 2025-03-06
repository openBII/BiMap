import os
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt

df_processed_results = pd.read_csv('cent_simulation/processed_results.csv')

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
devices = {
    'Llama2-7B': 8,
    'Llama2-13B': 20,
    'Llama2-70B': 32,
}
seqlen = 4096


CENT_latency_list = []
GPU_latency_list = []
speedup_list = []
df_latency_speedup = pd.DataFrame(columns=['Model', 'GPU/CENT Normalized Latency'])
df_GPU_latency = pd.read_csv('data/GPU_latency.csv')
for model in models:
    GPU_latency_list.append(df_GPU_latency[(df_GPU_latency['Model'] == model)]['End-to-end Latency (s)'].iloc[0])
    df = df_processed_results[(df_processed_results['Model'] == model) & (df_processed_results['Seqlen'] == seqlen) & (df_processed_results['Pipeline parallelism'] == 1) & (df_processed_results['Tensor parallelism'] == devices[model]) & (df_processed_results['Phase'] == 'end2end')]
    CENT_latency_list.append(df['Total Latency (s)'].iloc[0])
    speedup_list.append(GPU_latency_list[-1] / (df['Total Latency (s)'].iloc[0]))
    new_df = pd.DataFrame({'Model': [model], 'GPU/CENT Normalized Latency': [speedup_list[-1]]})
    df_latency_speedup = pd.concat([df_latency_speedup, new_df], ignore_index=True)
if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df_latency_speedup.to_csv('figure_source_data/figure_13a.csv', index=False)


x_labels = models + ['Geomean']
speedup_list.append(gmean(speedup_list))
# Plot
plt.figure(figsize=(6, 5))
plt.bar(x_labels, speedup_list, color='skyblue', edgecolor='black')

# Labels
plt.xlabel("End-to-End", fontsize=12)
plt.ylabel("CENT/GPU Latency Speedup", fontsize=12)

# Formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_13a.pdf')



