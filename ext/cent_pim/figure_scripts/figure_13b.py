import os
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt

df_processed_results = pd.read_csv('cent_simulation/processed_results.csv')

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
phases = ['prefill', 'decoding', 'end2end']
transformer_block = {
    'Llama2-7B': 32,
    'Llama2-13B': 40,
    'Llama2-70B': 80,
}
seqlen = 4096


CENT_throughput_list = []
GPU_throughput_list = []
speedup_list = []
df_throughput_speedup = pd.DataFrame(columns=['Model', 'CENT/GPU Normalized Throughput (Tokens/s)'])
df_GPU_throughput = pd.read_csv('data/GPU_throughput.csv')
for phase in phases:
    for model in models:
        GPU_throughput_list.append(df_GPU_throughput[(df_GPU_throughput['Model'] == model)][phase].iloc[0])
        df = df_processed_results[(df_processed_results['Model'] == model) & (df_processed_results['Seqlen'] == seqlen) & (df_processed_results['Pipeline parallelism'] == transformer_block[model]) & (df_processed_results['Tensor parallelism'] == 1) & (df_processed_results['Phase'] == phase)]
        CENT_throughput_list.append(df['Throughput (tokens/s)'].iloc[0])
        speedup_list.append(df['Throughput (tokens/s)'].iloc[0] / GPU_throughput_list[-1])
        new_df = pd.DataFrame({'Model': [model], 'CENT/GPU Normalized Throughput (Tokens/s)': [speedup_list[-1]]})
        df_throughput_speedup = pd.concat([df_throughput_speedup, new_df], ignore_index=True)
geomean = gmean(speedup_list[-3:])
new_df = pd.DataFrame({'Model': ['Geomean'], 'CENT/GPU Normalized Throughput (Tokens/s)': [geomean]})
df_throughput_speedup = pd.concat([df_throughput_speedup, new_df], ignore_index=True)
if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df_throughput_speedup.to_csv('figure_source_data/figure_13b.csv', index=False)


x_labels = models * 3 + ['Geomean']
x_labels[0] = '7B\nPrefill'
x_labels[1] = '13B\nPrefill'
x_labels[2] = '70B\nPrefill'
x_labels[3] = '7B\nDecoding'
x_labels[4] = '13B\nDecoding'
x_labels[5] = '70B\nDecoding'
x_labels[6] = '7B\nEnd-to-End'
x_labels[7] = '13B\nEnd-to-End'
x_labels[8] = '70B\nEnd-to-End'
speedup_list.append(geomean)
speedup_list = [float(i) for i in speedup_list]

# Plot
plt.figure(figsize=(13, 5))
plt.bar(x_labels, speedup_list, color='pink', edgecolor='black')

# Labels
plt.ylabel("CENT/GPU Throughput Speedup", fontsize=12)

# Formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_13b.pdf')

