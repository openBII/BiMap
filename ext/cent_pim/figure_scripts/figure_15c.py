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


CENT_energy_list = []
GPU_energy_list = []
speedup_list = []
df_energy_speedup = pd.DataFrame(columns=['Model', 'CENT/GPU Normalized Token / J'])
df_GPU_energy = pd.read_csv('data/GPU_energy.csv')
for phase in phases:
    for model in models:
        GPU_energy_list.append(df_GPU_energy[(df_GPU_energy['Model'] == model)][phase].iloc[0])
        df = df_processed_results[(df_processed_results['Model'] == model) & (df_processed_results['Seqlen'] == seqlen) & (df_processed_results['Pipeline parallelism'] == transformer_block[model]) & (df_processed_results['Tensor parallelism'] == 1) & (df_processed_results['Phase'] == phase)]
        CENT_energy_list.append(1000 / df['Energy per Token (mJ)'].iloc[0])
        speedup_list.append(CENT_energy_list[-1] / GPU_energy_list[-1])
        new_df = pd.DataFrame({'Model': [model], 'CENT/GPU Normalized Token / J': [speedup_list[-1]]})
        df_energy_speedup = pd.concat([df_energy_speedup, new_df], ignore_index=True)
geomean = gmean(speedup_list[-3:])
new_df = pd.DataFrame({'Model': ['Geomean'], 'CENT/GPU Normalized Token / J': [geomean]})
df_energy_speedup = pd.concat([df_energy_speedup, new_df], ignore_index=True)
if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df_energy_speedup.to_csv('figure_source_data/figure_15c.csv', index=False)


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
plt.bar(x_labels, speedup_list, color='green', edgecolor='black')

# Labels
plt.ylabel("CENT/GPU Normalized Token / J", fontsize=12)

# Formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_15c.pdf')

