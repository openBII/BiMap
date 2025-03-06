import os
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt

df_processed_results = pd.read_csv('cent_simulation/processed_results.csv')

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
phases = ['prefill', 'decoding', 'end2end']
tokens = {
    'prefill': 512,
    'decoding': 3584,
    'end2end': 4096,
}
transformer_block = {
    'Llama2-7B': 32,
    'Llama2-13B': 40,
    'Llama2-70B': 80,
}
seqlen = 4096


CENT_power_list = []
GPU_power_list = []
df_power = pd.DataFrame(columns=['Model', 'Power(W)'])
df_GPU_power = pd.read_csv('data/GPU_power.csv')
for model in models:
    for phase in phases:
        GPU_power_list.append(df_GPU_power[(df_GPU_power['Model'] == model)][phase].iloc[0])
        df = df_processed_results[(df_processed_results['Model'] == model) & (df_processed_results['Seqlen'] == seqlen) & (df_processed_results['Pipeline parallelism'] == transformer_block[model]) & (df_processed_results['Tensor parallelism'] == 1) & (df_processed_results['Phase'] == phase)]
        CENT_power_list.append(df['Total power (W)'].iloc[0])
        new_df = pd.DataFrame({'Model': [model], 'Power(W)': [CENT_power_list[-1]]})
        df_power = pd.concat([df_power, new_df], ignore_index=True)
    for phase in phases:
        new_df = pd.DataFrame({'Model': [model], 'Power(W)': df_GPU_power[(df_GPU_power['Model'] == model)][phase].iloc[0]})
        df_power = pd.concat([df_power, new_df], ignore_index=True)
if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df_power.to_csv('figure_source_data/figure_15a.csv', index=False)


x_labels = []
for model in models:
    x_labels.append('8 CENT\n' + model.split("-")[-1] + '\nPrefill')
    x_labels.append('1 A100\n' + model.split("-")[-1] + '\nPrefill')
    x_labels.append('20 CENT\n' + model.split("-")[-1] + '\nDecoding')
    x_labels.append('2 A100\n' + model.split("-")[-1] + '\nDecoding')
    x_labels.append('32 CENT\n' + model.split("-")[-1] + '\nEnd2End')
    x_labels.append('4 A100\n' + model.split("-")[-1] + '\nEnd2End')

# Plot
plt.figure(figsize=(20, 8))
power_list = df_power['Power(W)'].tolist()
plt.bar(x_labels, power_list, color='orange', edgecolor='black')

# Labels
plt.ylabel("Power(W)", fontsize=12)

# Formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_15a.pdf')

