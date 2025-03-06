import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_simulation_results = pd.read_csv('cent_simulation/simulation_results.csv')
gpu_decoding = pd.read_csv('data/GPU_70B_decoding.csv')

decoding = 3584
seqlen_list = [4096, 8192, 16384, 32768]
for i in range(len(seqlen_list)):
    seqlen_list[i] = (seqlen_list[i] + seqlen_list[i] - 3584) // 2
cent_throughput_list = []
gpu_throughput_list = [t for t in gpu_decoding['Throughput (tokens/s)']]
speedup_list = []

for seqlen in seqlen_list:
    df = df_simulation_results[(df_simulation_results['Model'] == 'Llama2-70B') & (df_simulation_results['Device number'] == 32) & (df_simulation_results['Sequence length'] == seqlen) & (df_simulation_results['Pipeline parallelism'] == 80) & (df_simulation_results['Tensor parallelism'] == 1)]
    cent_throughput_list.append(df['Throughput (tokens/s)'].mean().item())
    speedup_list.append(cent_throughput_list[-1] / gpu_throughput_list[seqlen_list.index(seqlen)])

# print(cent_throughput_list)
# print(gpu_throughput_list)

context_lengths = ["4K", "8K", "16K", "32K"]
# Plot
plt.figure(figsize=(5, 5))
plt.bar(context_lengths, speedup_list, color='skyblue', edgecolor='black')

# Labels
plt.xlabel("Context Length", fontsize=12)
plt.ylabel("CENT / GPU Speedup", fontsize=12)
plt.ylim(0, 4)

# Formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_14a.pdf')

df = pd.DataFrame(columns=['CENT/GPU Normalized Throughput'])

for i in range(len(seqlen_list)):
    new_row = {
        'CENT/GPU Normalized Throughput': speedup_list[i],
    }
    df_new = pd.DataFrame(new_row, index=[0])    
    df = pd.concat([df, df_new], ignore_index=True)

if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df.to_csv('figure_source_data/figure_14a.csv', index=False)