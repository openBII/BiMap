import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_simulation_results = pd.read_csv('cent_simulation/simulation_results.csv')
prefill_size = 512
decoding_list = [128, 512, 1024, 3584]
GPU_decoding_latency_list =  []
CENT_decoding_latency_list = []
transformer_blocks = 80

df_GPU_latency = pd.read_csv('data/GPU_70B_latency.csv')
GPU_prefill_latency = df_GPU_latency[(df_GPU_latency['Phase'] == 'Prefill')]['Latency (min)'].iloc[0]
for d in decoding_list:
    GPU_decoding_latency = df_GPU_latency[(df_GPU_latency['Phase'] == 'Decoding_'+str(d))]['Latency (min)'].iloc[0]
    GPU_decoding_latency_list.append(GPU_decoding_latency)

df_simulation_results = df_simulation_results[(df_simulation_results['Model'] == "Llama2-70B") & (df_simulation_results['Device number'] == 32) & (df_simulation_results['Pipeline parallelism'] == transformer_blocks) & (df_simulation_results['Tensor parallelism'] == 1)]

df = df_simulation_results[(df_simulation_results['Sequence length'] <= prefill_size)]
CENT_prefill_latency = df['Token latency (ms)'].mean() * prefill_size / 1000 / 60

for d in decoding_list:
    df = df_simulation_results[(df_simulation_results['Sequence length'] > prefill_size) & (df_simulation_results['Sequence length'] <= d + prefill_size)]
    CENT_decoding_latency_list.append(df['Token latency (ms)'].mean() * d / 1000 / 60)

df_results = pd.DataFrame(columns=['Prefill Latency (min)', 'Decoding Latency (min)'])
for d in decoding_list:
    new_row = {
        'Prefill Latency (min)': GPU_prefill_latency,
        'Decoding Latency (min)': GPU_decoding_latency_list[decoding_list.index(d)]
    }
    df_new = pd.DataFrame(new_row, index=[0])
    df_results = pd.concat([df_results, df_new], ignore_index=True)
    new_row = {
        'Prefill Latency (min)': CENT_prefill_latency,
        'Decoding Latency (min)': CENT_decoding_latency_list[decoding_list.index(d)]
    }
    df_new = pd.DataFrame(new_row, index=[0])    
    df_results = pd.concat([df_results, df_new], ignore_index=True)
if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df_results.to_csv('figure_source_data/figure_14d.csv', index=False)

# Labels for x-axis
x_labels = [
    "GPU\nIn 512\nOut 128", "CENT",
    "GPU\nIn 512\nOut 512", "CENT",
    "GPU\nIn 512\nOut 1k", "CENT",
    "GPU\nIn 512\nOut 3.5k", "CENT"
]

# Data (example values, adjust accordingly)
prefill_data = []
decoding_data = []
for d in decoding_list:
    prefill_data.append(GPU_prefill_latency)
    prefill_data.append(CENT_prefill_latency)
    decoding_data.append(GPU_decoding_latency_list[decoding_list.index(d)])
    decoding_data.append(CENT_decoding_latency_list[decoding_list.index(d)])

prefill = np.array(prefill_data)
decoding = np.array(decoding_data)

x = np.arange(len(x_labels))  # X positions

# Plot
fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(x, prefill, color="lightskyblue", edgecolor='black', label="Prefill")
ax.bar(x, decoding, bottom=prefill, color="sandybrown", edgecolor='black', label="Decoding")

# Labels & Formatting
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10, rotation=0)
ax.set_ylabel("Query Latency (minute)", fontsize=12)
ax.set_ylim(0, 9)  # Adjust based on data range
ax.legend(fontsize=10, loc="upper left")

# Grid & Styling
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show plot
plt.tight_layout()
if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_14d.pdf')