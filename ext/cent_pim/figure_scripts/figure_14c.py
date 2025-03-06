import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cent_simulation.utils import InOut_latency

df_simulation_results = pd.read_csv('cent_simulation/simulation_results.csv')

batch = [80, 16, 8, 4, 2, 1]
seqlen = 4096
num_devices = 32
transformer_blocks = 80

parallel_list = ["PP=80", "PP=16 TP=2", "PP=8 TP=4", "PP=4 TP=8", "PP=2 TP=16", "PP=1 TP=32"]
PIM_latency_list = []
CXL_latency_list = []
Acc_latency_list = []
CPU_latency_list = []

df = df_simulation_results[(df_simulation_results['Model'] == 'Llama2-70B') & (df_simulation_results['Device number'] == 32) & (df_simulation_results['Pipeline parallelism'] == 80) & (df_simulation_results['Tensor parallelism'] == 1)]

PIM_latency = df['PIM latency'].mean() * transformer_blocks / 1000 / 60 * seqlen
CXL_latency = df['CXL latency'].mean() * transformer_blocks / 1000 / 60 * seqlen
Acc_latency = df['Acc latency'].mean() * transformer_blocks / 1000 / 60 * seqlen
CPU_latency = InOut_latency * seqlen / 1000 / 60

PIM_latency_list.append(PIM_latency)
CXL_latency_list.append(CXL_latency)
Acc_latency_list.append(Acc_latency)
CPU_latency_list.append(CPU_latency)

for pp in batch[1:]:

    tp = num_devices // pp
    df = df_simulation_results[(df_simulation_results['Model'] == 'Llama2-70B') & (df_simulation_results['Device number'] == 32) & (df_simulation_results['Pipeline parallelism'] == pp) & (df_simulation_results['Tensor parallelism'] == tp)]

    PIM_latency = df['PIM latency'].mean() * transformer_blocks / 1000 / 60 * seqlen
    CXL_latency = df['CXL latency'].mean() * transformer_blocks / 1000 / 60 * seqlen
    Acc_latency = df['Acc latency'].mean() * transformer_blocks / 1000 / 60 * seqlen
    CPU_latency = InOut_latency * seqlen / 1000 / 60
    
    PIM_latency_list.append(PIM_latency)
    CXL_latency_list.append(CXL_latency)
    Acc_latency_list.append(Acc_latency)
    CPU_latency_list.append(CPU_latency)

df_results = pd.DataFrame(columns=['PIM Latency (min)', 'CXL Latency (min)', 'Acc Latency (min)', 'CPU Latency (min)'])

for i in range(len(parallel_list)):
    new_row = {
        'PIM Latency (min)': PIM_latency_list[i],
        'CXL Latency (min)': CXL_latency_list[i],
        'Acc Latency (min)': Acc_latency_list[i],
        'CPU Latency (min)': CPU_latency_list[i]
    }
    df_new = pd.DataFrame(new_row, index=[0])    
    df_results = pd.concat([df_results, df_new], ignore_index=True)

if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df_results.to_csv('figure_source_data/figure_14c.csv', index=False)

# Stacked bar chart
fig, ax = plt.subplots(figsize=(6, 4))
y_pos = np.arange(len(parallel_list))

ax.barh(y_pos, PIM_latency_list, color="navajowhite", edgecolor='black', label="PIM")
ax.barh(y_pos, CXL_latency_list, left=PIM_latency_list, color="lightblue", edgecolor='black', label="CXL")
ax.barh(y_pos, Acc_latency_list, left=np.array(PIM_latency_list) + np.array(CXL_latency_list), color="darkgreen", edgecolor='black', label="PNM")
ax.barh(y_pos, CPU_latency, left=np.array(PIM_latency_list) + np.array(CXL_latency_list) + np.array(Acc_latency_list), color="black", edgecolor='black', label="Host CPU")

# Labels & Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(parallel_list, fontsize=12)
ax.set_xlabel("Query Latency (minute)", fontsize=12)
ax.legend(fontsize=10, loc="upper right")

plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_14c.pdf')