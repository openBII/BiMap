import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 定义颜色
color = {
    "Llama2-70B": "grey",
    "Llama2-13B": "#003f7c",
    "Llama2-7B": "#9400d3",
}

color1 = {
    "PIM": "grey",
    "NLM": "#003f7c"
}

hatch1 = {
    "PIM": "xxx",
    "NLM": "///"
}

# 读取CSV文件
data = pd.read_csv('figs/CompAir/pim_vs_nonlinear.report')

model = data['Model']
seq_len = data['Sequence length']
pim_latency = data['PIM latency']
cxl_latency = data['CXL latency']
acc_latency = data['Acc latency']
group = 3
data_per_group = len(model) // group

all_latency = pim_latency + cxl_latency + acc_latency
acc_cent = acc_latency / all_latency * 100

bar_width = 0.3
# plt.figure(figsize=(6, 3))
fig = plt.figure(figsize=(9, 2.5))

ax0 = plt.subplot2grid((1,5),(0,0),colspan=3,rowspan=1)


x = np.arange(data_per_group)
alpha = 0.5
ax0.grid(alpha=alpha)

ax0.plot(x - bar_width, acc_cent[data_per_group:2*data_per_group], color=color[model[data_per_group]], marker='*', ms=4, label='ACC-'+model[data_per_group])
ax0.plot(x , acc_cent[0:data_per_group], color=color[model[0]], marker='*', ms=4, label='ACC-'+model[0])
ax0.plot(x + bar_width, acc_cent[data_per_group*2:3*data_per_group], color=color[model[data_per_group*2]], marker='*', ms=4, label='ACC-'+model[data_per_group*2])

# ax0.bar(x - bar_width, acc_cent[data_per_group:2*data_per_group], width=bar_width, color=color[model[data_per_group]], edgecolor='white', label='ACC-'+model[data_per_group])
# ax0.bar(x, acc_cent[0:data_per_group], width=bar_width, color=color[model[0]], edgecolor='white', label='ACC-'+model[0])
# ax0.bar(x + bar_width, acc_cent[2*data_per_group:3*data_per_group], width=bar_width, color=color[model[data_per_group*2]], edgecolor='white', label='ACC-'+model[data_per_group*2])

ax0.set_xticks(x, seq_len[0:data_per_group], rotation=90, fontsize=10)
ax0.set_xlabel('Sequence Length', fontsize=15)
ax0.set_ylabel('Acc Proportion / %', fontsize=15)
ax0.legend(loc='upper left')

ax1 = plt.subplot2grid((1,5),(0,3),colspan=2,rowspan=1)
seq_len = [512, 1024, 2048, 4096, 8192, 9215]
pim_cycle = [123217, 133681, 156133, 200893, 290413, 312793]
no_merge_cycle = [114601, 120969, 135253, 153629, 220381, 234669]
nlmv_cycle = []
for i in range(len(seq_len)):
    nlmv_cycle.append(pim_cycle[i]-no_merge_cycle[i])
    
x = np.arange(len(seq_len))
alpha = 0.5
ax1.grid(alpha=alpha)
ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax1.bar(x - bar_width, pim_cycle, width=bar_width, color=color1["PIM"], hatch=hatch1['PIM'], edgecolor='white', label='Linear Overall')
ax1.bar(x, nlmv_cycle, width=bar_width, color=color1["NLM"], hatch=hatch1['NLM'], edgecolor='white', label='Non-Linear MV')
ax1.set_xticks(x - bar_width * 0.5, seq_len, rotation=90, fontsize=10)
ax1.set_xlabel('Sequence Length', fontsize=15)
ax1.set_ylabel('Cycles', fontsize=15)
ax1.legend()
plt.tight_layout()
plt.savefig('figs/CompAir/pim_vs_nonlinear.pdf')
