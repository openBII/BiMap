import numpy as np
from matplotlib import pyplot as plt
from src.simulator.resource_simulator.evaluation_model.chiplet.exploration import single_module_multiple_chiplets

mcm_latencies = np.load('test/exp/multi_package_data_1mb_mcm.npy')
si_latencies = np.load('test/exp/multi_package_data_1mb_cowos.npy')

numbers = (1, 2, 4, 8, 16)
cost_decreasing_mcms, cost_decreasing_sis, cost_increasing_mcms, cost_increasing_sis = single_module_multiple_chiplets(numbers=numbers)
number_labels = [str(num) for num in numbers]

numbers = (1, 2, 3, 4, 6, 8, 12)
cost_decreasing_mcms, cost_decreasing_sis, cost_increasing_mcms, cost_increasing_sis = single_module_multiple_chiplets(numbers=numbers,
                                                                                                                       module_area=400)

# plt.plot(number_labels, cost_decreasing_mcms, marker='o')
# plt.plot(number_labels, cost_decreasing_sis, marker='o')
plt.plot(number_labels, mcm_latencies, linestyle='--', marker='o', label="Latency")
# plt.plot(numbers, 400 * np.array(cost_increasing_mcms[:1] + cost_increasing_mcms[2:]), marker='o')
plt.bar(number_labels, 400 * np.array(cost_increasing_mcms[:1] + cost_increasing_mcms[2:]), width=0.5, label="Cost")
# plt.plot(numbers, 40000000000000 / np.array(cost_increasing_mcms[:1] + cost_increasing_mcms[2:]) / latencies, marker='o')
plt.plot(number_labels, 5000000000000 / np.array(cost_increasing_mcms[:1] + cost_increasing_mcms[2:]) / mcm_latencies, marker='o', label="Performance-Cost Ratio")
plt.title("Multi-Package Multi-Chiplet Architecture")
plt.xlabel("Num of Chiplets in One Package")
plt.legend()
plt.savefig('temp/multi_package_mcm.png', dpi=300)

plt.clf()
plt.plot(number_labels, si_latencies, linestyle='--', marker='o', label="Latency")
# plt.plot(numbers, 400 * np.array(cost_increasing_mcms[:1] + cost_increasing_mcms[2:]), marker='o')
plt.bar(number_labels, 400 * np.array(cost_increasing_sis[:1] + cost_increasing_sis[2:]), width=0.5, label="Cost")
# plt.plot(numbers, 40000000000000 / np.array(cost_increasing_mcms[:1] + cost_increasing_mcms[2:]) / latencies, marker='o')
plt.plot(number_labels, 5000000000000 / np.array(cost_increasing_sis[:1] + cost_increasing_sis[2:]) / si_latencies, marker='o', label="Performance-Cost Ratio")
plt.title("Multi-Package Multi-Chiplet Architecture")
plt.xlabel("Num of Chiplets in One Package")
plt.legend()
plt.savefig('temp/multi_package_si.png', dpi=300)

