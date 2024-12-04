import numpy as np
from matplotlib import pyplot as plt
from src.simulator.resource_simulator.evaluation_model.chiplet.exploration import single_module_multiple_chiplets

latencies = np.load('test/exp/multi_package_data.npy')

numbers = (1, 2, 4, 8, 16)
cost_decreasing_mcms, cost_decreasing_sis, cost_increasing_mcms, cost_increasing_sis = single_module_multiple_chiplets(numbers=numbers)
number_labels = [str(num) for num in numbers]

# plt.plot(number_labels, cost_decreasing_mcms, marker='o')
# plt.plot(number_labels, cost_decreasing_sis, marker='o')
plt.plot(number_labels, latencies, linestyle='--', marker='o', label="latency")
# plt.plot(numbers, 400 * np.array(cost_increasing_mcms[:1] + cost_increasing_mcms[2:]), marker='o')
plt.plot(number_labels, 400 * np.array(cost_increasing_sis[:1] + cost_increasing_sis[2:]), linestyle='--', marker='o', label="cost")
# plt.plot(numbers, 40000000000000 / np.array(cost_increasing_mcms[:1] + cost_increasing_mcms[2:]) / latencies, marker='o')
plt.plot(number_labels, 10000000000000 / np.array(cost_increasing_sis[:1] + cost_increasing_sis[2:]) / latencies, marker='o', label="performance-price ratio")
plt.title("Multi-Package Multi-Chiplet Architecture Performance-Cost Ratio")
plt.xlabel("Num of Chiplets in One Package")
plt.legend()
plt.savefig('temp/multi_package.png', dpi=300)

