import exploration as ex
import matplotlib.pyplot as plt

# y1 = ex.yield_area()
# y2 = ex.single_chiplet_multiple_systems(5000)

# print(y1)
# print(y2)

numbers = (1, 2, 4, 8, 16)
cost_decreasing_mcms, cost_decreasing_sis, cost_increasing_mcms, cost_increasing_sis = ex.single_module_multiple_chiplets(numbers=numbers)
number_labels = ['SoC'] + [str(num) for num in numbers]

plt.plot(number_labels, cost_decreasing_mcms, marker='o')
plt.plot(number_labels, cost_decreasing_sis, marker='o')
plt.plot(number_labels, cost_increasing_mcms, marker='o')
plt.plot(number_labels, cost_increasing_sis, marker='o')
plt.title("Multi-Chip Cost")
plt.xlabel("Num of Chiplets")
plt.ylabel("Cost")
plt.savefig('temp/chiplet_cost.png', dpi=300)

