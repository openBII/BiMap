import matplotlib.pyplot as plt
import numpy as np

x = np.array([1.168 * 1e6, (31207.04979 + 471.870002) * 4])
y = np.array([31207.049794 * 4, 471.870002 * 4])

color = {
    "SRAM_PIM": "grey",
    "Router": "#003f7c",
    "Router_no_ALU": "#003f7c",
    "Curry_ALU": "#9400d3",
}

plt.subplot(121)
plt.pie(x,
        colors=[color["SRAM_PIM"], color["Router"]],
        explode=(0, 0.4),
        # autopct='%.2f%%',
       )

plt.subplot(122)
plt.pie(y,
        colors=[color["Router"], color["Curry_ALU"]],
        explode=(0, 0.4),
        # autopct='%.2f%%',
       )

print("Area", sum(x), "um^2")
plt.show()