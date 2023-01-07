# import libraries
import numpy as np
import matplotlib.pyplot as plt

# Creating dataset
x = np.arange(1.0, 100.0, 0.191)
dataset_1 = np.exp(x**0.25) - np.exp(x**0.5)
dataset_2 = np.sin(0.4 * np.pi * x**0.5) + np.cos(0.8 * np.pi * x**0.25)

# Creating plot with dataset_1
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y1-axis', color = color)
ax1.plot(x, dataset_1, color = color)
ax1.tick_params(axis ='y', labelcolor = color)

# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()

color = 'tab:green'
ax2.set_ylabel('Y2-axis', color = color)
ax2.plot(x, dataset_2, color = color)
ax2.tick_params(axis ='y', labelcolor = color)

# Adding title
plt.title('Use different y-axes on the left and right of a Matplotlib plot', fontweight ="bold")

# Show plot
plt.show()
