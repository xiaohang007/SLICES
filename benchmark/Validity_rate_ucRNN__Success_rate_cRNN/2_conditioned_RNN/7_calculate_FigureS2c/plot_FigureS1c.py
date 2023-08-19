import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Read CSV file; the first row is the column header
data = pd.read_csv('energy_formation_m3gnet_lists.csv')

# Get the number of columns in the data
num_cols = len(data.columns)

# Create a figure and subplots
fig, axs = plt.subplots(num_cols, 1, figsize=(6, num_cols*1.4), sharex=True, gridspec_kw={'hspace': 0})

# Create an array for bin boundaries
bins = np.linspace(-6, 0, 50)

# Loop through each column of data
for i, col_name in enumerate(data.columns):
    # If it is the histogram for the last two rows, set the color to violet
    if i >= num_cols - 2:
        color = 'violet'
    else:
        color = 'lightblue'
        
    # Plot the histogram for the column data
    axs[i].hist(data[col_name].dropna(), bins=bins, density=True, color=color, edgecolor='black', alpha=0.7)
    # Fit a normal distribution to the column data
    mu, std = norm.fit(data[col_name].dropna())
    # Generate a range of x values
    x = np.linspace(-6, 0, 100)
    # Generate the probability density function for the normal distribution
    p = norm.pdf(x, mu, std)
    # Plot the PDF
    axs[i].plot(x, p, 'k', linewidth=2)
    # Place a text box in the subplot
    axs[i].text(0.37, 0.95, col_name, transform=axs[i].transAxes, fontsize=10, fontweight='bold', va='top')

# Ensure all subplots have the same range on the Y-axis
y_max = max(ax.get_ylim()[1] for ax in axs)
for ax in axs:
    ax.set_ylim(0, y_max)

# Set titles for the X and Y axis
#plt.xlabel('XXX')
#plt.ylabel('YYY')

# Set the x-axis range for all subplots
plt.xlim(-6, 0)

# Display the figure
plt.show()

# Save the figure
plt.savefig("test3.svg", dpi=600)
