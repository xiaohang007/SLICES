import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import matplotlib.ticker as ticker
import os
pwd=os.getcwd()
# Read CSV file; the first row is the column header
data = pd.read_csv('energy_formation_m3gnet_lists.csv')
plt.rcParams['ytick.labelsize']=6

# Get the number of columns in the data
num_cols = len(data.columns)

# Create a figure and subplots
cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(num_cols, 1, figsize=(num_cols*0.988*cm, 11.31*cm) , sharex=True, gridspec_kw={'hspace': 0})

# Create an array for bin boundaries
bins = np.linspace(-6, 0, 50)

# Loop through each column of data
for i, col_name in enumerate(data.columns):
    # If it is the histogram for the last two rows, set the color to violet
    if i >= num_cols - 1:
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
    if i < num_cols - 1:
        axs[i].plot(x, p, 'k', linewidth=1)
    
    # Place a text box in the subplot
    axs[i].text(0.37, 0.95, col_name, transform=axs[i].transAxes, fontsize=6, va='top')
    
    # Calculate the mean of the data in the current column
    mean_val = data[col_name].mean()
    
    # Plot a vertical red line at the mean value
    axs[i].axvline(mean_val, color='red', linestyle='--', linewidth=1)
    
    # Print the mean value on the plot
    axs[i].text(mean_val, axs[i].get_ylim()[1]*0.9, f"{mean_val:.2f}", color='red', fontsize=6, ha='left')

# Ensure all subplots have the same range on the Y-axis
y_max = max(ax.get_ylim()[1] for ax in axs)
for ax in axs:
    ax.set_ylim(0, y_max)

# Set the x-axis range for all subplots
plt.xlim(-6, 0)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

# Display the figure
plt.show()

# Save the figure
plt.savefig("../results_"+pwd.split("/")[-1]+".jpg", dpi=600)
plt.savefig("./results_"+pwd.split("/")[-1]+".jpg", dpi=600)
