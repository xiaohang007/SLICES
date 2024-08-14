import os
import glob
from utils import splitRun_csv, show_progress, collect_csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def process_data():
    os.system("rm ./candidates/*")
    splitRun_csv(filename='../2_inverse_novelty/results.csv', threads=30, skip_header=True)
    show_progress()
    collect_csv(output="results.csv",
                glob_target="./job_*/result.csv", cleanup=True,
                header="bandgap_target,eform_target,SLICES,poscar,novelty,dir_bandgap,indir_bandgap,eform\n")

def plot_2d_scatter():
    target_x = [1.0]
    target_y = [-2.0]

    sns.set(style="ticks")

    bandgap = []
    fenergy = []
    with open("results.csv") as f:
        next(f)  # Skip header
        for line in f:
            ll = line.strip().split(',')
            if len(ll) == 8 and ll[-1] : # and ll[4] == '1'
                bandgap.append(float(ll[6]))
                fenergy.append(float(ll[7]))

    bandgap = [x if x > 0.0 else 0.0 for x in bandgap]

    g = sns.jointplot(x=bandgap, y=fenergy, kind="kde", fill=True, cmap="Purples", height=7, levels=15)

    g.plot_joint(sns.scatterplot, color="red", s=15, edgecolor='black', linewidth=0.7, alpha=0.6)

    g.ax_joint.scatter(target_x, target_y, color="yellow", s=130, marker="*", edgecolor='black', linewidth=0.5, zorder=5)

    for x in target_x:
        g.ax_joint.axvline(x, color='blue', linestyle='--', linewidth=0.5, zorder=1)
    for y in target_y:
        g.ax_joint.axhline(y, color='blue', linestyle='--', linewidth=0.5, zorder=1)

    sns.kdeplot(bandgap, ax=g.ax_marg_x, color='purple', fill=True, alpha=0.1)
    sns.kdeplot(fenergy, ax=g.ax_marg_y, color='purple', fill=True, vertical=True, alpha=0.1)

    for x in target_x:
        g.ax_marg_x.axvline(x, color='blue', linestyle='--', linewidth=0.5, zorder=1)
    for y in target_y:
        g.ax_marg_y.axhline(y, color='blue', linestyle='--', linewidth=0.5, zorder=1)

    g.set_axis_labels('Bandgap (eV)', 'Formation Energy (eV/atom)', fontsize=16)
    g.ax_joint.set_xlabel('Bandgap (eV)', fontsize=16)
    g.ax_joint.set_ylabel('Formation Energy (eV/atom)', fontsize=16)
    g.ax_marg_x.set_xlabel('Bandgap (eV)', fontsize=16)
    g.ax_marg_y.set_ylabel('Formation Energy (eV/atom)', fontsize=16)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

    sns.despine()

    g.fig.set_size_inches(6, 6)

    plt.savefig("2d_scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    #process_data()
    plot_2d_scatter()
    print("Data processing completed. 2D scatter plot saved as 2d_scatter_plot.png")
