import os
import glob
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import matplotlib.ticker as ticker
import pickle,json
from pymatgen.core.structure import Structure

def load_and_save_structure_database():
    # json file (including "cif") of training dataset for novelty check
    with open('../../mp20_nonmetal/cifs_filtered.json', 'r') as f:
        cifs = json.load(f)
    
    structure_database = []
    for i in range(len(cifs)):
        cif_string = cifs[i]["cif"]
        try:
            stru = Structure.from_str(cif_string, "cif")
            structure_database.append([stru, cifs[i]["band_gap"]])
        except Exception as e:
            print(e)
    
    # Serialize the data
    with open('structure_database.pkl', 'wb') as f:
        pickle.dump(structure_database, f)

def process_data():
    load_and_save_structure_database()
    splitRun_csv_local(filename='../1_train_generate/inverse_designed_SLICES.csv', threads=8, skip_header=True)
    show_progress_local()
    collect_csv(output="results.csv",
                glob_target="./job_*/result.csv", cleanup=True,
                header="eform_target,SLICES,eform_chgnet,poscar,novelty\n")


def prepare_data(results_file, training_file):
    
    results_1 = pd.read_csv(results_file)
    trainingset = pd.read_csv(training_file, header=0)
    
    header_values = results_1.iloc[:, 0].tolist()
    data_values = results_1.iloc[:, 2].tolist()
    trainingset_values = trainingset.iloc[:, 1].tolist()
    novelty_values = results_1.iloc[:, 4].tolist()
    
    data_dict = {}
    for header, value, novelty in zip(header_values, data_values, novelty_values):
        if header not in data_dict:
            data_dict[header] = {'all': [], 'novel': []}
        data_dict[header]['all'].append(value)
        if novelty == 1:
            data_dict[header]['novel'].append(value)
    
    sorted_keys = sorted(data_dict.keys(), reverse=True)
    
    return data_dict, sorted_keys, trainingset_values

def create_dataframe(data_dict, sorted_keys, trainingset_values, type='all'):
    df = pd.DataFrame({k: pd.Series(data_dict[k][type], index=range(len(data_dict[k][type]))) for k in sorted_keys})
    df = pd.concat([df, pd.Series(trainingset_values, name='training_dataset')], axis=1)
    return df

def plot_histograms(data, output_file):
    num_cols = len(data.columns)
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(num_cols, 1, figsize=(num_cols*0.988*cm, 11.31*cm), sharex=True, gridspec_kw={'hspace': 0})
    
    bins = np.linspace(-6, 0, 50)
    
    for i, col_name in enumerate(data.columns):
        color = 'violet' if col_name == 'training_dataset' else 'lightblue'
        
        axs[i].hist(data[col_name].dropna(), bins=bins, density=True, color=color, edgecolor='black', alpha=0.7)
        
        mu, std = norm.fit(data[col_name].dropna())
        x = np.linspace(-6, 0, 100)
        p = norm.pdf(x, mu, std)
        
        axs[i].text(0.37, 0.95, col_name, transform=axs[i].transAxes, fontsize=6, va='top')
        
        mean_val = data[col_name].mean()
        axs[i].axvline(mean_val, color='red', linestyle='--', linewidth=1)
        axs[i].text(mean_val, axs[i].get_ylim()[1]*0.9, f"{mean_val:.2f}", color='red', fontsize=6, ha='left')
    
    y_max = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(0, y_max)
    
    plt.xlim(-6, 0)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    axs[-1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.savefig(output_file, dpi=600)
    plt.close()

if __name__ == "__main__":
    os.system("rm -rf job_*")
    process_data()
    
    data_dict, sorted_keys, trainingset_values = prepare_data("results.csv", "../../mp20_nonmetal/train_data_reduce_zero.csv")
    
    # Process all materials
    all_materials_df = create_dataframe(data_dict, sorted_keys, trainingset_values, 'all')
    all_materials_df.to_csv('energy_formation_chgnet_lists.csv', index=False)
    plot_histograms(all_materials_df, "./results_all.jpg")
    
    # Process novel materials
    novel_materials_df = create_dataframe(data_dict, sorted_keys, trainingset_values, 'novel')
    novel_materials_df.to_csv('energy_formation_chgnet_lists_novel.csv', index=False)
    plot_histograms(novel_materials_df, "./results_novel.jpg")
    os.system("rm energy_formation_chgnet_lists.csv energy_formation_chgnet_lists_novel.csv")
    
    print("程序执行完毕，结果已保存到energy_formation_chgnet_lists.csv和energy_formation_chgnet_lists_novel.csv")
    print("图表已保存为results_all.jpg和results_novel.jpg")
