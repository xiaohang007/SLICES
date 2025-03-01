import os
import glob
import argparse  # Import argparse for command-line argument parsing
from slices.utils import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import matplotlib.ticker as ticker
import pickle
import json
from pymatgen.core.structure import Structure


def load_and_save_structure_database(structure_json_path):
    """
    Loads CIF data from a JSON file, converts them to pymatgen Structures,
    and serializes the structure database using pickle.

    Parameters:
    - structure_json_path (str): Path to the JSON file containing CIFs.
    """
    try:
        with open(structure_json_path, 'r') as f:
            cifs = json.load(f)
    except FileNotFoundError:
        print(f"Error: The structure JSON file '{structure_json_path}' does not exist.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON file '{structure_json_path}': {e}")
        exit(1)

    structure_database = []
    for i, cif_entry in enumerate(cifs):
        cif_string = cif_entry.get("cif")
        if not cif_string:
            print(f"Warning: No 'cif' key found in entry {i}. Skipping.")
            continue
        try:
            stru = Structure.from_str(cif_string, "cif")
            structure_database.append([stru, cif_entry.get("band_gap", None)])
        except Exception as e:
            print(f"Error processing CIF at index {i}: {e}")

    # Serialize the data
    with open('structure_database.pkl', 'wb') as f:
        pickle.dump(structure_database, f)
    print(f"Structure database saved to 'structure_database.pkl'.")


def process_data(input_csv, output_csv, structure_json_path, threads):
    """
    Processes the input CSV file by splitting it into jobs, running them locally,
    and collecting the results.

    Parameters:
    - input_csv (str): Path to the input CSV file to be processed.
    - structure_json_path (str): Path to the JSON file containing CIFs.
    - threads (int): Number of threads to use for processing.
    """
    # 1) Clean up old job directories
    print("Cleaning up old job directories...")
    os.system("rm -rf job_*")

    # 2) Build structure_database.pkl
    print(f"Loading and saving structure database from '{structure_json_path}'...")
    load_and_save_structure_database(structure_json_path)

    # 3) Split the input CSV into job files
    print("Splitting the input CSV into job files...")
    splitRun_csv(filename=input_csv, threads=threads, skip_header=True)

    # 4) Show progress of local jobs
    print("Showing progress of local jobs...")
    show_progress()

    # 5) Collect all job results into a single results.csv
    print("Collecting results into 'results.csv'...")
    collect_csv(
        output=output_csv,
        glob_target="./job_*/result.csv",  # All partial results from job_* folders
        cleanup=True,
        header="eform_target,SLICES,eform_chgnet,poscar,novelty,spacegroup\n"
    )
    print("Results collected into 'results.csv'.")


def prepare_data(results_file, training_file):
    """
    Prepares the data by reading results and training files, organizing the data
    into dictionaries, and sorting the keys.

    Parameters:
    - results_file (str): Path to the results CSV file.
    - training_file (str): Path to the training CSV file.

    Returns:
    - data_dict (dict): Dictionary containing all and novel values.
    - sorted_keys (list): Sorted list of headers.
    - trainingset_values (list): List of training dataset values.
    """
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


def create_dataframe(data_dict, sorted_keys, trainingset_values, data_type='all'):
    """
    Creates a pandas DataFrame from the data dictionary.

    Parameters:
    - data_dict (dict): Dictionary containing data.
    - sorted_keys (list): Sorted list of headers.
    - trainingset_values (list): List of training dataset values.
    - data_type (str): Type of data to include ('all' or 'novel').

    Returns:
    - df (pd.DataFrame): Constructed DataFrame.
    """
    df = pd.DataFrame({k: pd.Series(data_dict[k][data_type], index=range(len(data_dict[k][data_type]))) for k in sorted_keys})
    df = pd.concat([df, pd.Series(trainingset_values, name='training_dataset')], axis=1)
    return df


def plot_combined_histograms(all_data, novel_data, output_file):
    num_cols = len(all_data.columns)
    fig, axs = plt.subplots(num_cols, 2, figsize=(12, 3*num_cols), sharex=True)
    bins = np.linspace(-6, 0, 50)
    # Ensure axs is always 2D
    if num_cols == 1:
        axs = axs.reshape(1, -1)
    for i, col_name in enumerate(all_data.columns):
        for j, (data, title) in enumerate(zip([all_data, novel_data], ['All Materials', 'Novel Materials'])):
            color = 'violet' if col_name == 'training_dataset' else 'lightblue'
            axs[i, j].hist(data[col_name].dropna(), bins=bins, density=True, color=color, edgecolor='black', alpha=0.7)
            mu, std = norm.fit(data[col_name].dropna())
            axs[i, j].text(0.05, 0.95, f"{col_name}\n{title}", transform=axs[i, j].transAxes, fontsize=8, va='top')
            mean_val = data[col_name].mean()
            axs[i, j].axvline(mean_val, color='red', linestyle='--', linewidth=1)
            axs[i, j].text(mean_val, axs[i, j].get_ylim()[1]*0.9, f"{mean_val:.2f}", color='red', fontsize=6, ha='left')
    for ax in axs.flat:
        ax.set_xlim(-6, 0)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.text(0.5, 0.02, 'Formation Energy (eV/atom)', ha='center', va='center', fontsize=10)
    fig.text(0.02, 0.5, 'Density', ha='center', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0, wspace=0.1)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig


def main():
    """
    Main function to parse command-line arguments and initiate data processing.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process and analyze structure data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to the input CSV file to be processed."
    )
    parser.add_argument(
        "--structure_json_for_novelty_check",
        type=str,
        help="Path to the JSON file containing CIFs (structure database)."
    )
    parser.add_argument(
        "--training_file",
        type=str,
        help="Path to the training CSV file (e.g., 'train_data_reduce_zero.csv')."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results.csv",
        help="Path for the output CSV file to be generated."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use for processing."
    )
    parser.add_argument(
        "--cleanup",
        action='store_true',
        help="If set, cleanup intermediate files after processing."
    )

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.isfile(args.input_csv):
        print(f"Error: The input CSV file '{args.input_csv}' does not exist.")
        exit(1)

    if not os.path.isfile(args.structure_json_for_novelty_check):
        print(f"Error: The structure JSON file '{args.structure_json_for_novelty_check}' does not exist.")
        exit(1)

    if not os.path.isfile(args.training_file):
        print(f"Error: The training file '{args.training_file}' does not exist.")
        exit(1)

    # Process data
    process_data(args.input_csv, args.output_csv, args.structure_json_for_novelty_check, args.threads)

    # Prepare data
    data_dict, sorted_keys, trainingset_values = prepare_data(args.output_csv, args.training_file)

    # Process all materials
    all_materials_df = create_dataframe(data_dict, sorted_keys, trainingset_values, 'all')

    # Process novel materials
    novel_materials_df = create_dataframe(data_dict, sorted_keys, trainingset_values, 'novel')
    fig = plot_combined_histograms(all_materials_df, novel_materials_df, "combined_results.png")

    # Optional cleanup
    if args.cleanup:
        os.system("rm energy_formation_chgnet_lists.csv energy_formation_chgnet_lists_novel.csv")
        print("Cleaned up intermediate CSV files.")


if __name__ == "__main__":
    main()
