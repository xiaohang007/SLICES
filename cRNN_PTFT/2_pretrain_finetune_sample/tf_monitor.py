#!/usr/bin/env python

from post_processing import sascorer, data
from rdkit import DataStructs
from rdkit import Chem
import numpy as np
import pandas as pd

"""
Analyses transfer learning results from one epoch of sampling.
Get SA scores, closest neighbours, number of promising candidates
from the generations.
"""


class TFEpoch:

    def __init__(self, training, gen):
        """

        Args:
            training: SMILES of molecules for transfer training input.
            gen: Generated smiles from one epoch of transfer learning
        """
        self.training = training
        self.gen = gen
        self.novel_gen = []

        train_mols = data.get_mols(self.training)
        gen_mols = data.get_mols(self.gen)
        self.train_fps, _ = data.get_fingerprints(train_mols)
        self.gen_fps, _ = data.get_fingerprints(gen_mols)

        assert (len(self.train_fps) == len(self.training))
        assert (len(self.gen_fps) == len(self.gen))

    def statistics(self, canolize=False):
        """

        Returns: Number of valid, valid_unique, valid_unique_novel molecules sampled from an epoch.

        """
        def can_smile(smi_list):
            can_list = []
            for item in smi_list:
                if Chem.MolFromSmiles(item) is not None:
                    can_item = Chem.MolToSmiles(Chem.MolFromSmiles(item))
                    can_list.append(can_item)
            return can_list

        if canolize:
            valid_lst = can_smile(self.gen)
            training_lst = can_smile(self.training)
        else:
            valid_lst = self.gen
            training_lst = self.training
        valid_count = len(valid_lst)
        unique_lst = list(set(valid_lst))
        unique_count = len(unique_lst)
        novel_lst = [item for item in unique_lst if item not in training_lst]
        novel_count = len(novel_lst)
        return valid_lst, valid_count, unique_lst, unique_count, novel_lst, novel_count

    def calc_sa(self):
        """

        Returns: SA scores of the generated molecules

        """
        sa_scores = []
        for smi in self.gen:
            gen_mol = Chem.MolFromSmiles(smi)
            sa_scores.append(sascorer.calculateScore(gen_mol))
        assert (len(self.gen) == len(sa_scores)), "Invalid SMILE string present, number of valid SMILES less" \
                                                      " than input list length. {} smiles, {} sa_scores."\
                                                         .format(len(self.gen), len(sa_scores))
        return sa_scores

    def calc_gap_dip(self):
        """

        Returns: HOMO LUMO gaps and dipole moments calculated using GBDT

        """
        gaps = data.predict_property('./post_processing/gbdt_regessor_gap_regu.joblib', self.gen_fps)
        dips = data.predict_property('./post_processing/gbdt_regessor_dip_reg.joblib',self.gen_fps)
        return gaps, dips, sum((np.array(gaps) <=2) & (np.array(dips) <= 3.66))

    def get_neighbour(self):
        """

        Returns: List of (closest neighbour, similarity) of the generated smiles

        """
        all_neighbours = []
        for i in range(len(self.gen)):
            tmp_fp = self.gen_fps[i]
            similarity = 0
            neighbour = ""
            for j in range(len(self.training)):
                tmp_sim = DataStructs.DiceSimilarity(tmp_fp, self.train_fps[j])
                if tmp_sim > similarity:
                    similarity = tmp_sim
                    neighbour = self.training[j]
            all_neighbours.append((similarity, neighbour))
        return all_neighbours


if __name__ == '__main__':
    train_df = pd.read_csv('Training_Model2_SA_NN.csv')
    gen_df = pd.read_csv('tf_da_model2_userinpt_10epoch_process.csv')
    model_summary = pd.DataFrame()
    train_lst = train_df['SMILES'].tolist()
    models = [2] * 10
    epochs = list(range(1,11))
    valid_counts = []
    unique_counts = []
    novel_counts = []
    for i in range(1,11):
        epoch_df = gen_df[gen_df['Epoch'] == i]
        epoch_smi = epoch_df['SMILES'].tolist()
        epoch = TFEpoch(train_lst, epoch_smi)
        _, valid_cnt, _, unique_cnt, _, novel_cnt = epoch.statistics(canolize=False)
        valid_counts.append(valid_cnt)
        unique_counts.append(unique_cnt)
        novel_counts.append(novel_cnt)
    model_summary['Model'] = models
    model_summary['Epoch'] = epochs
    model_summary['Valid'] = valid_counts
    model_summary['Unique'] = unique_counts
    model_summary['Novel'] = novel_counts






