from scipy.stats import rankdata
import numpy as np
import json
import os

class DataLoader:
    def __init__(self, data_dict_path):
        self.data_dict_path = data_dict_path
        with open(data_dict_path, 'r') as f:
            self.dataset_paths_dict = json.load(f)
        
    def load_dataset(self, dataset_name, nifti_type='niftis'):
        """
        Loads data for `dataset_name`. By default, returns raw 'niftis'.
        If `nifti_type='niftis_ranked'`, ensures 'niftis_ranked' exist, 
        creating them if necessary, then loads them.
        """
        paths = self.dataset_paths_dict[dataset_name]
        
        if nifti_type == 'niftis_ranked':
            self._init_ranked_niftis(dataset_name)
        
        data = {
            'niftis': np.load(paths[nifti_type]),
            'indep_var': np.load(paths['indep_var']),
            'covariates': np.load(paths['covariates'])
        }
        return data
    
    @staticmethod
    def load_dataset_static(data_paths_dict, dataset_name):
        paths = data_paths_dict[dataset_name]

        data_dict = {
            'niftis': np.load(paths['niftis']),
            'indep_var': np.load(paths['indep_var']),
            'covariates': np.load(paths['covariates'])
        }
        return data_dict
    
    def _rank_niftis(self, arr, vectorize=False):
        """
        Rank a 2D array (voxels, samples) ignoring ties.
        This is a 'sloppy' approach: ties get consecutive ranks arbitrarily.
        """
        if vectorize: 
            ranks = np.empty_like(arr, dtype=float)
            rows = np.arange(arr.shape[0])[:, None]
            cols = np.arange(arr.shape[1])
            idx_sorted = np.argsort(arr, axis=0)
            ranks[idx_sorted, cols] = rows
        else: 
            ranks = np.apply_along_axis(rankdata, 0, arr) 
        return ranks
    
    def _init_ranked_niftis(self, dataset_name):
        """
        Internal helper that creates and saves 'niftis_ranked' for the 
        specified dataset, then updates JSON so future runs skip re-ranking.
        """
        paths = self.dataset_paths_dict[dataset_name]
        if 'niftis_ranked' not in paths or not os.path.exists(paths['niftis_ranked']):
            print(f"Initializaing ranked data for: {dataset_name}")
            original_path = paths['niftis']  # The original unranked data
            ranked_path = os.path.splitext(original_path)[0] + '_ranked.npy'

            # Load & rank
            arr = np.load(original_path)
            ranked_arr = self._rank_niftis(arr)
            np.save(ranked_path, ranked_arr)

            # Update the JSON in memory
            self.dataset_paths_dict[dataset_name]['niftis_ranked'] = ranked_path

            # Write back to disk
            with open(self.data_dict_path, 'w') as fw:
                json.dump(self.dataset_paths_dict, fw, indent=4)