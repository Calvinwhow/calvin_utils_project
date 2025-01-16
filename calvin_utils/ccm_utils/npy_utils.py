import numpy as np
import json

class DataLoader:
    def __init__(self, data_dict_path):
        with open(data_dict_path, 'r') as f:
            self.dataset_paths_dict = json.load(f)
    
    def load_dataset(self, dataset_name):
        paths = self.dataset_paths_dict[dataset_name]
        data = {
            'niftis': np.load(paths['niftis']),
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