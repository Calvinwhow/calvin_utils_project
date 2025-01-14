import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import os

class CSVComposer:
    """
    A class to compose a CSV file from a dictionary of dataset parameters.

    This class takes a dictionary where each key represents a dataset and the value is another dictionary containing
    parameters such as the path to NIFTI files, the path to a CSV file, and column names for subjects, independent variables,
    and covariates. It then composes a CSV file with columns for dataset, subject, NIFTI file path, independent variable,
    and covariate columns.

    Parameters:
    -----------
    data_dict : dict
        A dictionary where each key is a dataset name and the value is a dictionary with the following keys:
        - 'nifti_path': str, a wildcarded (* are wildcards) path to NIFTI files.
        - 'csv_path': str, the absolute path to a CSV file containing subject data.
        - 'subj_col': str, the column name in the CSV file that contains subject IDs.
        - 'indep_col': str, the column name in the CSV file that contains the independent variable.
        - 'covariate_col': dict, a dictionary where keys are common covariate names and values are the corresponding column names in the CSV file.

    Methods:
    --------
    compose_csv(output_csv_path):
        Composes the CSV file and saves it to the specified output path.

    _extract_subject_id(nifti_path, csv_data, subj_col):
        Extracts the subject ID from the NIFTI file path based on the provided CSV data and subject column.
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict

    def compose_df(self):
        all_data = []

        for dataset, params in self.data_dict.items():
            print("Processing dataset:", dataset)
            nifti_paths = glob(params['nifti_path'])
            csv_data = pd.read_csv(params['csv_path'])

            for subject_id in tqdm(csv_data[params['subj_col']].unique(), desc='processing subjects'):
                matched_nifti_path = self._find_nifti_path_for_subject(nifti_paths, subject_id)
                if matched_nifti_path is None:         # Skip if no NIFTI file found for the subject
                    continue
                
                sub_idx = csv_data[params['subj_col']] == subject_id # Filter the CSV data for the current subject
                indep_var = csv_data.loc[sub_idx, params['indep_col']].values[0]
                covariates = self._handle_covariates(csv_data, sub_idx, params)

                row = {
                    'Dataset': dataset,
                    'sub': subject_id,
                    'Nifti_File_Path': matched_nifti_path,
                    'Indep. Var.': indep_var,
                }

                row.update(covariates)
                all_data.append(row)

        self.composed_df = pd.DataFrame(all_data)
    
    def _handle_covariates(self, csv_data, sub_idx, params):
        covariates = {}
        for key, col in params['covariate_col'].items():
            if col not in csv_data.columns:
                csv_data.loc[sub_idx, col] = col # Add a dummy value if the column is not found
            covariates[key] = csv_data.loc[sub_idx, col].values[0] # Extract the covariate value
        return covariates
        
        
    def _find_nifti_path_for_subject(self, nifti_paths, subject_id):
        subject_id_str = str(subject_id)
        matched_paths = [path for path in nifti_paths if subject_id_str in path]
        if matched_paths:
            nifti_paths.remove(matched_paths[0])
            return matched_paths[0]
        return None
    
    # def _extract_subject_id(self, nifti_path, csv_data, subj_col):
    #     # Assuming subject ID can be extracted from the file name
    #     file_name = os.path.basename(nifti_path)
    #     subject_id = file_name.split('_')[0]  # Modify this based on your file naming convention
    #     return subject_id
    
    def save_csv(self, output_csv_path):
        self.composed_df.to_csv(output_csv_path, index=False)
        
    def save_dict_as_json(self, output_json_path):
        with open(output_json_path, 'w') as json_file:
            json.dump(self.data_dict, json_file, indent=4)

    