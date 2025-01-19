import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import os
from numpy import isnan

class CSVComposer:
    """
    Composes a CSV file from a dictionary of dataset parameters.

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key is a dataset name and the value is a dictionary with:
        - 'nifti_path': str, wildcarded path to NIfTI files (e.g. '/some/path/*_T1w.nii.gz').
        - 'csv_path': str, path to a CSV file containing subject data.
        - 'subj_col': str, column name in the CSV file containing subject IDs.
        - 'covariate_col': dict, mapping of desired covariate name -> actual column name in CSV.
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.composed_df = pd.DataFrame()  # Initialize an empty dataframe

    def compose_df(self):
        all_rows = []

        for dataset, params in self.data_dict.items():
            nifti_paths = glob(params['nifti_path'])
            csv_data = pd.read_csv(params['csv_path'])
            unique_subjects = csv_data[params['subj_col']].unique()

            for subject_id in tqdm(unique_subjects, desc=f"Processing {dataset}"):
                subject_id_str = self._format_subject_id(subject_id)
                matched_path = self._find_nifti_path_for_subject(nifti_paths, subject_id_str)

                # Filter rows for this subject
                sub_idx = csv_data[params['subj_col']] == subject_id
                row_data = {
                    'Dataset': dataset,
                    'Subject': subject_id_str,
                    'Nifti_File_Path': matched_path,
                }

                # Extract covariate values
                for cov_name, col_name in params.get('covariate_col', {}).items():
                    row_data[cov_name] = self._extract_covariate_value(csv_data, sub_idx, col_name)

                all_rows.append(row_data)

        self.composed_df = pd.DataFrame(all_rows)

    def _format_subject_id(self, subject_id):
        """Ensure subject ID is a string and handle NaNs."""
        if isinstance(subject_id, float):
            if isnan(subject_id):
                return 'nan'
            return str(int(subject_id))
        return str(subject_id)

    def _find_nifti_path_for_subject(self, nifti_paths, subject_id):
        """Return the first NIfTI path that contains the subject_id; remove from list to avoid reuse."""
        matches = [path for path in nifti_paths if subject_id in path]
        if matches:
            nifti_paths.remove(matches[0])
            return matches[0]
        return ''

    def _extract_covariate_value(self, df, row_filter, col_name, infill_colname=True):
        """
        Safely extract a single covariate value.
        If the column doesn't exist or there's no value, return None.
        """
        if col_name not in df.columns:
            return None

        selected_vals = df.loc[row_filter, col_name]
        if not selected_vals.empty: 
            return selected_vals.values[0]
        else:
            if infill_colname: return col_name
            else: return None

    def save_csv(self, output_csv_path):
        """
        Save the composed DataFrame to a CSV file.
        """
        self.composed_df.to_csv(output_csv_path, index=False)

    def save_dict_as_json(self, output_json_path):
        """
        Save the original data dictionary as JSON for record-keeping.
        """
        with open(output_json_path, 'w') as json_file:
            json.dump(self.data_dict, json_file, indent=4)

# import pandas as pd
# import json
# from glob import glob
# from tqdm import tqdm
# import os
# from numpy import isnan
# class CSVComposer:
#     """
#     A class to compose a CSV file from a dictionary of dataset parameters.

#     This class takes a dictionary where each key represents a dataset and the value is another dictionary containing
#     parameters such as the path to NIFTI files, the path to a CSV file, and column names for subjects,
#     and covariates. It then composes a CSV file with columns for dataset, subject, NIFTI file path,
#     and covariate columns. Covariate columns should contain the independent variable. 

#     Parameters:
#     -----------
#     data_dict : dict
#         A dictionary where each key is a dataset name and the value is a dictionary with the following keys:
#         - 'nifti_path': str, a wildcarded (* are wildcards) path to NIFTI files.
#         - 'csv_path': str, the absolute path to a CSV file containing subject data.
#         - 'subj_col': str, the column name in the CSV file that contains subject IDs.
#         - 'covariate_col': dict, a dictionary where keys are common covariate names and values are the corresponding column names in the CSV file.

#     Methods:
#     --------
#     compose_csv(output_csv_path):
#         Composes the CSV file and saves it to the specified output path.

#     _extract_subject_id(nifti_path, csv_data, subj_col):
#         Extracts the subject ID from the NIFTI file path based on the provided CSV data and subject column.
#     """

#     def __init__(self, data_dict):
#         self.data_dict = data_dict

#     def compose_df(self):
#         all_data = []

#         for dataset, params in self.data_dict.items():
#             print("Processing dataset:", dataset)
#             nifti_paths = glob(params['nifti_path'])
#             csv_data = pd.read_csv(params['csv_path'])
            
#             for subject_id in tqdm(csv_data[params['subj_col']].unique(), desc='processing subjects'):
#                 subject_id = self._format_subject_id(subject_id)
#                 matched_nifti_path = self._find_nifti_path_for_subject(nifti_paths, subject_id)
#                 sub_idx = csv_data[params['subj_col']] == subject_id # Filter the CSV data for the current subject
#                 covariates = self._handle_covariates(csv_data, sub_idx, params)

#                 row = {
#                     'Dataset': dataset,
#                     'sub': subject_id,
#                     'Nifti_File_Path': matched_nifti_path,
#                 }

#                 row.update(covariates)
#                 all_data.append(row)

#         self.composed_df = pd.DataFrame(all_data)
    
#     def _handle_covariates(self, csv_data, sub_idx, params):
#         covariates = {}
#         for key, col in params['covariate_col'].items():
#             if col not in csv_data.columns:
#                 csv_data.loc[:, [col]] = col # Add a column of dummy values if the column is not found
#             print(csv_data.loc[sub_idx, col])
#             try: 
#                 covariates[key] = csv_data.loc[sub_idx, col].values[0] # Extract the covariate value
#             except:
#                 print(f"WARNING: Could not extract value for covariate {col}. Setting value = {col}")
#                 covariates[key] = col
#         return covariates
        
#     def _format_subject_id(self, subject_id):
#         if isinstance(subject_id, float):
#             if isnan(subject_id): return 'nan'
#             else: return str(int(subject_id))
#         return str(subject_id)
    
#     def _find_nifti_path_for_subject(self, nifti_paths, subject_id):
#         matched_paths = [path for path in nifti_paths if subject_id in path]
#         if matched_paths:
#             nifti_paths.remove(matched_paths[0])
#             return matched_paths[0]
#         else:
#             return ''
    
#     # def _extract_subject_id(self, nifti_path, csv_data, subj_col):
#     #     # Assuming subject ID can be extracted from the file name
#     #     file_name = os.path.basename(nifti_path)
#     #     subject_id = file_name.split('_')[0]  # Modify this based on your file naming convention
#     #     return subject_id
    
#     def save_csv(self, output_csv_path):
#         self.composed_df.to_csv(output_csv_path, index=False)
        
#     def save_dict_as_json(self, output_json_path):
#         with open(output_json_path, 'w') as json_file:
#             json.dump(self.data_dict, json_file, indent=4)

    