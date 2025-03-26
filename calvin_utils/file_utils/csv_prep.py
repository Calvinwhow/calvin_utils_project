import re
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
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

    def __init__(self, data_dict, allow_loose_match=False):
        self.data_dict = data_dict
        self.composed_df = pd.DataFrame()  # Initialize an empty dataframe
        self.allow_loose_match=allow_loose_match

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
                    row_data[cov_name] = self._extract_covariate_value(csv_data, sub_idx, col_name, infill_colname=True)

                all_rows.append(row_data)

        self.composed_df = pd.DataFrame(all_rows)
    
    def _extract_covariate_value(self, df, row_filter, col_name, infill_colname=True):
        """
        Safely extract a single covariate value.
        If the column doesn't exist or there's no value, return a specific string.
        """
        if col_name not in df.columns:
            return col_name

        selected_vals = df.loc[row_filter, col_name]
        if not selected_vals.empty: 
            return selected_vals.values[0]
        else:
            if infill_colname: 
                return f"Value for '{col_name}' not found"
            else: 
                return f"Missing value for '{col_name}'"

    def _format_subject_id(self, subject_id):
        """Ensure subject ID is a string and handle NaNs."""
        if isinstance(subject_id, float):
            if isnan(subject_id):
                return 'nan'
            return str(int(subject_id))
        return str(subject_id)

    def _bids_match(self, nifti_paths, subject_id):
        """
        Match paths containing exactly 'sub-<subject_id>' with no digits after <subject_id>.
        E.g., subject_id='4' should match 'sub-4' but not 'sub-40'.
        """
        pattern = re.compile(re.escape('sub-' + subject_id) + r'(?!\d)')
        matches = [path for path in nifti_paths if pattern.search(path)]
        return matches
    
    def _regex_match(self, nifti_paths, subject_id):
        """
        Match paths containing 'sub-{leading zeros}{subject_id}' but NOT any extra digits afterward.
        For example, subject_id='4' should match sub-4, sub-04, sub-004, etc.,
        but NOT sub-14, sub-1234, etc.
        """
        # Explanation:
        #   'sub-'        matches literally
        #   '0*'          allows any number of leading zeros
        #   re.escape(...) ensures the subject_id is treated literally
        #   '(?!\\d)'     negative lookahead ensuring no digit immediately follows subject_id 
        #                 (prevents matching sub-1234 if subject_id=4)
        pattern = re.compile(r'sub-0*' + re.escape(subject_id) + r'(?!\d)')

        matches = [path for path in nifti_paths if pattern.search(path)]
        return matches
    
    def _loose_match(self, nifti_paths, subject_id):
        if self.allow_loose_match:
            return [path for path in nifti_paths if subject_id in path]
    
    def _find_nifti_path_for_subject(self, nifti_paths, subject_id):
        """
        Return the first NIfTI path that contains the subject_id; 
        Work through sequentially less strict matches. 
        remove from list to avoid reuse.
        """
        for match_function in [self._bids_match, self._regex_match, self._loose_match]:
            matches = match_function(nifti_paths, subject_id)
            if matches:
                nifti_paths.remove(matches[0])
                return matches[0]
        return ''

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
