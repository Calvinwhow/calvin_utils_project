import os 
from sklearn.model_selection import train_test_split
from calvin_utils.ml_utils.smote import SmoteOversampler

class TrainTestSplitter:
    '''
    The following class will split `data_df` into train and test sets (80/20 split), 
    ensuring that no synthetic patients (as indicated by the `is_synthetic` column) are included in the test set. 
    If the `is_synthetic` column does not exist, it will perform a standard random split.
    '''
    def __init__(self, test_size=0.2, random_state=None, synthetic_data=True):
        self.test_size = test_size
        self.random_state = random_state
        self.synthetic_data = synthetic_data

    def split(self, df, stratify=None):
        if 'is_synthetic' in df.columns:
            real_df = df[~df['is_synthetic']]
            synth_df = df[df['is_synthetic']]
            train_real, test_df = train_test_split(
                real_df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=real_df[stratify] if stratify in real_df.columns else None
            )
            train_df = pd.concat([train_real, synth_df], ignore_index=True)
        else:
            train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state, stratify=df[stratify] if stratify in df.columns else None)
        return train_df, test_df
    
    def oversample(self, col, df):
        '''Returns a DataFrame with oversampled classes using SMOTE. Synthetic rows are tagged with `is_synthetic` column.'''
        return SmoteOversampler(col, sampling_strategy="auto").fit_resample(df)

    def save_splits(self, train_df, test_df, out_dir):
        if out_dir is not None:
            train_path = os.path.join(out_dir, 'train_data.csv')
            test_path = os.path.join(out_dir, 'test_data.csv')
            os.makedirs(out_dir, exist_ok=True)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            print(f"Train data saved to {train_path}. Proportion of synthetic data in train set: {train_df['is_synthetic'].mean()}")
        print(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
        
    def run(self, df, out_dir, stratify=None):
        train_df, test_df = self.split(df, stratify=stratify)
        train_df = self.oversample(stratify, train_df) if self.oversample else train_df
        self.save_splits(train_df, test_df, out_dir)
        return train_df, test_df