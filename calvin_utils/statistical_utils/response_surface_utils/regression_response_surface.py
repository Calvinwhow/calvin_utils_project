import numpy as np
import pandas as pd
import os 
import itertools
from typing import Dict, Optional, List, Tuple
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.genmod.generalized_linear_model import GLMResults
from typing import Dict, Optional

class GLMPredictionComparison:
    def __init__(self, data_df: pd.DataFrame, formula: str, cohort_col: str, method='pearsonr'):
        """
        Initialize the GLMPredictionComparison class with data, formula, and cohort column.

        Parameters
        ----------
        data_df : pd.DataFrame
            The primary dataframe containing all necessary data, including the cohort indicator.
        formula : str
            The formula string for the GLM model (e.g., 'outcome ~ var1 + var2 + var1:var2').
        cohort_col : str
            The name of the column in data_df that indicates the cohort.
        method : str
            The scipy.stats correlation method to use. ex) pearsonr | spearmanr | kendalltau
        """
        self.data_df = data_df
        self.formula = formula
        self.cohort_col = cohort_col
        self.cohorts = self.data_df[cohort_col].unique()
        self.predictions: Dict[str, np.ndarray] = {}
        self.spatial_corrs: Dict[str, float] = {}
        self.method = method
        self.correlation_methods = {
            'pearsonr': pearsonr,
            'spearmanr': spearmanr,
            'kendalltau': kendalltau
            }
        
    def _get_cohort_df(self, cohort_id):
        return self.data_df[self.data_df[self.cohort_col] == cohort_id].copy()
    
    def _cohort_fit(self, cohort_id):
        df = self._get_cohort_df(cohort_id)
        return smf.ols(self.formula, data=df).fit()
        
    def _run_GLMPlot_predictions(self, model, df_cohort, num_slices):
        glm_plot = GLMPlot(model_results=model, data_df=df_cohort, formula=self.formula)
        _, independent_vars = glm_plot.outcome, glm_plot.independent_vars
        if len(independent_vars) < 2:
            raise ValueError(f"Not enough independent variables to create a grid for cohort '{df_cohort[self.cohort_col].iloc[0]}'.")
        x1, x2 = independent_vars[:2]
        slider_vars = independent_vars[2:]
        X_grid = glm_plot.prepare_X_grid(num_slices=num_slices, x1=x1, x2=x2, slider_vars=slider_vars)
        return glm_plot.get_predictions(X_grid, num_slices=num_slices)
        
        
    def _generate_predictions(self, num_slices: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate predictions for each cohort and store them in a dictionary.

        Parameters
        ----------
        num_slices : int, optional
            Number of slices in the grid for each predictor (default is 100).

        Returns
        -------
        dict
            A dictionary mapping each cohort identifier to its corresponding prediction array.
        """
        for cohort_id in self.cohorts:
            df = self._get_cohort_df(cohort_id)
            model = self._cohort_fit(cohort_id)
            self.predictions[cohort_id] = self._run_GLMPlot_predictions(model, df, num_slices)
            
    def _ensure_flat(self, ndarray):
        if ndarray.ndim > 1: 
            return ndarray.flatten()
        else: 
            return ndarray

    def _get_spatial_correlation(self):  
        corr_func = self.correlation_methods[self.method]      
        for cohort_pair in itertools.combinations(self.predictions.keys(), 2):
            pred_c1 = self._ensure_flat(self.predictions[cohort_pair[0]])
            pred_c2 = self._ensure_flat(self.predictions[cohort_pair[1]])
            
            corr, _ = corr_func(pred_c1, pred_c2)
            key = f"{cohort_pair[0]}-to-{cohort_pair[1]}"
            self.spatial_corrs[key] = corr
               
    def run(self):
        self._generate_predictions()
        self._get_spatial_correlation()
        return self.spatial_corrs

class GLMPlot:
    def __init__(self, model_results: GLMResults, data_df: pd.DataFrame, formula: str = None):
        """
        Initialize the GLMPlot class with model results, data, and an optional formula.

        Parameters
        ----------
        model_results : GLMResults
            The fitted model from statsmodels.
        data_df : pd.DataFrame
            The dataframe containing the predictor variables and outcome.
        formula : str, optional
            A custom formula provided by the user. If None, the model's formula will be used.
        """
        self.model_results = model_results
        self.data_df = data_df.copy()
        self.formula = formula
        self.outcome, self.independent_vars = self._parse_formula()
        self.labels = self._create_labels()
        self.categorical_vars, self.numeric_vars = self._identify_variable_types()
        self.category_mappings = self._encode_categorical_vars()

    def _parse_formula(self) -> Tuple[str, List[str]]:
        """
        Parse the formula (either user-provided or from the fitted model) to extract the outcome and independent variables.

        Returns
        -------
        tuple
            A tuple containing:
            - outcome (str): The name of the dependent variable (outcome).
            - independent_vars (list): List of unique independent variables (excluding interactions).
        """
        # Use the custom formula if provided, otherwise use the model's formula
        formula = self.formula if self.formula is not None else self.model_results.model.formula

        # Split by '~' to separate outcome from predictors
        outcome, predictors = formula.split('~')
        outcome = outcome.strip()

        # Split the predictors by '*', ':', and '+', and strip each
        predictors_list = predictors.replace('*', '+').replace(':', '+').split('+')
        predictors_list_stripped = [var.strip() for var in predictors_list]
        independent_vars = list(dict.fromkeys(predictors_list_stripped))  # Preserve order and get unique variables

        return outcome, independent_vars

    def _identify_variable_types(self) -> Tuple[List[str], List[str]]:
        """
        Identify which independent variables are categorical and which are numerical.

        Returns
        -------
        tuple
            A tuple containing two lists:
            - categorical_vars (list): List of categorical independent variables.
            - numeric_vars (list): List of numerical independent variables.
        """
        categorical_vars = []
        numeric_vars = []
        for var in self.independent_vars:
            if self.data_df[var].dtype == 'object' or pd.api.types.is_categorical_dtype(self.data_df[var]):
                categorical_vars.append(var)
            else:
                numeric_vars.append(var)
        return categorical_vars, numeric_vars

    def _encode_categorical_vars(self) -> Dict[str, Dict[int, str]]:
        """
        Encode categorical variables using Ordinal Encoding while preserving their original labels.

        Returns
        -------
        dict
            A dictionary mapping each categorical variable to its integer-to-category mapping.
        """
        category_mappings = {}
        for var in self.categorical_vars:
            self.data_df[var] = self.data_df[var].astype('category')
            category_mappings[var] = {i: cat for i, cat in enumerate(self.data_df[var].cat.categories)}
            self.data_df[var] = self.data_df[var].cat.codes
        return category_mappings

    def _create_labels(self) -> Dict[str, str]:
        """
        Creates a dictionary of labels for each variable extracted from the formula.
        Labels are generated by capitalizing each word and replacing underscores with spaces.

        Returns
        -------
        dict
            A dictionary mapping variable names to their formatted labels.
        """
        labels = {}

        # Format labels for independent variables and outcome
        for var in self.independent_vars + [self.outcome]:
            formatted_label = ' '.join([word.capitalize() for word in var.split('_')])
            labels[var] = formatted_label

        return labels

    def prepare_X_grid(self, num_slices: int, x1: str, x2: str, slider_vars: List[str]) -> pd.DataFrame:
        """
        Create a grid of predictor values for generating predictions.

        Parameters
        ----------
        num_slices : int
            Number of slices in the grid for each predictor.
        x1 : str
            The first independent variable for the x-axis.
        x2 : str
            The second independent variable for the y-axis.
        slider_vars : list
            The remaining independent variables to control via sliders.

        Returns
        -------
        pd.DataFrame
            The grid dataframe with all predictor combinations.
        """
        # Create a grid for x1 and x2
        x1_vals = np.linspace(self.data_df[x1].min(), self.data_df[x1].max(), num_slices)
        x2_vals = np.linspace(self.data_df[x2].min(), self.data_df[x2].max(), num_slices)
        x1v, x2v = np.meshgrid(x1_vals, x2_vals)

        # Initialize the grid dataframe
        X_grid = pd.DataFrame({
            x1: x1v.ravel(),
            x2: x2v.ravel()
        })

        # Set remaining variables to their median (numeric) or mode (categorical)
        for var in slider_vars:
            if var in self.numeric_vars:
                median_val = self.data_df[var].median()
                X_grid[var] = median_val
            elif var in self.categorical_vars:
                mode_val = self.data_df[var].mode()[0]
                X_grid[var] = mode_val
            else:
                raise ValueError(f"Variable '{var}' is neither numeric nor categorical.")

        return X_grid

    def get_predictions(self, X_grid: pd.DataFrame, num_slices: int) -> np.ndarray:
        """
        Generate predictions based on the provided grid.

        Parameters
        ----------
        X_grid : pd.DataFrame
            The grid dataframe with all predictor combinations.
        num_slices : int
            Number of slices in the grid for reshaping.

        Returns
        -------
        np.ndarray
            The array of predicted values reshaped into a grid.
        """
        expected_size = num_slices * num_slices
        actual_size = X_grid.shape[0]
        if actual_size != expected_size:
            raise ValueError(f"Expected X_grid to have {expected_size} rows, but got {actual_size}.")

        try:
            Y_HAT = self.model_results.predict(X_grid).values.reshape(num_slices, num_slices)
        except ValueError as e:
            if self.formula is not None:
                print("ValueError encountered. Refitting the model with the provided formula.")
                self.model_results = smf.ols(self.formula, data=self.data_df).fit()
                Y_HAT = self.model_results.predict(X_grid).values.reshape(num_slices, num_slices)
            else:
                raise ValueError(f"Value Error: {e}")
        return Y_HAT

    def plot_actual_points(self, fig: go.Figure, x1: str, x2: str, outcome: str):
        """
        Plot the actual data points on the 3D plot.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure object to add traces to.
        x1 : str
            The first independent variable.
        x2 : str
            The second independent variable.
        outcome : str
            The dependent variable.
        """
        # Decode categorical variables for plotting
        x1_decoded = self._decode_variable(x1)
        x2_decoded = self._decode_variable(x2)
        outcome_decoded = self.data_df[outcome].tolist()  # Assuming outcome is numeric

        fig.add_trace(go.Scatter3d(
            x=x1_decoded,
            y=x2_decoded,
            z=outcome_decoded,
            mode='markers',
            marker=dict(color='red', size=5),
            name='Actual Points'
        ))

    def plot_residuals(self, fig: go.Figure, x1: str, x2: str, outcome: str, slider_vals: List[float], slider_vars: List[str]):
        """
        Plot residuals as lines from actual to predicted values.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure object to add traces to.
        x1 : str
            The first independent variable.
        x2 : str
            The second independent variable.
        outcome : str
            The dependent variable.
        slider_vals : list
            Current values of the slider-controlled variables.
        slider_vars : list
            The remaining independent variables controlled via sliders.
        """
        # Create a DataFrame with slider variables set to slider_vals
        slider_data = {}
        for i, var in enumerate(slider_vars):
            slider_data[var] = [slider_vals[i]] * len(self.data_df)

        X_pred = pd.DataFrame({
            x1: self.data_df[x1],
            x2: self.data_df[x2],
            **slider_data
        })

        # Generate predictions
        Y_pred = self.model_results.predict(X_pred).values

        for idx, row in self.data_df.iterrows():
            x1_val, x2_val, y_val = row[x1], row[x2], row[outcome]
            pred_val = Y_pred[idx]

            # Decode categorical variables for plotting
            x1_val_decoded = self._decode_single_value(x1, x1_val)
            x2_val_decoded = self._decode_single_value(x2, x2_val)
            y_val_decoded = y_val  # Assuming outcome is numeric
            pred_val_decoded = pred_val  # Assuming outcome is numeric

            fig.add_trace(go.Scatter3d(
                x=[x1_val_decoded, x1_val_decoded],
                y=[x2_val_decoded, x2_val_decoded],
                z=[y_val_decoded, pred_val_decoded],
                mode='lines',
                line=dict(color='blue', dash='dash'),
                name='Residual',
                showlegend=False
            ))

    def create_sliders(self, slider_vars: List[str], initial_slider_vals: List[float], num_steps: int = 10) -> List[dict]:
        """
        Create slider configurations for additional variables.

        Parameters
        ----------
        slider_vars : list
            The remaining independent variables controlled via sliders.
        initial_slider_vals : list
            The initial values for the sliders.
        num_steps : int
            Number of steps for each slider.

        Returns
        -------
        list
            A list of slider configuration dictionaries.
        """
        sliders = []
        total_padding_space = 0.2  # Reserve 20% of the figure height for sliders
        slider_height_start = 0.1  # Start at 10% of the padding area
        slider_height_step = (total_padding_space * 0.8) / len(slider_vars)  # Evenly distribute sliders within 80% of padding

        for i, slider_var in enumerate(slider_vars):
            if slider_var in self.numeric_vars:
                var_min = self.data_df[slider_var].min()
                var_max = self.data_df[slider_var].max()
                slider_steps = [
                    dict(
                        method='restyle',
                        args=[{'z': [self.update_z_with_slider_values(
                            slider_vals=[val if j == i else initial_slider_vals[j] for j in range(len(slider_vars))],
                            x1=self.independent_vars[0],
                            x2=self.independent_vars[1],
                            slider_vars=slider_vars,
                            num_slices=100
                        )]}],
                        label=str(round(val, 2))
                    )
                    for val in np.linspace(var_min, var_max, num_steps)
                ]
            elif slider_var in self.categorical_vars:
                categories = list(self.category_mappings[slider_var].values())
                slider_steps = [
                    dict(
                        method='restyle',
                        args=[{'z': [self.update_z_with_slider_values(
                            slider_vals=[self._encode_single_category(slider_var, cat) if j == i else initial_slider_vals[j] for j in range(len(slider_vars))],
                            x1=self.independent_vars[0],
                            x2=self.independent_vars[1],
                            slider_vars=slider_vars,
                            num_slices=100
                        )]}],
                        label=cat
                    )
                    for cat in categories
                ]
            else:
                raise ValueError(f"Variable '{slider_var}' is neither numeric nor categorical.")

            sliders.append({
                'active': 0,
                'currentvalue': {"prefix": f"{self.labels.get(slider_var, slider_var)}: "},
                'pad': {"t": 0},
                'y': slider_height_start + i * slider_height_step,
                'len': 0.9,
                'steps': slider_steps
            })

        return sliders

    def _encode_single_category(self, var: str, category: str) -> int:
        """
        Encode a single category value.

        Parameters
        ----------
        var : str
            The categorical variable name.
        category : str
            The category label to encode.

        Returns
        -------
        int
            The encoded integer value for the category.
        """
        for code, cat in self.category_mappings[var].items():
            if cat == category:
                return code
        raise ValueError(f"Category '{category}' not found in variable '{var}'.")

    def update_z_with_slider_values(self, slider_vals: List[float], x1: str, x2: str, slider_vars: List[str], num_slices: int) -> np.ndarray:
        """
        Update the Z-axis predictions based on slider values.

        Parameters
        ----------
        slider_vals : list
            Current values of the sliders.
        x1 : str
            The first independent variable.
        x2 : str
            The second independent variable.
        slider_vars : list
            The remaining independent variables controlled via sliders.
        num_slices : int
            Number of slices in the grid for reshaping.

        Returns
        -------
        np.ndarray
            The updated Z-axis prediction values.
        """
        # Create a copy of X_grid to modify
        X_grid_modified = self.X_grid.copy()
        for i, var in enumerate(slider_vars):
            if var in self.categorical_vars:
                X_grid_modified[var] = int(slider_vals[i])  # Ensure categorical vars are integers
            else:
                X_grid_modified[var] = slider_vals[i]
        y_pred = self.get_predictions(X_grid_modified, num_slices)
        return y_pred

    def add_sliders_to_layout(self, fig: go.Figure, sliders: List[dict], total_padding_space: float):
        """
        Add sliders to the Plotly figure layout.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure object.
        sliders : list
            The slider configurations.
        total_padding_space : float
            The total padding space allocated for sliders.
        """
        fig.update_layout(
            scene=dict(
                xaxis_title=self.labels.get(self.independent_vars[0], self.independent_vars[0]),
                yaxis_title=self.labels.get(self.independent_vars[1], self.independent_vars[1]),
                zaxis_title=self.labels.get(self.outcome, self.outcome),
                domain=dict(y=[0.1, 1.0])  # Adjust domain to leave space for sliders
            ),
            sliders=sliders
        )

        fig.update_layout(
            width=1200,
            height=800 + len(sliders) * 100,  # Adjust height based on number of sliders
            margin=dict(l=20, r=20, t=20, b=20 + len(sliders) * 30)  # Add bottom margin to hold sliders
        )

    def create_surface_plot(self, x1_vals: np.ndarray, x2_vals: np.ndarray, y_pred_initial: np.ndarray) -> go.Figure:
        """
        Create the initial surface plot.

        Parameters
        ----------
        x1_vals : np.ndarray
            Values for the x1 axis.
        x2_vals : np.ndarray
            Values for the x2 axis.
        y_pred_initial : np.ndarray
            Initial Z-axis prediction values.

        Returns
        -------
        go.Figure
            The Plotly figure object with the surface plot.
        """
        fig = go.Figure()
        fig.add_trace(go.Surface(
            z=y_pred_initial,
            x=x1_vals,
            y=x2_vals,
            colorscale='Greys',
            opacity=0.7,
            name='Prediction Surface'
        ))
        return fig

    def _decode_variable(self, var: str) -> List[Optional[str]]:
        """
        Decode a categorical variable back to its original category labels.

        Parameters
        ----------
        var : str
            The variable name to decode.

        Returns
        -------
        list
            A list of decoded category labels or the original numeric values.
        """
        if var in self.categorical_vars:
            return self.data_df[var].map({v: k for k, v in self.category_mappings[var].items()}).tolist()
        else:
            return self.data_df[var].tolist()

    def _decode_single_value(self, var: str, value: float) -> Optional[str]:
        """
        Decode a single categorical variable value back to its original category label.

        Parameters
        ----------
        var : str
            The variable name to decode.
        value : float
            The numeric value to decode.

        Returns
        -------
        str or float
            The decoded category label or the original numeric value.
        """
        if var in self.categorical_vars:
            return self.category_mappings[var].get(int(value), value)
        else:
            return value

    def create_interaction_plot(self, num_slices: int = 100, labels: dict = None, out_dir: str = None, plot_residuals: bool = False) -> go.Figure:
        """
        Create a 3D plot visualizing the interaction of two predictor variables on the outcome with sliders for additional variables.

        Parameters
        ----------
        num_slices : int, optional
            Number of slices to create in the 3D grid, default is 100.
        labels : dict, optional
            Custom labels for the axes, default is None.
        out_dir : str, optional
            Directory to save the output HTML file, default is the current directory.
        plot_residuals : bool, optional
            If True, plots the data points and residuals, default is False.

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        # Define variables
        x_vars = self.independent_vars
        if len(x_vars) < 2:
            raise ValueError("At least two independent variables are required for plotting.")
        x1, x2 = x_vars[:2]
        slider_vars = x_vars[2:]

        # Create a grid of values for x1 and x2
        X_grid = self.prepare_X_grid(num_slices=num_slices, x1=x1, x2=x2, slider_vars=slider_vars)
        self.X_grid = X_grid  # Store for slider callbacks

        # Generate initial predictions
        y_pred_initial = self.get_predictions(X_grid, num_slices=num_slices)

        # Create the surface plot
        fig = self.create_surface_plot(
            x1_vals=np.linspace(self.data_df[x1].min(), self.data_df[x1].max(), num_slices),
            x2_vals=np.linspace(self.data_df[x2].min(), self.data_df[x2].max(), num_slices),
            y_pred_initial=y_pred_initial
        )

        # Plot residuals if required
        if plot_residuals:
            self.plot_actual_points(fig, x1=x1, x2=x2, outcome=self.outcome)
            initial_slider_vals = [
                self.data_df[var].median() if var in self.numeric_vars else self.data_df[var].mode()[0] for var in slider_vars
            ]
            self.plot_residuals(fig, x1=x1, x2=x2, outcome=self.outcome, slider_vals=initial_slider_vals, slider_vars=slider_vars)

        # Create sliders if there are additional variables
        if slider_vars:
            initial_slider_vals = [
                self.data_df[var].median() if var in self.numeric_vars else self.data_df[var].mode()[0] for var in slider_vars
            ]
            sliders = self.create_sliders(slider_vars=slider_vars, initial_slider_vals=initial_slider_vals)
            self.add_sliders_to_layout(fig, sliders=sliders, total_padding_space=0.2)
        else:
            fig.update_layout(
                scene=dict(
                    xaxis_title=labels.get(x1, self.labels.get(x1, x1)) if labels else self.labels.get(x1, x1),
                    yaxis_title=labels.get(x2, self.labels.get(x2, x2)) if labels else self.labels.get(x2, x2),
                    zaxis_title=labels.get(self.outcome, self.labels.get(self.outcome, self.outcome)) if labels else self.labels.get(self.outcome, self.outcome)
                ),
                width=1200,
                height=800,
                margin=dict(l=20, r=20, t=20, b=20)
            )

        # Save as HTML for interaction
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            file_path = os.path.join(out_dir, 'interactive_plot_with_residuals.html')
            fig.write_html(file_path)

        return fig

    def run(self, num_slices: int = 100, out_dir: str = None, plot_residuals: bool = False) -> go.Figure:
        """
        Orchestrates the entire process: parses the formula, extracts variables, and creates the plot.

        Parameters
        ----------
        num_slices : int, optional
            Number of slices to create in the 3D grid, default is 100.
        out_dir : str, optional
            Directory to save the output HTML file, default is the current directory.
        plot_residuals : bool, optional
            If True, plots the data points and residuals, default is False.

        Returns
        -------
        go.Figure
            The Plotly figure object.
        """
        return self.create_interaction_plot(num_slices=num_slices, labels=self.labels, out_dir=out_dir, plot_residuals=plot_residuals)