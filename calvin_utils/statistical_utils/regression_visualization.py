import numpy as np
import pandas as pd
import os 
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLMResults

class InteractionPlot:
    def __init__(self, model_results: GLMResults, data_df: pd.DataFrame, formula: str = None):
        """
        Initialize the InteractionPlot class with model results, data, and an optional formula.

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
        self.data_df = data_df
        self.formula = formula
        self.outcome, self.independent_vars = None, None
        self.outcome, self.independent_vars = self._parse_formula()

    def _parse_formula(self):
        """
        Parse the formula (either user-provided or from the fitted model) to extract the outcome and independent variables.
        
        Returns
        -------
        outcome : str
            The name of the dependent variable (outcome).
        independent_vars : list
            List of unique independent variables (excluding interactions).
        """
        # Use the custom formula if provided, otherwise use the model's formula
        formula = self.formula if self.formula is not None else self.model_results.model.formula
        
        # Split by '~' to separate outcome from predictors
        outcome, predictors = formula.split('~')
        outcome = outcome.strip()

        # Split the predictors by *, :, and +, and strip each
        predictors_list = predictors.replace('*', '+').replace(':', '+').split('+')
        predictors_list_stripped = [var.strip() for var in predictors_list]
        independent_vars = list(set(predictors_list_stripped))  # Get unique variables
        
        final_independent_vars_list = []
        for i, var in enumerate(predictors_list_stripped): 
            if var in independent_vars:
                final_independent_vars_list.append(var)
        return outcome, final_independent_vars_list
    
    def _create_labels(self):
        """
        Creates a dictionary of labels for each variable extracted from the formula.
        Labels are generated by capitalizing each word and replacing underscores with spaces.

        This method sets self.labels, which can be used in subsequent plotting functions.
        """
        # Ensure that independent variables and outcome are parsed
        if not hasattr(self, 'independent_vars') or not hasattr(self, 'outcome'):
            self.outcome, self.independent_vars = self._parse_formula()

        # Initialize the labels dictionary
        labels = {}

        # Format labels for independent variables and outcome
        for var in self.independent_vars + [self.outcome]:
            formatted_label = ' '.join([word.capitalize() for word in var.split('_')])
            labels[var] = formatted_label
        return labels


    def _get_predictions(self, X_grid, num_slices):
        try:
            Y_HAT = self.model_results.predict(X_grid).values.reshape(num_slices, num_slices)
        except ValueError as e: 
            if self.formula is not None:
                self.model_results = smf.ols(formula, data=data_df).fit()
                Y_HAT = self.model_results.predict(X_grid).values.reshape(num_slices, num_slices)
            else:
                raise ValueError(f"\n Value Error--to solve, pass the formula used for the original model to the init. \n Traceback: {e}")
        return Y_HAT
            
    def create_interaction_plot(self, num_slices: int = 100, labels: dict = None, out_dir: str = None, plot_residuals: bool = False):
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
        """
        # The first two variables are plotted on x and y axes, the rest will have sliders
        x_vars = self.independent_vars
        x1, x2 = x_vars[:2]
        slider_vars = x_vars[2:]  # Remaining variables will be controlled via sliders

        # Create a grid of values for x1 and x2
        x1_vals = np.linspace(min(self.data_df[x1]), max(self.data_df[x1]), num_slices)
        x2_vals = np.linspace(min(self.data_df[x2]), max(self.data_df[x2]), num_slices)
        x1v, x2v = np.meshgrid(x1_vals, x2_vals)

        # Set up the base grid of predictors
        X_grid = pd.DataFrame({
            x1: x1v.ravel(),
            x2: x2v.ravel()
        })

        # Create the figure
        fig = go.Figure()

        # Helper function to update the grid with slider values and predict outcomes
        def update_grid_with_slider_values(slider_vals):
            # Update grid with slider values for the remaining variables
            for i, slider_var in enumerate(slider_vars):
                X_grid[slider_var] = slider_vals[i]

            # Predict outcomes
            y_pred = self._get_predictions(X_grid, num_slices)
            return y_pred

        # Initialize the plot with the default slider values (median of each variable)
        initial_slider_vals = [self.data_df[var].median() for var in slider_vars]
        y_pred_initial = update_grid_with_slider_values(initial_slider_vals)

        # Add the surface plot with initial data
        fig.add_trace(go.Surface(z=y_pred_initial, x=x1_vals, y=x2_vals, colorscale='Greys', opacity=0.7))

        # Function to update the points and residuals on the plot
        def plot_points_and_residuals(fig, slider_vals):
            # Plot actual data points
            fig.add_trace(go.Scatter3d(
                x=self.data_df[x1], y=self.data_df[x2], z=self.data_df[self.outcome],
                mode='markers', marker=dict(color='red', size=5), name='Actual Points'))

            # Calculate and plot residuals (lines from actual to predicted values)
            for _, row in self.data_df.iterrows():
                x1_val, x2_val, y_val = row[x1], row[x2], row[self.outcome]
                slider_data = {var: slider_vals[i] for i, var in enumerate(slider_vars)}
                pred_val = self.model_results.predict(pd.DataFrame({**{x1: [x1_val], x2: [x2_val]}, **slider_data})).values[0]
                
                # Add residual line without showing it in the legend
                fig.add_trace(go.Scatter3d(
                    x=[x1_val, x1_val], y=[x2_val, x2_val], z=[y_val, pred_val],
                    mode='lines', line=dict(color='blue', dash='dash'), name='Residual', showlegend=False))  # Hide residuals from legend


        # Plot the initial points and residuals if the boolean flag is set
        if plot_residuals:
            plot_points_and_residuals(fig, initial_slider_vals)

        # Internal function to create sliders and position them correctly
        def create_sliders(slider_vars, initial_slider_vals):
            sliders = []
            
            # Calculate total padding area
            total_padding_space = 0.2  # Reserve 20% of the figure height for sliders
            
            # Calculate the start position for the first slider and the step
            slider_height_start = total_padding_space * 0.01  # Start at 10% of the padding area
            slider_height_step = total_padding_space / len(slider_vars)  # Evenly distribute sliders

            for i, slider_var in enumerate(slider_vars):
                # Define the steps for the slider (using min, max, and median as initial)
                slider_steps = [
                    dict(method='restyle',
                        args=[{'z': [update_grid_with_slider_values([val if j == i else initial_slider_vals[j] for j, _ in enumerate(slider_vars)])]}],
                        label=str(round(val, 2)))
                    for val in np.linspace(min(self.data_df[slider_var]), max(self.data_df[slider_var]), num=10)
                ]

                # Add the slider for the variable
                sliders.append({
                    'active': 5,
                    'currentvalue': {"prefix": f"{slider_var}: "},
                    'pad': {"t": 0},
                    'y': slider_height_start + i * slider_height_step,  # Increment position for each slider relative to the padding area
                    'len': 0.9,  # Length of the slider (0 to 1)
                    'steps': slider_steps
                })
            return sliders, total_padding_space

        # Create and position sliders
        sliders, total_padding_space = create_sliders(slider_vars, initial_slider_vals)

        # Update the layout with sliders
        fig.update_layout(
            scene=dict(
                xaxis_title=labels.get('x', x1) if labels else x1,
                yaxis_title=labels.get('y', x2) if labels else x2,
                zaxis_title=labels.get('z', self.outcome) if labels else self.outcome,
                domain=dict(y=[total_padding_space, 1.0])  # Shrink the plot to leave space for sliders
            ),
            sliders=sliders
        )

        # Adjust figure height based on the number of sliders
        fig.update_layout(
            width=1200,
            height=800 + len(slider_vars) * 100,  # Adjust height to account for padding and slider count
            margin=dict(l=20, r=20, t=20, b=20 + len(slider_vars) * 30)  # Add bottom margin to hold sliders
        )

        # Save as HTML for interaction
        if out_dir is not None:
            fig.write_html(os.path.join(out_dir, 'interactive_plot_with_residuals.html'))
            print('Saved to file', out_dir)

        return fig


    def run(self, num_slices: int = 100, out_dir: str = None, plot_residuals: bool = True):
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
        """
        if not hasattr(self, 'labels'):
            labels = self._create_labels()
        
        # Create the interaction plot
        return self.create_interaction_plot(num_slices=num_slices, labels=labels, out_dir=out_dir, plot_residuals=plot_residuals)