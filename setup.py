from setuptools import setup, find_packages

setup(
    name='calvin_utils',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        # Plotting and Visualization
        'seaborn==0.12.2',
        'plotly==5.15.0',
        'matplotlib==3.7.1',
        'matplotlib-inline',
        'matplotlib-venn',
        'forestplot',
        'tqdm',

        # Data Handling and Analysis
        'numpy',
        'h5py',
        'progressbar2==4.2.0',
        'jax',
        'paramiko',

        # Statistical and ML Packages
        'statsmodels==0.14.4',
        'patsy',
        'pandas',
        'keras',
        'tensorly==0.9.0',
        'sympy==1.13.1',
        'scikit-learn',
        'scipy',
        'pytensor',
        'hdbscan',
        'umap-learn',
        'pydot',

        # Neuroimaging and Medical Imaging
        'nilearn',
        'nibabel',
        'nipype',
        'vtk',
        'pydicom',

        # Additional requirements for the project
        'natsort',
        'jsonschema',
        'mamba',
        'ply',
        'simplejson',
        'tables',
        'tabulate',

        # For Notebooks
        'ipykernel',
        'ipython',
        'ipython-genutils',
        'ipywidgets',
        'jupyter',
        'jupyter-console',
        'jupyter-events',
        'jupyter_client',
        'jupyter_core',
        'jupyter_server',
        'jupyter_server_terminals',
        'jupyterlab-pygments',
        'jupyterlab-widgets',
        'Markdown',
        'markdown-it-py',
        'MarkupSafe',
        'notebook',
        'notebook_shim',
    ],
)
