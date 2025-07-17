import numpy as np
import nibabel as nib
import umap
import hdbscan
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import pandas as pd

class BrainUmap:
    """
    Spatially-aware UMAP + HDBSCAN workflow for 3-D brain volumes.

    The class converts a collection of volumetric maps (one per subject) into a
    low-dimensional embedding that jointly encodes voxel intensities and their
    spatial coordinates, then clusters the embedding to identify coherent
    anatomical/functional regions.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Each row is a voxel, each column is a map (subject or condition).  
        Shape = (n_voxels, n_maps).
    n_components : int, default 3
        Target dimensionality of the UMAP embedding.
    n_neighbors : int, default 15
        Size of the local neighborhood UMAP uses to balance local vs. global
        structure.
    min_dist : float, default 0.1
        Minimum distance between points in the embedded space (UMAP *min_dist*).
    metric : str or callable, default "correlation"
        Distance metric passed to UMAP when *cluster_voxels=True*. When
        *cluster_voxels=False* the metric is only used for pair-wise distance
        calculation after projection. See ``umap.UMAP`` docs for options.
    projection : {"sphere", None}, default "sphere"
        • "sphere" — radially projects a 3-D embedding onto the unit sphere  
        • None      — no projection
    min_cluster_size : int, default 5
        Minimum cluster size for HDBSCAN.
    mask : str or None, default None
        Path to a NIfTI mask; voxels with 0 are excluded. If None, all voxels
        are used.
    verbose : bool, default True
        If True, prints status messages.
    cluster_voxels : bool, default True
        If True, clusters individual voxels in embedding space.  
        If False, clusters whole maps using a pre-computed distance matrix.

    Attributes
    ----------
    mask : ndarray or None
        Boolean flat mask applied to the voxel axis.
    brain_array : ndarray, shape (n_maps, n_selected_voxels)
        Masked data matrix.
    features : ndarray, shape (n_maps, n_selected_voxels)
        Z-scored intensity features fed to UMAP.
    embedding : ndarray, shape (n_samples, n_components)
        Low-dimensional representation after optional projection.
    distances : ndarray or None
        Pair-wise distance matrix (only when *cluster_voxels=False*).
    cluster_labels : ndarray, shape (n_samples,)
        Label of the HDBSCAN cluster assigned to each point; -1 for noise.
    cluster_probabilities : ndarray
        Soft cluster membership probabilities.
    cluster_persistence : ndarray
        HDBSCAN cluster persistence scores.

    Methods
    -------
    plot_embedding(point_size=3, opacity=0.6)
        Interactive Plotly scatter/Scatter3d of the embedding with cluster
        labels.
    run()
        Convenience wrapper that calls ``plot_embedding()``.

    Notes
    -----
    • Data are expected row-wise as voxels (long format) to avoid reshaping the
      4-D array in RAM.  
    • When *projection="sphere"* the embedding must be 3-D; the class rescales
      each point onto the unit sphere before clustering.  
    • Setting *cluster_voxels=False* yields a subject-level clustering based on
      pre-computed distances in embedding space.

    Examples
    --------
    >>> umap_runner = BrainUmap(df, mask="brainmask.nii.gz",
    ...                         n_components=3, projection="sphere",
    ...                         metric="correlation", cluster_voxels=True)
    >>> umap_runner.plot_embedding()
    """
    def __init__(self, data, n_components=2, n_neighbors=15, min_dist=0.1, metric='correlation', projection="sphere", min_cluster_size=5, mask=None, verbose=True, cluster_voxels=False):
        """
        metric: 'euclidean', 'manhattan', 'cosine', 'correlation', 'haversine', etc. (see umap.UMAP docs)
        min_dist recommendation: 0.1 (low for tight clusters, higher for more spread out embedding)
        n_neighbors recommendation: 15 (smaller for local structure, larger for global structure)
        n_components recommendation: 2 or 3 (for visualization; higher for downstream ML tasks)
        """
        self.verbose = verbose
        self.projection = projection
        arr = self._get_arr(data)
        self.mask = self._get_mask(mask)
        self.brain_array = self._mask_arr(arr, cluster_voxels) # (N maps, N voxels)
        self.features = self._get_features()
        self.embedding = self._run_umap(n_components, n_neighbors, min_dist, metric)
        self.embedding = self._project_embedding()
        self.distances = self._compute_distance(metric, cluster_voxels)
        self.cluster_labels, self.cluster_probabilities, self.cluster_persistence = self._run_hdbscan(min_cluster_size, cluster_voxels)
    
    ### Setters and Getters ###
    def _get_arr(self, data):
        if isinstance(data, pd.DataFrame):
            arr = data.values
        else:
            arr = np.asarray(data)
        return arr.T
        
    def _get_mask(self, path):
        '''Imports a nifti mask'''
        return nib.load(path).get_fdata().flatten() > 0 if path else None
    
    def _mask_arr(self, arr, cluster_voxels):
        '''Masks the array'''
        if self.mask is not None:
            arr = arr[:, self.mask]
        return arr.T if cluster_voxels else arr
            
    def _get_features(self):
        """Creates voxel-wise feature vectors combining spatial and intensity information."""
        return StandardScaler().fit_transform(self.brain_array) # standardizes expecting (N maps, N voxels)
    
    ### Internal API ###
    def _run_umap(self, n_components, n_neighbors, min_dist, metric):
        """Runs UMAP dimensionality reduction."""
        umapper = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
        return umapper.fit_transform(self.features) # learns on (N maps, N voxels)
    
    def _project_embedding(self, e=1e-20):
        '''Projects the embedding if the embedding space is created'''
        if self.projection=="sphere": 
            if self.embedding.shape[1] != 3:
                raise ValueError("Embedding must have 3 dimensions. Set n_components=3.")
            radii = np.linalg.norm(self.embedding, axis=1, keepdims=True)
            return self.embedding / (radii + e)
        elif self.projection is None:
            return self.embedding
        else:
            raise ValueError(f"projection={self.projection} not implemented.")
        
    def _compute_distance(self, metric: str, cluster_voxels: bool):
        """Return an (n, n) distance matrix for HDBSCAN when cluster_voxels=False."""
        if cluster_voxels:
            return None
        
        X = self.embedding.astype(np.float64) # shape (obs, latent_dims)
        if (self.projection == "sphere") and (metric == "cosine"):  # Cosine Similarity
            norm = np.linalg.norm(X, axis=0, keepdims=True) # norming across patients (within cols)
            sim = (X @ X.T) / (norm @ norm.T)
            d = 1.0 - sim
        elif self.projection == "sphere":                           # Haversine Distance
            cos_theta = np.clip(X @ X.T, -1.0, 1.0) # guard to ensure strictly in [-1,1]
            d = np.arccos(cos_theta) # Exit early 
        else:                                                       # Euclidean Distance     
            d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        return d
    
    def _run_hdbscan(self, min_cluster_size, cluster_voxels):
        '''Clusters using distance metrics calculated from the embedding'''
        metric = 'euclidean' if cluster_voxels else 'precomputed'
        arr = self.embedding if cluster_voxels else self.distances
        kwargs = {}
        kwargs['cluster_selection_method'] = 'eom'      # leaf or eom. eom is typical HDBSCAN, while leaf gets smaller homogeneous clusters. 
        kwargs['allow_single_cluster'] = False          # False is standard HDBSCAN operation 
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, **kwargs).fit(arr)
        return clusterer.labels_, clusterer.probabilities_, clusterer.cluster_persistence_
    
    def _get_opacity(self, override_probabilities):
        probabilities = override_probabilities if override_probabilities is not None else self.cluster_probabilities
        labels_norm = (self.cluster_labels - np.min(self.cluster_labels)) / (np.ptp(self.cluster_labels) + 1e-6)
        colors = cm.get_cmap('viridis')(labels_norm)
        colors[:, 3] = probabilities + ((1 - probabilities) / 2)
        rgba_colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})' for r, g, b, a in colors]
        return rgba_colors

    ### Public API ###
    def plot_embedding(self, *, point_size: int = 3, override_probabilities = None, verbose = True):
        """Interactive scatter of the embedding using Plotly. Points coloured by cluster label, opacity by probability of belonging to a cluster."""
        rgba_colors = self._get_opacity(override_probabilities)
        customdata = np.stack([self.cluster_labels, self.cluster_probabilities], axis=-1)
        if self.embedding.shape[1] == 2:
            fig = px.scatter(x=self.embedding[:, 0], y=self.embedding[:, 1], color=rgba_colors,
                             size=np.full(self.embedding.shape[0], point_size),
                             labels={"x": "UMAP 1", "y": "UMAP 2"})
        elif self.embedding.shape[1] == 3:
            fig = go.Figure(go.Scatter3d(
                x=self.embedding[:, 0], y=self.embedding[:, 1], z=self.embedding[:, 2],
                mode='markers', 
                marker=dict(size=point_size,color=rgba_colors),
                customdata=customdata,
                hovertemplate=(
                    'UMAP 1: %{x}<br>'
                    'UMAP 2: %{y}<br>'
                    'UMAP 3: %{z}<br>'
                    'Cluster: %{customdata[0]}<br>'
                    'Probability: %{customdata[1]:.2f}<br>'
                    '<extra></extra>'
                )
            ))
            fig.update_layout(scene=dict(xaxis_title="UMAP 1",
                                         yaxis_title="UMAP 2",
                                         zaxis_title="UMAP 3"))
        else:
            raise ValueError("Embedding must be 2D or 3D for plotting.")

        fig.update_layout(title="Interactive UMAP Embedding", template="plotly_white")
        if verbose: fig.show()
        return fig

    def run(self):
        '''Orchestrator method'''
        self.plot_embedding()