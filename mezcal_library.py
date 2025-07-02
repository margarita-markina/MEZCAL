# ---------------- Reconstruction Framework ----------------
#
# This package implements a flexible architecture for reconstructing
# climate indices (e.g., AMOC) from gridded fields (e.g., SST).
#
# Architecture overview:
# - BaseReconstruction: abstract base class for all reconstruction methods.
# - ZeroLagRegression, RidgeZeroLagRegression: specific reconstruction models.
# - CoupledModel: handles loading and smoothing of climate model data.
# - ReconstructionDataset: container for predictors/target from one or more models.
# - CoupledModelRecon: executes the reconstruction with optional impact diagnostics.
# - CoupledModelReconLibrary: manages collections of models and experiments.
#
# To implement a new reconstruction model:
# - Create a subclass of BaseReconstruction and implement `train` and `predict`.
#
# To add new predictor fields:
# - Add the fields to CoupledModel and ReconstructionDataset.
# - Update `build_predictor_matrix` to include the new predictors.

import os
import xarray as xr
import numpy as np

# ---------------- Base class for reconstruction ----------------
class BaseReconstruction:
    def train(self, Y, x):
        """
        Train the model using predictors and target.

        Parameters:
        Y : ndarray of shape (n_features, n_time)
            Predictor matrix (e.g., SSTs flattened over space).
        x : ndarray of shape (n_time,)
            Target time series (e.g., AMOC).
        """
        raise NotImplementedError

    def predict(self, Y, true_x=None):
        """
        Reconstruct target from predictors.

        Parameters:
        Y : ndarray of shape (n_features, n_time)
            Predictor matrix.
        true_x : ndarray or None
            If provided, compute RMSE against this ground truth.

        Returns:
        dict with keys:
            - 'x_tilde': reconstructed signal
            - 'rmse': RMSE if true_x is given
        """
        raise NotImplementedError

# ---------------- Shared utility for impact calculation ----------------
def compute_impact(Y, f_tilde, normalize=False):
    """
    Compute impact map from regression weights and predictors.

    Parameters:
    Y : ndarray (n_features, n_time)
        Predictor matrix.
    f_tilde : ndarray (n_features,)
        Regression weights.
    normalize : bool
        If True, normalize impact values to sum to 1.

    Returns:
    impact_flat : ndarray (n_features,)
        Estimated local contributions to variance.
    """
    F = f_tilde[:, None]
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    projection = (U * np.sqrt(s)) @ F
    impact_flat = np.sum(projection**2, axis=1) / (Y.shape[1] - 1)

    if normalize:
        impact_flat /= impact_flat.sum()

    return impact_flat

# ---------------- Specific implementation: zero-lag regression ----------------
class ZeroLagRegression(BaseReconstruction):
    def __init__(self):
        self.f_tilde = None

    def train(self, Y, x):
        """Standard least squares regression."""
        xtx = np.dot(x, x)
        self.f_tilde = Y @ x / xtx

    def predict(self, Y, true_x=None):
        x_tilde = self.f_tilde @ Y
        results = {'x_tilde': x_tilde.squeeze()}

        if true_x is not None:
            results['rmse'] = np.sqrt(np.mean((x_tilde.squeeze() - true_x)**2))

        return results

    def predict_with_impact(self, Y, true_x=None, normalize=False):
        result = self.predict(Y, true_x)
        result['impact_flat'] = compute_impact(Y, self.f_tilde, normalize)
        return result

    def get_weights(self):
        return self.f_tilde

# ---------------- Regularized regression variants ----------------
class RidgeZeroLagRegression(BaseReconstruction):
    def __init__(self, lam=0.1):
        self.lam = lam
        self.f_tilde = None

    def train(self, Y, x):
        xtx = np.dot(x, x) + self.lam
        self.f_tilde = Y @ x / xtx

    def predict(self, Y, true_x=None):
        x_tilde = self.f_tilde @ Y
        results = {'x_tilde': x_tilde.squeeze()}

        if true_x is not None:
            results['rmse'] = np.sqrt(np.mean((x_tilde.squeeze() - true_x)**2))

        return results

    def predict_with_impact(self, Y, true_x=None, normalize=False):
        result = self.predict(Y, true_x)
        result['impact_flat'] = compute_impact(Y, self.f_tilde, normalize)
        return result

    def get_weights(self):
        return self.f_tilde

# ---------------- Data container for reconstruction ----------------
class ReconstructionDataset:
    def __init__(self, name, predictors, target, years_train, years_test, years_eval,
                 is_multimodel=False, left_out_model=None):
        """
        Parameters:
        name : str
            Identifier for dataset.
        predictors : dict of str -> ndarray
            Mapping of variable name to predictor arrays (shape: [space, time]).
        target : ndarray
            Target time series (e.g., AMOC).
        years_train, years_test, years_eval : list of int
            Year indices for train/test/eval.
        is_multimodel : bool
            Flag for multi-model experiments.
        left_out_model : str or None
            If leave-one-out, name of excluded model.
        """
        self.name = name
        self.predictors = predictors
        self.target = target
        self.years_train = years_train
        self.years_test = years_test
        self.years_eval = years_eval
        self.is_multimodel = is_multimodel
        self.left_out_model = left_out_model

    def build_predictor_matrix(self, keys=None):
        """
        Combine selected predictors into a single predictor matrix.

        Parameters:
        keys : list of str or None
            Names of predictors to include (default: all).

        Returns:
        ndarray of shape (n_features, n_time)
        """
        if keys is None:
            keys = self.predictors.keys()
        return np.concatenate([self.predictors[k] for k in keys], axis=0)

# ---------------- Coupled model for managing SST and AMOC ----------------
class CoupledModel:
    def __init__(self, model_name, institution, ensemble_member, scenario):
        """
        Represents a single GCM configuration.

        Parameters:
        model_name : str
        institution : str
        ensemble_member : str
        scenario : str
        """
        self.model_name = model_name
        self.institution = institution
        self.ensemble_member = ensemble_member
        self.scenario = scenario

        self.filepath_amoc = None
        self.filepath_sst = None

        self.sst_raw = None
        self.amoc_raw = None

        self.sst_smoothed = None
        self.amoc_smoothed = None

        self.running_mean_window = None

    def load_data(self, path_sst, path_amoc, lat_min, lat_max, lon_min, lon_max):
        """
        Load SST and AMOC NetCDF files and clip to desired region.
        """
        filename_amoc = f"{path_amoc}amoc_{self.scenario}_{self.institution}_{self.model_name}_{self.ensemble_member}.nc"
        filename_sst = f"{path_sst}tos_Omon_{self.model_name}_{self.scenario}_{self.ensemble_member}.nc"

        if os.path.exists(filename_amoc) and os.path.exists(filename_sst):
            self.filepath_amoc = filename_amoc
            self.filepath_sst = filename_sst

            sst = xr.open_dataset(filename_sst)['tos'].sel(
                lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
            )
            sst['year'] = xr.DataArray(np.arange(1, len(sst['year']) + 1), dims='year')

            amoc = xr.open_dataset(filename_amoc)['amoc']
            if self.model_name == 'IPSL-CM6A-LR':
                amoc = amoc.isel(year=slice(0, len(sst)), x=0)
            else:
                amoc = amoc.isel(year=slice(0, len(sst)))

            amoc['year'] = xr.DataArray(np.arange(1, len(amoc['year']) + 1), dims='year')

            self.sst_raw = sst
            self.amoc_raw = amoc

    def smooth_data(self, running_mean_window):
        """
        Apply centered running mean and remove temporal mean.
        """
        if self.sst_raw is None or self.amoc_raw is None:
            raise ValueError("Must load data before smoothing.")

        self.running_mean_window = running_mean_window

        sst_smoothed = (
            self.sst_raw.rolling(year=running_mean_window, center=True)
            .mean()
            .dropna(dim='year', how='all')
        )
        sst_smoothed = sst_smoothed - sst_smoothed.mean('year')
        sst_smoothed.attrs = self.sst_raw.attrs.copy()
        for coord in self.sst_raw.coords:
            sst_smoothed[coord].attrs = self.sst_raw[coord].attrs

        amoc_smoothed = (
            self.amoc_raw.rolling(year=running_mean_window, center=True)
            .mean()
            .dropna(dim='year', how='all')
        )
        amoc_smoothed = amoc_smoothed - amoc_smoothed.mean('year')
        amoc_smoothed.attrs = self.amoc_raw.attrs.copy()
        for coord in self.amoc_raw.coords:
            amoc_smoothed[coord].attrs = self.amoc_raw[coord].attrs

        self.sst_smoothed = sst_smoothed
        self.amoc_smoothed = amoc_smoothed

    def subset_years(self, num_years):
        """
        Truncate data to first `num_years`.
        """
        if self.sst_smoothed is None or self.amoc_smoothed is None:
            raise ValueError("Must smooth data before subsetting.")

        if num_years >= len(self.sst_smoothed.year):
            return

        self.sst_smoothed = self.sst_smoothed.isel(year=slice(0, num_years))
        self.amoc_smoothed = self.amoc_smoothed.isel(year=slice(0, num_years))

# ---------------- Coupled model recon using given reconstruction class ----------------
class CoupledModelRecon:
    def __init__(self, model_train: CoupledModel, model_truth: CoupledModel, recon_model: BaseReconstruction):
        """
        Executes the reconstruction given a model to train on, a model to predict on,
        and a reconstruction method.
        """
        self.model_train = model_train
        self.model_truth = model_truth
        self.recon_model = recon_model

        self.x_tilde = None
        self.impact_map = None
        self.rmse = None

    def run(self, return_impact_map=False, normalize=False):
        """
        Perform reconstruction and optionally compute impact map.

        Returns:
        x_tilde : ndarray
            Reconstructed AMOC.
        impact_map : xarray.DataArray, optional
            Spatial map of impact if return_impact_map is True.
        """
        Y = self.model_truth.sst_smoothed
        x = self.model_train.amoc_smoothed.values

        Y_stack = Y.stack(space=('lat', 'lon')).transpose('space', 'year')
        valid = ~np.isnan(Y_stack.isel(year=0))

        Y_valid = Y_stack.sel(space=valid)
        Y_vals = Y_valid.values

        self.recon_model.train(Y_vals, x)

        if return_impact_map:
            result = self.recon_model.predict_with_impact(Y_vals, true_x=self.model_truth.amoc_smoothed.values, normalize=normalize)
        else:
            result = self.recon_model.predict(Y_vals, true_x=self.model_truth.amoc_smoothed.values)

        self.x_tilde = result['x_tilde']
        self.rmse = result.get('rmse', None)

        if return_impact_map and 'impact_flat' in result:
            impact_full = xr.DataArray(np.full(Y_stack.shape[0], np.nan), coords=[Y_stack.coords['space']], dims='space')
            impact_full.loc[valid] = result['impact_flat']
            self.impact_map = impact_full.unstack('space')

        if return_impact_map:
            return self.x_tilde, self.impact_map
        return self.x_tilde

# ---------------- Model and experiment management ----------------
class CoupledModelReconLibrary:
    def __init__(self):
        """
        Container for climate models and experiments.
        """
        self.models = {}
        self.experiments = []

    def add_model(self, model: CoupledModel):
        self.models[model.model_name] = model

    def add_experiment(self, experiment: CoupledModelRecon):
        self.experiments.append(experiment)

    def get_model(self, model_name):
        return self.models.get(model_name)

    def run_all_experiments(self, return_impact_map=False, normalize=False):
        for experiment in self.experiments:
            experiment.run(return_impact_map, normalize)

# ---------------- Calibration and testing window utilities ----------------
def partition_years(years, calibration_frac=0.6, validation_frac=0.2):
    """
    Split list of years into calibration, validation, and test sets.

    Parameters:
    years : list of int
        Sequence of year indices.
    calibration_frac : float
        Fraction of years used for calibration.
    validation_frac : float
        Fraction of years used for validation.

    Returns:
    tuple of (calibration_years, validation_years, testing_years)
    """
    n = len(years)
    n_cal = int(n * calibration_frac)
    n_val = int(n * validation_frac)
    n_test = n - n_cal - n_val

    calibration_years = years[:n_cal]
    validation_years = years[n_cal:n_cal+n_val]
    testing_years = years[n_cal+n_val:]

    return calibration_years, validation_years, testing_years
