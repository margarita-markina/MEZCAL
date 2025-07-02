import os
import xarray as xr
import numpy as np

class CoupledModel:
    def __init__(self, model_name, institution, ensemble_member, scenario):
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

        self.f_tilde = None
        self.N_tilde = None

    def load_data(self, path_sst, path_amoc, lat_min, lat_max, lon_min, lon_max):
        """
        Load SST and AMOC time series for the model.
        Skips smoothing and mean removal.
        """
        filename_amoc = f"{path_amoc}amoc_{self.scenario}_{self.institution}_{self.model_name}_{self.ensemble_member}.nc"
        filename_sst = f"{path_sst}tos_Omon_{self.model_name}_{self.scenario}_{self.ensemble_member}.nc"

        if os.path.exists(filename_amoc) and os.path.exists(filename_sst):
            self.filepath_amoc = filename_amoc
            self.filepath_sst = filename_sst

            # Load SST
            sst = xr.open_dataset(filename_sst)['tos'].sel(
                lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
            )
            sst['year'] = xr.DataArray(np.arange(1, len(sst['year']) + 1), dims='year')

            # Load AMOC
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
        Apply a centered running mean and remove the time mean after smoothing.
        All metadata, coordinates, and attributes are preserved.
        """
        if self.sst_raw is None or self.amoc_raw is None:
            raise ValueError("Must load data before smoothing.")

        self.running_mean_window = running_mean_window

        # === Smooth SST ===
        sst_smoothed = (
            self.sst_raw.rolling(year=running_mean_window, center=True)
            .mean()
            .dropna(dim='year', how='all')
        )
        sst_smoothed = sst_smoothed - sst_smoothed.mean('year')
        sst_smoothed.attrs = self.sst_raw.attrs.copy()
        for coord in self.sst_raw.coords:
            sst_smoothed[coord].attrs = self.sst_raw[coord].attrs

        # === Smooth AMOC ===
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

    def train_regression(self):
        """
        Train regression weights from smoothed SST and AMOC data.
        Stores f_tilde and N_tilde as attributes.
        """
        if self.sst_smoothed is None or self.amoc_smoothed is None:
            raise ValueError("Must smooth data before training regression.")

        Y = self.sst_smoothed.stack(space=('lat', 'lon')).transpose('space', 'year').values
        x = self.amoc_smoothed.values

        xtx = np.dot(x, x)
        f_tilde = Y @ x / xtx
        N_tilde = Y - np.outer(f_tilde, x)

        self.f_tilde = f_tilde
        self.N_tilde = N_tilde


class CoupledModelRecon:
    def __init__(self, model_train: CoupledModel, model_truth: CoupledModel):
        self.model_train = model_train
        self.model_truth = model_truth

        self.x_tilde = None
        self.impact_map = None
        self.rmse = None

    def invert_for_x_with_impact(self, Cn_diag=None, Cx=None, return_impact_map=False, normalize=False):
        """
        Run inversion using trained model and compare to ground truth.
        """
        Y = self.model_truth.sst_smoothed
        f_tilde = xr.DataArray(self.model_train.f_tilde, coords=[Y.stack(z=('lat', 'lon')).coords['z']], dims='z')

        Y_valid = Y.stack(z=('lat', 'lon')).transpose('z', 'year')
        valid = ~np.isnan(f_tilde)
        f_valid = f_tilde.values[valid.values]
        Y_valid = Y_valid.sel(z=valid)

        F = f_valid#[:, None]
        m, T = Y_valid.shape

        if Cn is None and Cx is None:
            W_num = F @ np.linalg.pinv(F.T @ F)
        elif Cx is None:  # matrix inversion lemma
            W_num = F @ np.linalg.pinv(F.T @ F + Cn)
        elif Cn is None:
            raise ValueError("Prior cov without noise not supported.")
        else:
            W_num = Cx @ F @ np.linalg.pinv(F.T @ Cx @ F + Cn)
        print(Y)
        print(Cx.shape)
        print(F.shape)

        def invert_for_x_with_impact(self, Cn_diag=None, Cx=None, return_impact_map=False, normalize=False):
        Y = self.model_truth.sst_smoothed
        f_tilde = xr.DataArray(self.model_train.f_tilde, coords=[Y.stack(z=('lat', 'lon')).coords['z']], dims='z')

        Y_valid = Y.stack(z=('lat', 'lon')).transpose('z', 'year')
        valid = ~np.isnan(f_tilde)
        f_valid = f_tilde.values[valid.values]
        Y_valid = Y_valid.sel(z=valid)

        F = f_valid[:, None]  # shape (m, 1)
        Yvals = Y_valid.values  # shape (m, T)
        m, T = Yvals.shape
        Cn = np.diag(Cn_diag) if Cn_diag is not None else None

        if Cn is None and Cx is None:
            x_tilde = (F.T @ Yvals) / (F.T @ F)
            x_tilde = x_tilde.squeeze()
        elif Cx is None:
            x_tilde = (F.T @ np.linalg.inv(F @ F.T + Cn)) @ Yvals
            x_tilde = x_tilde.squeeze()
        elif Cn is None:
            raise ValueError("Prior covariance without noise not supported.")
        else:
            W_num = Cx @ F @ np.linalg.inv(F.T @ Cx @ F + Cn)
            x_tilde = (W_num @ Yvals).squeeze()

        self.x_tilde = x_tilde
        
        if return_impact_map:
            U, s, Vt = np.linalg.svd(Y_valid.values, full_matrices=False)
            projection = (U * np.sqrt(s)) @ W_num.T
            impact_flat = np.sum(projection**2, axis=1) / (T - 1)
            if normalize:
                impact_flat /= impact_flat.sum()

            impact_full = f_tilde.copy(deep=True)
            impact_full.values[:] = np.nan
            impact_full.values[valid.values] = impact_flat
            self.impact_map = impact_full

        true_x = self.model_truth.amoc_smoothed.values
        self.rmse = np.sqrt(np.mean((self.x_tilde - true_x)**2))

        if return_impact_map:
            return self.x_tilde, self.impact_map
        return self.x_tilde


class CoupledModelReconLibrary:
    def __init__(self):
        self.models = {}
        self.experiments = []

    def add_model(self, model: CoupledModel):
        key = model.model_name
        self.models[key] = model

    def add_experiment(self, experiment: CoupledModelRecon):
        self.experiments.append(experiment)

    def get_model(self, model_name):
        return self.models.get(model_name)

    def run_all_experiments(self, Cn=None, Cx=None, return_impact_map=False, normalize=False):
        for experiment in self.experiments:
            experiment.invert_for_x_with_impact(Cn, Cx, return_impact_map, normalize)

            
            # Leave-one-out cross validation