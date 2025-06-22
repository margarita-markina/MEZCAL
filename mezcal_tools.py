import os
import random
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# === DATA PREPARATION ===

def prepare_and_normalize_sst(sst):
    """
    Normalize SST by spatial point after flattening (lat, lon).
    NaNs are replaced with zero before normalization.
    """
    sst_selected = sst.fillna(0).stack(spatial=("lat", "lon")).values
    sst_mean = np.mean(sst_selected, axis=0)
    sst_std = np.std(sst_selected, axis=0)
    sst_std[sst_std == 0] = 1  # Prevent divide-by-zero
    return (sst_selected - sst_mean) / sst_std


def prepare_and_normalize_amoc(amoc):
    """
    Normalize AMOC time series after filling NaNs with 0.
    Returns original, normalized, mean, and std.
    """
    amoc_selected = amoc.fillna(0).values
    amoc_mean = np.mean(amoc_selected)
    amoc_std = np.std(amoc_selected)
    amoc_normalized = (amoc_selected - amoc_mean) / amoc_std
    return amoc_selected, amoc_normalized, amoc_mean, amoc_std


# === REGRESSION & RECONSTRUCTION ===

def train_reg(Y, x):
    """
    Solve Y = f x.T + N where f x.T is an outer product.
    
    Parameters:
    -----------
    Y : ndarray of shape (m, T)
        Observations (e.g. SST patterns)
    x : ndarray of shape (T,)
        Time series (e.g. AMOC index)

    Returns:
    --------
    f_hat : ndarray of shape (m,)
        Regression weights
    """
    # Solve least squares: f = Y x / (x^T x)
    xtx = np.dot(x, x)
    f_hat = Y @ x / xtx

    # NB residuals are given by:
    N_tilde = Y - np.outer(f_hat, x)

    return f_hat, N_tilde



# === OUTPUT ===

def save_regression_map_sst_amoc_to_file(
    f, ny, nx, years, sst_for_coords, sst_norm, amoc_norm, amoc_std, amoc_recon,
    model_name, path_output, smoothing_period=20
):
    """
    Save regression weights, normalized SST, and AMOC timeseries to NetCDF.
    """
    fr = np.reshape(f, (ny, nx))
    fr_masked = np.where(fr == 0, np.nan, fr)

    lat, lon = sst_for_coords["lat"], sst_for_coords["lon"]

    ds = xr.Dataset({
        "fr": xr.DataArray(fr_masked, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"]),
        "normalized_sst": xr.DataArray(
            sst_norm.reshape(len(years), ny, nx),
            coords={"year": years, "lat": lat, "lon": lon},
            dims=["year", "lat", "lon"]
        ),
        "amoc_normalised": xr.DataArray(amoc_norm, coords={"year": years}, dims=["year"]),
        "amoc_reconstructed": xr.DataArray(amoc_recon, coords={"year": years}, dims=["year"]),
        "amoc_std": xr.DataArray(amoc_std)
    })

    filename = f"{model_name}_{len(years)}yrs.nc"
    filepath = os.path.join(path_output, filename)
    ds.to_netcdf(filepath)
    print(f"Saved to {filepath}")


# === PLOTTING ===

def plot_regression_coefficients(f, ny, nx, title, model_names):
    """
    Plot regression weights reshaped to (ny, nx).
    """
    fr_masked = np.where(np.reshape(f, (ny, nx)) == 0, np.nan, np.reshape(f, (ny, nx)))
    if model_names[0] == 'MPI-ESM1-2-HR':
        fr_masked = np.flipud(fr_masked)

    plt.figure(figsize=(8, 6))
    plt.contourf(fr_masked, cmap="viridis", levels=20)
    plt.colorbar(label="Regression Coefficients")
    plt.title(title)
    plt.show()


def plot_amoc_comparison(amoc_original, amoc_reconstructed, years, title, smoothing_period=20):
    """
    Compare original and reconstructed AMOC time series.
    """
    da_orig = xr.DataArray(amoc_original, dims="year", coords={"year": years})
    da_recon = xr.DataArray(amoc_reconstructed, dims="year", coords={"year": years})

    plt.figure(figsize=(8, 5))
    da_orig.plot(label="Original AMOC", linewidth=2)
    da_recon.plot(label="Reconstructed AMOC", linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("AMOC [Sv]")
    plt.legend()
    plt.show()


# === PIPELINE ===

def compute_regression_from_random_years(
    model_names,
    amoc_institution_names,
    ens_members_appendices,
    scenario,
    PATH_SST,
    PATH_AMOC,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    running_mean_window,
    tag,
    PATH_saved_regressions,
    plot=True,
    random_seed=None
):
    """
    Compute regression of AMOC onto SST using 10 randomly selected years per model.
    """

    if random_seed is not None:
        random.seed(random_seed)

    sst_random_samples, amoc_random_samples = [], []

    for i, model in enumerate(model_names):
        print(f"Processing: {model}")

        f_amoc = f"{PATH_AMOC}amoc_{scenario}_{amoc_institution_names[i]}_{model}_{ens_members_appendices[i]}.nc"
        f_sst = f"{PATH_SST}tos_Omon_{model}_{scenario}_{ens_members_appendices[i]}.nc"

        sst = xr.open_dataset(f_sst)['tos'].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        sst['year'] = np.arange(sst.sizes['year'])

        sst_mean = sst.mean('year')
        sst_smoothed = (sst - sst_mean).rolling(year=running_mean_window, center=True).mean().dropna('year', how='all')

        amoc = xr.open_dataset(f_amoc)['amoc']
        amoc = amoc.isel(x=0) if model == 'IPSL-CM6A-LR' else amoc
        amoc = amoc.isel(year=slice(0, sst.sizes['year']))
        amoc = (amoc - amoc.mean('year')).rolling(year=running_mean_window, center=True).mean().dropna('year', how='all')

        valid_timesteps = sst_smoothed.sizes['year']
        indices = sorted(random.sample(range(valid_timesteps), k=10))

        sst_sample = sst_smoothed.isel(year=indices)
        amoc_sample = amoc.isel(year=indices)

        for var in ['sector', 'basin', 'region']:
            sst_sample = sst_sample.drop_vars(var, errors='ignore')
            amoc_sample = amoc_sample.drop_vars(var, errors='ignore')
            if var in amoc_sample.coords:
                amoc_sample = amoc_sample.reset_coords(var, drop=True)

        sst_random_samples.append(sst_sample)
        amoc_random_samples.append(amoc_sample)

    # Concatenate across models
    sst_combined = xr.concat(sst_random_samples, dim='year')
    amoc_combined = xr.concat(amoc_random_samples, dim='year')

    ny, nx = sst_combined.sizes['lat'], sst_combined.sizes['lon']
    lat_vals, lon_vals = sst_combined.lat.values, sst_combined.lon.values
    years = np.arange(sst_combined.sizes['year'])

    sst_all = xr.DataArray(sst_combined, dims=["year", "lat", "lon"], coords={"year": years, "lat": lat_vals, "lon": lon_vals})
    amoc_all = xr.DataArray(amoc_combined, dims=["year"], coords={"year": years})

    # Train model
    sst_normalized = prepare_and_normalize_sst(sst_all)
    amoc, amoc_norm, amoc_mean, amoc_std = prepare_and_normalize_amoc(amoc_all)
    f, amoc_reconstructed = train_regression_and_reconstruct(sst_normalized, amoc_norm, amoc_std)

    if plot:
        save_regression_map_sst_amoc_to_file(
            f, ny, nx, years, sst_all, sst_normalized, amoc_norm, amoc_std, amoc_reconstructed,
            tag, PATH_saved_regressions, smoothing_period=running_mean_window
        )
        plot_regression_coefficients(f, ny, nx, f"Regression Coefficients (f) ({tag})", model_names)
        plot_amoc_comparison(amoc, amoc_reconstructed, years, f"Original vs Reconstructed AMOC ({tag})", smoothing_period=running_mean_window)

    return f, amoc_reconstructed, amoc, sst_all, amoc_all


def load_sst_and_amoc_timeseries(model_names, amoc_institution_names, ens_members_appendices, scenario, PATH_SST, PATH_AMOC, lat_min, lat_max, lon_min, lon_max):
    """
    Load SST and AMOC time series for each model and return a dictionary indexed by model name.
    Skips smoothing and mean removal.
    """
    timeseries_by_model = {}

    for i, model in enumerate(model_names):
        filename_amoc = f"{PATH_AMOC}amoc_{scenario}_{amoc_institution_names[i]}_{model}_{ens_members_appendices[i]}.nc"
        filename_sst = f"{PATH_SST}tos_Omon_{model}_{scenario}_{ens_members_appendices[i]}.nc"

        if os.path.exists(filename_amoc) and os.path.exists(filename_sst):
            # Load SST
            sst_time_series = xr.open_dataset(filename_sst)['tos'].sel(
                lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
            )
            sst_time_series['year'] = xr.DataArray(
                np.arange(1, len(sst_time_series['year']) + 1), dims='year'
            )

            # Load AMOC
            amoc_raw = xr.open_dataset(filename_amoc)['amoc']
            if model == 'IPSL-CM6A-LR':
                amoc_time_series = amoc_raw.isel(year=slice(0, len(sst_time_series)), x=0)
            else:
                amoc_time_series = amoc_raw.isel(year=slice(0, len(sst_time_series)))

            amoc_time_series['year'] = xr.DataArray(
                np.arange(1, len(amoc_time_series['year']) + 1), dims='year'
            )

            # Store in dictionary
            timeseries_by_model[model] = {
                'sst': sst_time_series,
                'amoc': amoc_time_series
            }

    return timeseries_by_model


def smooth_sst_and_amoc_timeseries(timeseries_by_model, running_mean_window):
    """
    Apply a centered running mean and remove the time mean after smoothing.
    Input and output are dictionaries indexed by model names with 'sst' and 'amoc' keys.
    All metadata, coordinates, and attributes are preserved.
    """
    smoothed_by_model = {}

    for model, ts_dict in timeseries_by_model.items():
        sst = ts_dict['sst']
        amoc = ts_dict['amoc']

        # === Smooth SST ===
        sst_smoothed = (
            sst.rolling(year=running_mean_window, center=True)
            .mean()
            .dropna(dim='year', how='all')
        )
        sst_smoothed = sst_smoothed - sst_smoothed.mean('year')
        sst_smoothed['year'] = sst_smoothed['year']  # preserve original coords

        # Preserve attributes
        sst_smoothed.attrs = sst.attrs.copy()
        for coord in sst.coords:
            sst_smoothed[coord].attrs = sst[coord].attrs

        # === Smooth AMOC ===
        amoc_smoothed = (
            amoc.rolling(year=running_mean_window, center=True)
            .mean()
            .dropna(dim='year', how='all')
        )
        amoc_smoothed = amoc_smoothed - amoc_smoothed.mean('year')
        amoc_smoothed['year'] = amoc_smoothed['year']  # preserve original coords

        # Preserve attributes
        amoc_smoothed.attrs = amoc.attrs.copy()
        for coord in amoc.coords:
            amoc_smoothed[coord].attrs = amoc[coord].attrs

        smoothed_by_model[model] = {
            'sst': sst_smoothed,
            'amoc': amoc_smoothed
        }

    return smoothed_by_model


def invert_for_x(Y, f_tilde, Cn=None, Cx=None):
    """
    Solve Y = f_tilde x.T + N for x using either priors or least-squares.

    Parameters:
    -----------
    Y : xr.DataArray, shape (lat, lon, time)
        Observed spatiotemporal field
    f_tilde : xr.DataArray, shape (lat, lon)
        Spatial pattern (regression weights)
    Cn : np.ndarray or None, shape (m, m)
        Prior covariance of noise N in space (optional)
    Cx : np.ndarray or None, shape (T, T)
        Prior covariance of x (optional)

    Returns:
    --------
    x_hat : xr.DataArray, shape (time,)
        Reconstructed time series
    """
    import numpy as np
    import xarray as xr

    # Flatten spatial dims
    Y_flat = Y.stack(space=('lat', 'lon'))  # shape (space, time)
    f_flat = f_tilde.stack(space=('lat', 'lon'))  # shape (space,)

    # Mask invalid values
    valid = ~np.isnan(f_flat)
    f_valid = f_flat[valid].values  # shape (m_valid,)
    Y_valid = Y_flat.sel(space=valid).values  # shape (m_valid, time)

    if Cn is not None and Cx is not None:
        # Use full Bayesian estimator with priors and matrix inversion lemma
        Cx_f = Cx @ f_valid              # shape (T,)
        denom = f_valid @ Cx_f + Cn[np.ix_(valid.values, valid.values)]
        denom_inv = np.linalg.inv(denom)
        x_hat = Cx_f @ denom_inv @ Y_valid
    else:
        # Use uninformative prior (least squares projection)
        f_norm2 = np.dot(f_valid, f_valid)
        x_hat = f_valid @ Y_valid / f_norm2  # shape (time,)

    return xr.DataArray(x_hat, dims=['time'], coords={'time': Y['time']})


def invert_for_x_with_impact(Y, f_tilde, Cn=None, Cx=None, return_impact_map=False, normalize=False):
    """
    Solve Y = f_tilde x.T + N for x, with optional prior covariances.
    Optionally compute and return an impact map using the inverse operator.

    Parameters:
    -----------
    Y : xr.DataArray (m x T)
        Observed SST (space x time)
    f_tilde : xr.DataArray (m,)
        Regression weights (flattened space)
    Cn : ndarray or None
        Prior covariance of N (m x m). If None, assumes identity.
    Cx : ndarray or None
        Prior covariance of x (T x T). If None, assumes identity.
    return_impact_map : bool
        Whether to return the impact map
    normalize : bool
        Whether to normalize the impact map

    Returns:
    --------
    x_tilde : ndarray (T,)
        Inferred latent time series
    (optional) impact : xr.DataArray (same shape as spatial domain of f_tilde)
    """
    Y_valid = Y.stack(z=('lat', 'lon')).transpose('z', 'year')
    valid = ~np.isnan(f_tilde)
    f_valid = f_tilde.values[valid.values]
    Y_valid = Y_valid.sel(z=valid)

    F = f_valid[:, None]  # shape (m_valid, 1)

    # Prior covariances
    m, T = Y_valid.shape
    Cn = np.eye(m) if Cn is None else Cn[np.ix_(valid.values, valid.values)]
    Cx = np.eye(T) if Cx is None else Cx

    # Matrix inversion lemma form
    Cn_inv = np.linalg.inv(Cn)
    Cx_inv = np.linalg.inv(Cx)

    # Weighting matrices
    W_num = Cx @ F.T @ np.linalg.inv(F @ Cx @ F.T + Cn)
    x_tilde = (W_num @ Y_valid.values).squeeze()

    if return_impact_map:
        # Impact map via SVD of Y_valid
        U, s, Vt = np.linalg.svd(Y_valid.values, full_matrices=False)
        projection = (U * np.sqrt(s)) @ W_num.T
        impact_flat = np.sum(projection**2, axis=1) / (T - 1)
        if normalize:
            impact_flat /= impact_flat.sum()

        impact_full = f_tilde.copy(deep=True)
        impact_full.values[:] = np.nan
        impact_full.values[valid.values] = impact_flat

        return x_tilde, impact_full

    return x_tilde
