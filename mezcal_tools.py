import numpy as np
import xarray as xr
import os
import random

from scipy.stats import linregress

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_and_normalize_sst(sst):
    """
    Prepare SST by subsetting to North Atlantic, filling NaNs, stacking spatial points, and normalizing.
    """
    sst_selected = sst.fillna(0).stack(spatial=("lat", "lon")).values

    # Normalize SST per spatial point
    sst_mean = np.mean(sst_selected, axis=0)
    sst_std = np.std(sst_selected, axis=0)
    sst_std[sst_std == 0] = 1  # Avoid division by zero
    return (sst_selected - sst_mean) / sst_std


def prepare_and_normalize_amoc(amoc):
    """
    Prepare AMOC by subsetting, filling NaNs, and normalizing.
    """
    amoc_selected = amoc.fillna(0).values
    amoc_mean = np.mean(amoc_selected)
    amoc_std = np.std(amoc_selected)
    return amoc_selected, (amoc_selected - amoc_mean) / amoc_std, amoc_mean, amoc_std

def train_regression_and_reconstruct(sst_normalized, amoc, amoc_std):
    """
    Train regression (compute f) and reconstruct AMOC using SST.
    """
    f = sst_normalized.T @ amoc / sst_normalized.shape[0]
    amoc_reconstructed = sst_normalized @ f
    amoc_reconstructed = amoc_reconstructed * amoc_std / np.std(amoc_reconstructed)  # Rescale
    return f, amoc_reconstructed

def save_regression_map_sst_amoc_to_file(f, ny, nx, years, sst_for_coords, sst_norm, amoc_norm, amoc_std, amoc_recon, model_name, path_output, smoothing_period = 20):
    """
    Saves regression map to file, including:
    - fr (masked regression coefficients)
    - normalized SST (same shape as fr)
    - AMOC time series (1D, yearly values)
    """

    # Reshape `f` to (ny, nx) and mask land areas (0 → NaN)
    fr = np.reshape(f, [ny, nx])
    fr_masked = np.where(fr == 0, np.nan, fr)

    # Extract coordinates
    lat = sst_for_coords["lat"]  # (j, i)
    lon = sst_for_coords["lon"]  # (j, i)

    # Create DataArray for `fr`
    fr_dataarray = xr.DataArray(
        fr_masked, 
        coords={"lat": lat, "lon": lon}, 
        dims=["lat", "lon"], 
        name="fr"
    )

    # === Normalized SST  ===

    sst_masked = np.where(sst_norm == 0, np.nan, sst_norm)

    sst_reshaped = sst_masked.reshape(len(years), ny, nx)
        
    sst_dataarray = xr.DataArray(
        sst_reshaped,
        # coords={"year": np.arange(1 + smoothing_period/2, 251 - smoothing_period/2 + 1), "lat": lat, "lon": lon},
        coords={"year": years, "lat": lat, "lon": lon},
        dims=["year", "lat", "lon"],
        name="sst_normalized"
    )

    # === AMOC Time Series (Yearly Data from period_start to period_end) ===
    # years = np.arange(1 + smoothing_period/2, 251 - smoothing_period/2 + 1)
    assert len(amoc_norm) == len(years), "AMOC (normalised) time series length mismatch!"

    amoc_norm_dataarray = xr.DataArray(
        amoc_norm, 
        coords={"year": years}, 
        dims=["year"], 
        name="amoc"
    )

    amoc_std_dataarray = xr.DataArray(
        amoc_std, 
        name="amoc"
    )

    assert len(amoc_recon) == len(years), "AMOC (reconstructed) time series length mismatch!"

    amoc_recon_dataarray = xr.DataArray(
        amoc_recon, 
        coords={"year": years}, 
        dims=["year"], 
        name="amoc"
    )
    
    # === Merge into a dataset and save ===
    dataset = xr.Dataset({"fr": fr_dataarray, "normalized_sst": sst_dataarray, "amoc_normalised": amoc_norm_dataarray, "amoc_std": amoc_std_dataarray, "amoc_reconstructed": amoc_recon_dataarray})

    # Define output filename
    output_filename = f"{model_name}_{len(years)}yrs.nc"

    # Save to NetCDF
    dataset.to_netcdf(path_output + output_filename)
    print(f"Saved to {path_output + output_filename}")


def plot_regression_coefficients(f, ny, nx, title, model_names):
    """
    Reshape and plot regression coefficients with land masked as NaN.
    """
    fr = np.reshape(f, [ny, nx])
    fr_masked = np.where(fr == 0, np.nan, fr)
    plt.figure(figsize=(8, 6))

    if model_names[0]=='MPI-ESM1-2-HR':
        fr_masked = np.flipud(fr_masked)
        
    plt.contourf(fr_masked, cmap="viridis", levels=20)
    plt.colorbar(label="Regression Coefficients")
    plt.title(title)
    plt.show()

def plot_amoc_comparison(amoc_original, amoc_reconstructed, years, title, smoothing_period = 20):
    """
    Plot original vs reconstructed AMOC.
    """
    # years = np.arange(1 + smoothing_period/2, 251 - smoothing_period/2 + 1)
    amoc_original_da = xr.DataArray(amoc_original, dims='year', coords={'year': years})
    amoc_reconstructed_da = xr.DataArray(amoc_reconstructed, dims='year', coords={'year': years})
    
    plt.figure(figsize=(8, 5))
    amoc_original_da.plot(label="Original AMOC", linewidth=2)
    amoc_reconstructed_da.plot(label="Reconstructed AMOC", linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("AMOC [Sv]")
    plt.legend()
    plt.show()
    
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

    if random_seed is not None:
        random.seed(random_seed)

    sst_random_samples = []
    amoc_random_samples = []

    for i, model in enumerate(model_names):
        print(f"Processing: {model}")

        filename_amoc = f"{PATH_AMOC}amoc_{scenario}_{amoc_institution_names[i]}_{model}_{ens_members_appendices[i]}.nc"
        filename_sst = f"{PATH_SST}tos_Omon_{model}_{scenario}_{ens_members_appendices[i]}.nc"

        sst_ds = xr.open_dataset(filename_sst)['tos'].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        sst_ds['year'] = np.arange(sst_ds.sizes['year'])

        sst_mean = sst_ds.mean('year')
        sst_smoothed = (sst_ds - sst_mean).rolling(year=running_mean_window, center=True).mean().dropna(dim='year', how='all')

        amoc_ds = xr.open_dataset(filename_amoc)['amoc']
        amoc_series = amoc_ds.isel(x=0) if model == 'IPSL-CM6A-LR' else amoc_ds
        amoc_series = amoc_series.isel(year=slice(0, sst_ds.sizes['year']))
        amoc_mean = amoc_series.mean('year')
        amoc_smoothed = (amoc_series - amoc_mean).rolling(year=running_mean_window, center=True).mean().dropna(dim='year', how='all')

        valid_timesteps = sst_smoothed.year.size
        indices = sorted(random.sample(range(valid_timesteps), k=10))

        sst_sample = sst_smoothed.isel(year=indices)
        amoc_sample = amoc_smoothed.isel(year=indices)

        for var in ['sector', 'basin', 'region']:
            sst_sample = sst_sample.drop_vars(var, errors='ignore')
            if var in sst_sample.coords:
                sst_sample = sst_sample.reset_coords(var, drop=True)

            amoc_sample = amoc_sample.drop_vars(var, errors='ignore')
            if var in amoc_sample.coords:
                amoc_sample = amoc_sample.reset_coords(var, drop=True)
                if var == 'sector':
                    print('amoc sector is present')

        sst_random_samples.append(sst_sample)
        amoc_random_samples.append(amoc_sample)

    # Stack together
    sst_combined = xr.concat(sst_random_samples, dim='year')
    amoc_combined = xr.concat(amoc_random_samples, dim='year')
    ny, nx = sst_combined.sizes['lat'], sst_combined.sizes['lon']

    lat_vals = sst_random_samples[0].lat.values
    lon_vals = sst_random_samples[0].lon.values

    sst_all = xr.DataArray(
        sst_combined,
        dims=["year", "lat", "lon"],
        coords={"year": np.arange(sst_combined.shape[0]), "lat": lat_vals, "lon": lon_vals},
        name="sst"
    )

    amoc_all = xr.DataArray(
        amoc_combined,
        dims=["year"],
        coords={"year": np.arange(amoc_combined.shape[0])},
        name="amoc"
    )

    # Normalize and regress
    sst_normalized = prepare_and_normalize_sst(sst_all)
    amoc, amoc_normalized, amoc_mean, amoc_std = prepare_and_normalize_amoc(amoc_all)
    f, amoc_reconstructed = train_regression_and_reconstruct(sst_normalized, amoc_normalized, amoc_std)

    if plot:
        years = sst_all.year.values
        save_regression_map_sst_amoc_to_file(
            f, ny, nx, years, sst_all, sst_normalized, amoc_normalized, amoc_std, amoc_reconstructed,
            tag, PATH_saved_regressions, smoothing_period=running_mean_window
        )

        plot_regression_coefficients(f, ny, nx, f"Regression Coefficients (f) ({tag})", model_names)
        plot_amoc_comparison(amoc, amoc_reconstructed, years, f"Original vs Reconstructed AMOC ({tag})", smoothing_period=running_mean_window)

    return f, amoc_reconstructed, amoc, sst_all, amoc_all
