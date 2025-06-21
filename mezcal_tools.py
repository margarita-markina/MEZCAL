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

def train_regression_and_reconstruct(sst_normalized, amoc, amoc_std):
    """
    Train regression model: compute regression weights `f` and 
    reconstruct AMOC from normalized SST using projection.
    """
    f = sst_normalized.T @ amoc / sst_normalized.shape[0]
    amoc_reconstructed = sst_normalized @ f
    # Rescale reconstructed AMOC to match original std
    amoc_reconstructed = amoc_reconstructed * amoc_std / np.std(amoc_reconstructed)
    
    #f = ss
    
    return f, amoc_reconstructed


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
