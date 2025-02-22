# Import packages
import os
import pathlib
import pickle
import re
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import earthaccess
import earthpy as et
import geopandas as gpd
import geoviews as gv
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray as rxr
import rioxarray.merge as rxrmerge
from tqdm.notebook import tqdm
import xarray as xr
import zipfile
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"

warnings.simplefilter("ignore")

# Create data directry in the home folder
data_dir = os.path.join(
    pathlib.Path.home(),
    "earth-analytics",
    "data",
    "clustering",
)
os.makedirs(data_dir, exist_ok=True)


# Caching decorator
def cached(func_key, override=False):
    """
    A decorator to cache function results

    Parameters
    ==========
    key: str
      File basename used to save pickled results
    override: bool
      When True, re-compute even if the results are already stored
    """

    def compute_and_cache_decorator(compute_function):
        """
        Wrap the caching function

        Parameters
        ==========
        compute_function: function
          The function to run and cache results
        """

        def compute_and_cache(*args, **kwargs):
            """
            Perform a computation and cache, or load cached result.

            Parameters
            ==========
            args
              Positional arguments for the compute function
            kwargs
              Keyword arguments for the compute function
            """
            # Add an identifier from the particular function call
            if "cache_key" in kwargs:
                key = "_".join((func_key, kwargs["cache_key"]))
            else:
                key = func_key

            path = os.path.join(data_dir, "jars", f"{key}.pickle")

            # Check if the cache exists already or override caching
            if not os.path.exists(path) or override:
                # Make jars directory if needed
                os.makedirs(os.path.dirname(path), exist_ok=True)

                # Run the compute function as the user did
                result = compute_function(*args, **kwargs)

                # Pickle the object
                with open(path, "wb") as file:
                    pickle.dump(result, file)
            else:
                # Unpickle the object
                with open(path, "rb") as file:
                    result = pickle.load(file)

            return result

        return compute_and_cache

    return compute_and_cache_decorator


# Download watershed boundary shapefile


@cached("wbd_08")
def read_wbd_file(wbd_filename, huc_level, cache_key):
    # Download and unzip
    wbd_url = (
        "https://prd-tnm.s3.amazonaws.com"
        "/StagedProducts/Hydrography/WBD/HU2/Shape/"
        f"{wbd_filename}.zip"
    )

    # Download the zip file
    zip_path = os.path.join(data_dir, f"{wbd_filename}.zip")
    if not os.path.exists(zip_path):
        response = requests.get(wbd_url)

        with open(zip_path, "wb") as f:
            f.write(response.content)
    # Unzip file
    shp_dir = os.path.join(data_dir, "wbd")
    if not os.path.exists(shp_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(shp_dir)

    # Read desired data
    wbd_path = os.path.join(shp_dir, "Shape", f"WBDHU{huc_level}.shp")
    wbd_gdf = gpd.read_file(wbd_path, engine="pyogrio")

    return wbd_gdf


huc_level = 12
wbd_gdf = read_wbd_file(
    "WBD_08_HU2_Shape", huc_level, cache_key=f"hu{huc_level}"
)

delta_gdf = wbd_gdf[
    wbd_gdf[f"huc{huc_level}"].isin(["080902030506"])
].dissolve()

# Reproject the data to Mercator (EPSG:3395)
delta_gdf_merc = delta_gdf.to_crs(epsg=3395)

# Plotting with matplotlib and cartopy
fig, ax = plt.subplots(figsize=(8, 6), dpi=150,
                       subplot_kw={'projection': ccrs.Mercator()})

# Define the projection (Mercator)
projection = ccrs.Mercator()

# Set up the map with Cartopy
ax.set_extent([delta_gdf_merc.bounds.minx.min(),
               delta_gdf_merc.bounds.maxx.max(),
               delta_gdf_merc.bounds.miny.min(),
               delta_gdf_merc.bounds.maxy.max()], crs=projection)

# Add OpenStreetMap as base layer
osm = cimgt.OSM()  # OpenStreetMap as base layer
ax.add_image(osm, 10)  # The number 10 refers to the zoom level

# Add base map tiles (using Cartopy's NaturalEarthData for simplicity)
ax.add_feature(cfeature.LAND, facecolor='white')
ax.add_feature(cfeature.BORDERS, edgecolor='black')

# Plot the WBD boundary
delta_gdf_merc.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

# Add labels or titles
ax.set_title(f"Watershed Boundary: HUC-{huc_level} 080902030506", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Show the plot
plt.show()

# Set up earthaccess connection
earthaccess.login(strategy="interactive", persist=True)

# Search and download earthaccess HLS tiles
results = earthaccess.search_data(
    short_name="HLSL30",
    cloud_hosted=True,
    bounding_box=tuple(delta_gdf.total_bounds),
    temporal=("2023-05", "2023-09"),
)


def get_earthaccess_links(results):
    url_re = re.compile(
        r"\.(?P<tile_id>\w+)\.\d+T\d+\.v\d\.\d\.(?P<band>[A-Za-z0-9]+)\.tif"
    )

    # loop through each granule
    link_rows = []

    for granule in tqdm(results):
        # get granule information
        umm_dict = granule["umm"]
        granule_id = umm_dict["GranuleUR"]
        datetime = pd.to_datetime(
            umm_dict
            ['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        points = (
            umm_dict
            ['SpatialExtent']['HorizontalSpatialDomain']['Geometry']
            ['GPolygons'][0]['Boundary']['Points'])
        geometry = Polygon(
            [(point['Longitude'], point['Latitude']) for point in points])

        # Get URL
        files = earthaccess.open([granule])

        # Build metadata DataFrame
        for file in files:
            match = url_re.search(file.full_name)
            if match is not None:
                link_rows.append(
                    gpd.GeoDataFrame(
                        dict(
                            datetime=[datetime],
                            tile_id=[match.group("tile_id")],
                            band=[match.group("band")],
                            url=[file],
                            geometry=[geometry],
                        ),
                        crs="EPSG:4326",
                    )
                )

    # Concatenate metadata DataFrame
    file_df = pd.concat(link_rows).reset_index(drop=True)
    return file_df


@cached("delta_reflectance_da_df")
def compute_reflectance_da(search_results, boundary_gdf):
    """
    Connect to files over VSI, crop, cloud mask, and wrangle

    Returns a single reflectance DataFrame
    with all bands as columns and
    centroid coordinates and datetime as the index.

    Parameters
    ==========
    file_df : pd.DataFrame
        File connection and metadata (datetime, tile_id, band, and url)
    boundary_gdf : gpd.GeoDataFrame
        Boundary use to crop the data
    """

    def open_dataarray(url, boundary_proj_gdf, scale=1, masked=True):
        # Open masked DataArray
        da = rxr.open_rasterio(url, masked=masked).squeeze() * scale

        # Reproject boundary if needed
        if boundary_proj_gdf is None:
            boundary_proj_gdf = boundary_gdf.to_crs(da.rio.crs)

        # Crop
        cropped = da.rio.clip_box(*boundary_proj_gdf.total_bounds)
        return cropped

    def compute_quality_mask(da, mask_bits=[1, 2, 3]):
        """Mask out low quality data by bit"""
        # Unpack bits into a new axis
        bits = np.unpackbits(da.astype(np.uint8), bitorder="little").reshape(
            da.shape + (-1,)
        )

        # Select the required bits and check if any are flagged
        mask = np.prod(bits[..., mask_bits] == 0, axis=-1)
        return mask

    file_df = get_earthaccess_links(search_results)

    granule_da_rows = []
    boundary_proj_gdf = None

    # Loop through each image
    group_iter = file_df.groupby(["datetime", "tile_id"])
    for (datetime, tile_id), granule_df in tqdm(group_iter):
        print(f"Processing granule {tile_id} {datetime}")

        # Open granule cloud cover
        cloud_mask_url = granule_df.loc[
            granule_df.band == "Fmask", "url"
        ].values[0]
        cloud_mask_cropped_da = open_dataarray(
            cloud_mask_url, boundary_proj_gdf, masked=False
        )

        # Compute cloud mask
        cloud_mask = compute_quality_mask(cloud_mask_cropped_da)

        # Loop through each spectral band
        for i, row in granule_df.iterrows():
            if row.band.startswith("B"):
                # Open, crop, and mask the band
                band_cropped = open_dataarray(
                    row.url, boundary_proj_gdf, scale=0.0001
                )
                band_cropped.name = row.band
                # Add the DataArray to the metadata DataFrame row
                row["da"] = band_cropped.where(cloud_mask)
                granule_da_rows.append(row.to_frame().T)

    # Reassemble the metadata DataFrame
    return pd.concat(granule_da_rows)


reflectance_da_df = compute_reflectance_da(results, delta_gdf)


@cached('delta_reflectance_da')
def merge_and_composite_arrays(granule_da_df):
    # Merge and composite and image for each band
    da_list = []
    for band, band_df in tqdm(granule_da_df.groupby('band')):
        merged_das = []
        for datetime, date_df in tqdm(band_df.groupby('datetime')):
            # Merge granules for each date
            merged_da = rxrmerge.merge_arrays(list(date_df.da))
            # Mask negative values
            merged_da = merged_da.where(merged_da > 0)
            merged_das.append(merged_da)

        # Composite images across dates
        composite_da = xr.concat(merged_das, dim='datetime').median('datetime')
        composite_da['band'] = int(band[1:])
        composite_da.name = 'reflectance'
        da_list.append(composite_da)

    return xr.concat(da_list, dim='band')


reflectance_da = merge_and_composite_arrays(reflectance_da_df)
print(reflectance_da)

# Convert spectral DataArray to a tidy DataFrame
model_df = reflectance_da.to_dataframe().reflectance.unstack('band')
# Drop specific columns and rows with NaN
model_df = model_df.drop(columns=[10, 11]).dropna()

# Initialize a list to store silhouette and elbow scores
silhouette = []
elbow = []

# List of k values to loop through (try different numbers of clusters)
k_list = list(range(2, 12))

# Loop through different k values
for k in k_list:

    # Run KMeans and get predictions
    kmeans = KMeans(n_clusters=k, n_init=10)
    prediction = kmeans.fit_predict(model_df.values)

    # Add the predicted cluster labels to the model DataFrame
    model_df['clusters'] = prediction

    # Calculate the silhouette score for this k
    silhouette.append(silhouette_score(model_df.values, model_df['clusters']))

    # Store the inertia for this k
    elbow.append(kmeans.inertia_)

# Plot the Silhouette Scores and Elbow Method (Inertia)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Silhouette Score Plot
ax1.plot(k_list, silhouette, marker='o')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Scores for Different k Values')

# Elbow Method Plot (Inertia)
ax2.plot(k_list, elbow, marker='o')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Inertia')
ax2.set_title('Elbow Method for Different k Values')

# Show the plots
plt.tight_layout()
plt.show()

# Running the fit and predict functions at the same time.
# We can do this since we don't have target data.
prediction = KMeans(n_clusters=7).fit_predict(model_df.values)

# Add the predicted values back to the model DataFrame
model_df['clusters'] = prediction
print(model_df)

# Ensure you have the correct reflectance data array (reflectance_da)
# Select bands (4: red, 3: green, 2: blue) for the RGB image
rgb = reflectance_da.sel(band=[4, 3, 2])

# Convert to uint8
rgb_uint8 = (rgb * 255).astype(np.uint8)  # .where(rgb != np.nan)

# Brighten and saturate the image
rgb_bright = rgb_uint8 * 10
rgb_sat = rgb_bright.where(rgb_bright < 255, 255)

# Plotting the RGB image
plt.figure(figsize=(10, 10))
plt.imshow(rgb_sat.transpose('y', 'x', 'band').values)
plt.title('Brightened and Saturated RGB')
plt.axis('off')

# Make sure 'y' and 'x' coordinates are part of the DataFrame index
model_df.reset_index(inplace=True)

# Plot the clusters using scatter plot
plt.figure(figsize=(10, 8))

# Plot each point colored by its cluster
scatter = plt.scatter(
        model_df['x'], model_df['y'], c=model_df['clusters'],
        cmap='tab20', s=5
        )

# Add colorbar to indicate cluster values
plt.colorbar(scatter, label='Cluster')

# Set labels and title
plt.title('KMeans Clusters')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.show()
