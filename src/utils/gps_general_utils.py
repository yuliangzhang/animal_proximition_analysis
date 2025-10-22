import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from shapely.wkt import loads as wkt_loads
from typing import List, Tuple
from datetime import datetime, timedelta
import contextily as ctx

def gps_extract_and_combine(base_dir, output_file):
    # base_dir = "GPS_2024_Summer"
    # output_file = "yuin_summer_2024_gps.csv"
    gps_file_pattern = 'GPS????_gps.csv'  # Modified pattern to match the requirement

    gps_file_list = glob.glob(os.path.join(base_dir, gps_file_pattern))  # Ensure we search within base_dir

    gps_df_list = []
    processed_files = set() # To keep track of processed files and avoid duplicates

    for gps_file in gps_file_list:
        # Extract the filename without the directory path
        file_name = os.path.basename(gps_file)

        if file_name in processed_files:
            print(f"Skipping already processed file: {file_name}")
            continue

        gps_code = file_name[:7]  # Corrected to use the filename

        try:
            tmp_gps = pd.read_csv(gps_file)
        except Exception as e:
            print(f"Error reading file {gps_file}: {e}")
            continue

        # Standardize column names immediately after reading
        expected_columns = ['current_time_count', 'lat', 'lon', 'day', 'month', 'year', 'hour', 'minute', 'second']
        if len(tmp_gps.columns) == len(expected_columns):
            tmp_gps.columns = expected_columns
        else:
            print(f"Warning: File {gps_file} has an unexpected number of columns. Skipping.")
            continue

        # Rename the 'date' column to 'day'
        tmp_gps = tmp_gps.rename(columns={'date': 'day'})
        tmp_gps['gps_code'] = gps_code  # Add the GPS code to the DataFrame
        # Create the timestamp column
        try:
            tmp_gps['timestamp'] = pd.to_datetime(tmp_gps[['year', 'month', 'day', 'hour', 'minute', 'second']], errors='coerce')
        except Exception as e:
            print(f"Error creating timestamp for file {gps_file}: {e}")
            continue

        gps_df_list.append(tmp_gps)
        processed_files.add(file_name)

    if gps_df_list:
        gps_df = pd.concat(gps_df_list, ignore_index=True) # Added ignore_index for clean indexing after concat
        gps_df = gps_df.drop_duplicates(subset=['gps_code', 'lat', 'lon', 'timestamp'], keep='first') # Removed individual date/time columns, using 'timestamp'
        gps_df[['gps_code', 'timestamp','lat', 'lon']].to_csv(output_file, index=False, header=True, encoding='utf-8')
        print(f"Successfully processed and saved data to {output_file}")
    else:
        print("No matching GPS files found or processed.")



def parse_polygons_from_wkt_tsv(tsv_path: str) -> List[Tuple[str, Polygon]]:

    if '.tsv' in tsv_path:
        df = pd.read_csv(tsv_path, sep='\t')
    else:
        df = pd.read_csv(tsv_path)

    polygons = []
    for _, row in df.iterrows():
        name = row['name']
        try:
            polygon = wkt_loads(row['WKT'])
            polygons.append((name, polygon))
        except Exception as e:
            print(f"[Warning] Failed to parse: {name}, Error: {e}")
    return polygons


def time_tag_to_timestamp(time_tag: str) -> str:
    """Convert time_tag format like 2023-02-13-270 to datetime string like '2023-02-13 22:30:00'"""
    date_part, index_part = time_tag.rsplit("-", 1)
    base_time = datetime.strptime(date_part, "%Y-%m-%d")
    delta_minutes = (int(index_part) - 1) * 5
    full_time = base_time + timedelta(hours=8,minutes=delta_minutes)
    return full_time.strftime("%Y-%m-%d %H:%M:%S")

# get the boundary of the polygons
def get_polygon_boundary(polygons_named):
    min_y = min(polygon.bounds[0] for _, polygon in polygons_named)
    min_x = min(polygon.bounds[1] for _, polygon in polygons_named)
    max_y = max(polygon.bounds[2] for _, polygon in polygons_named)
    max_x = max(polygon.bounds[3] for _, polygon in polygons_named)
    return min_x, min_y, max_x, max_y


## Generate time_tag
def generate_time_tag(dt):
    """Generates a time_tag based on the date and 5-minute intervals."""
    start_of_day = dt.floor('D')
    time_diff = dt - start_of_day
    interval = int(time_diff.total_seconds() // 300)  # 300 seconds = 5 minutes
    return f"{start_of_day.strftime('%Y-%m-%d')}-{interval + 1:03d}"

def pre_process_gps_data(input_file, file_path, WKT_PATH, output_dir):

    # # Load the GPS data
    # file_path = 'yuin_summer_2024_gps.csv'
    # WKT_PATH = "./Yuin station.csv"  # WKT polygon definition file

    # Load the fence plygons
    polygons_named = parse_polygons_from_wkt_tsv(WKT_PATH)

    gps_data = pd.read_csv(input_file)

    # # Ensure datetime column is in datetime format
    gps_data['datetime'] = pd.to_datetime(gps_data['timestamp'])

    # Round datetime to the nearest 5 minutes
    gps_data['datetime_5min'] = gps_data['datetime'].dt.round('5min')

    # get the boundary of the polygons
    min_x, min_y, max_x, max_y = get_polygon_boundary(polygons_named)

    # filter out invalid GPS points
    gps_data = gps_data[
        (gps_data['lat'].between(min_x, max_x)) &
        (gps_data['lon'].between(min_y, max_y))
    ].copy()



    # Extract relevant columns for overall convex hull
    gps_coordinates = gps_data[['lat', 'lon']].dropna()

    # Calculate the convex hull for all GPS points
    # points = gps_coordinates.to_numpy()


    # Directory to save 5-minute heatmaps
    # output_dir = "2023_summer_heatmap_5min"
    # os.makedirs(output_dir, exist_ok=True)



    fill_value = gps_data['datetime'].min()
    gps_data['datetime_filled'] = gps_data['datetime'].fillna(fill_value)

    # Only keep 2024 data
    # date_filter_start = pd.Timestamp('2024')
    # # date_filter_end = pd.Timestamp('2023-04-23')
    # gps_data = gps_data[(gps_data['datetime_filled'] >= date_filter_start)]

    gps_data['time_tag'] = gps_data['datetime_filled'].apply(generate_time_tag)

    gps_data[["time_tag", 'lat', 'lon']].to_csv(file_path, index=False, header=True, encoding='utf-8')

    return polygons_named, min_x, min_y, max_x, max_y


def generate_heatmaps(file_path, output_dir, polygons_named, min_x, min_y, max_x, max_y):
    """
    Generate heatmaps of sheep activity with a satellite basemap,
    with heatmap transparency adjusted to reveal the underlying map.

    Parameters:
    - file_path (str): Path to the processed GPS data CSV.
    - output_dir (str): Directory to save the generated heatmaps.
    - polygons_named (List[Tuple[str, Polygon]]): List of named Shapely Polygon objects (electronic fences).
    - min_x (float): Minimum latitude boundary.
    - min_y (float): Minimum longitude boundary.
    - max_x (float): Maximum latitude boundary.
    - max_y (float): Maximum longitude boundary.
    """
    gps_data = pd.read_csv(file_path)
    os.makedirs(output_dir, exist_ok=True)

    time_tag_list = gps_data['time_tag'].unique().tolist()
    time_tag_list.sort()

    # I've kept your slice for demonstration, but you might want to remove it for full data processing
    for time_tag in time_tag_list:
        interval_data = gps_data[
            (gps_data['time_tag'] == time_tag) &
            (gps_data['lat'].between(min_x, max_x)) &
            (gps_data['lon'].between(min_y, max_y))
        ].copy()

        interval_coordinates = interval_data[['lat', 'lon']].dropna().values

        if len(interval_coordinates) > 0:
            try:
                kde = gaussian_kde(interval_coordinates.T)

                x_grid, y_grid = np.mgrid[min_x:max_x:200j, min_y:max_y:200j]
                positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
                density = np.reshape(kde(positions).T, x_grid.shape)
                # Mirror the density along the x-axis to match GPS coordinates
                mirrored_density = np.flipud(density)

                plt.figure(figsize=(10, 8))
                ax = plt.gca()



                ax.scatter(interval_data['lon'], interval_data['lat'], s=5, c='blue', alpha=0.5,
                            label='Sheep Points')

                for name, polygon in polygons_named:
                    poly_x, poly_y = polygon.exterior.xy
                    ax.plot(poly_x, poly_y, 'cyan', linewidth=2) # Added label for legend

                # for name, polygon in polygons_named:
                #     x, y = polygon.exterior.xy
                #     plt.plot(x, y, 'cyan', label="Shade_Area", linewidth=2)

                timestamp_str = time_tag_to_timestamp(time_tag)
                ax.set_title(f'Sheep Activity Hotspot Map for {timestamp_str}')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.legend() # Ensure legend is displayed

                try:
                    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Esri.WorldImagery,
                                    attribution=True,
                                    zoom='auto')
                except Exception as e:
                    print(f"Warning: Error adding basemap for {time_tag}: {e}")


                # --- MODIFICATION HERE: Add alpha to imshow ---
                ax.imshow(
                    mirrored_density,
                    cmap=plt.cm.hot,
                    extent=[min_y, max_y, min_x, max_x],
                    aspect='auto',
                    alpha=0.3 # Adjust this value (0.0 to 1.0) to control transparency
                )
                # plt.colorbar(im, ax=ax) # Re-enabled colorbar for density reference

                plt.tight_layout()
                output_path = os.path.join(output_dir, f"{time_tag}.png")
                plt.savefig(output_path)
                plt.close()
                print(f"{time_tag} Done! Saved to {output_path}")

            except Exception as e:
                print(f"Error generating heatmap for {time_tag}: {e}")
                plt.close()