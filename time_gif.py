import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
import plotly.graph_objects as go
import math
from dash import Dash, dcc, html, Input, Output
import pyproj
import copy

import numpy as np
import os
import json
from types import SimpleNamespace
import xarray as xr



if not os.path.exists('Plots/3D/'):
    os.mkdir('Plots/3D/')

def updata_graph(property, camera_angle, camera_height, year_iterate):
    # load params
    path_to_json_saved = "ReferenceSimulation/params_saved.json"
    with open(path_to_json_saved, "r") as json_file:
        json_text = json_file.read()
    params = json.loads(json_text, object_hook=lambda d: SimpleNamespace(**d))

    # read output.nc
    ds = xr.open_dataset('ReferenceSimulation/'+params.wncd_output_file, engine="netcdf4")

    # get attributes from ds

    time = np.array(ds.time)
    lat_range = np.array(ds.x)
    lon_range = np.array(ds.y)
    glacier_surfaces = np.array(ds.usurf)
    thicknesses = np.array(ds.thk)
    velocities = np.array(ds.velsurf_mag)
    smbs = np.array(ds.smb)
    try:
        bedrock = np.array(ds.topg[0])
    except:
        bedrock = glacier_surfaces[0] - thicknesses[0]

    # choose property that is displayed on the glacier surface

    property_maps = thicknesses
    color_scale = "Blues"
    max_property_map = np.max(property_maps)
    min_property_map = np.min(property_maps)


    # make edges equal so that it looks like a volume
    max_bedrock = np.max(bedrock)
    min_bedrock = np.min(bedrock)
    bedrock_border = copy.copy(bedrock)
    bedrock_border[0, :] = min_bedrock
    bedrock_border[-1, :] = min_bedrock
    bedrock_border[:, 0] = min_bedrock
    bedrock_border[:, -1] = min_bedrock



    # create time frames for slider

    # update elevation data and property map
    i = year_iterate - 2000
    property_map = property_maps[i]

    glacier_surface = glacier_surfaces[i]
    glacier_surface[thicknesses[i] < 1] = None

    glacier_bottom = copy.copy(bedrock)
    glacier_bottom[thicknesses[i] < 1] = None

    # create 3D surface plots with property as surface color
    surface_fig = go.Surface(
        z=glacier_surface,
        x=lat_range,
        y=lon_range,
        colorscale=color_scale,
        cmax=max_property_map,
        cmin=min_property_map,
        surfacecolor=property_map,
        showlegend=False,
        name="glacier surface",
        #colorbar=dict(title=property, titleside="right"),
    )


    # create 3D bedrock plots
    bedrock_fig = go.Surface(
        z=bedrock_border,
        x=lat_range,
        y=lon_range,
        colorscale='gray',
        opacity=1,
        showlegend=False,
        name="bedrock",
        cmax=max_bedrock,
        cmin=0,
        #colorbar=dict(x=-0.1, title="elevation [m]", titleside="right"),
    )


    # define figure layout
    try:
        title = params.oggm_RGI_ID
    except:
        title = str(year_iterate)

    # compute aspect ratio of the base
    resolution = int(lat_range[1] - lat_range[0])
    ratio_y = bedrock.shape[0] / bedrock.shape[1]
    ratio_z = (max_bedrock - min_bedrock) / (bedrock.shape[0] * resolution)
    ratio_z *= 2  # emphasize z-axis to make mountians look twice as steep

    # transform angle[0-180] into values between [0, 1] for camera postion
    radians = math.radians(camera_angle - 180)
    camera_x = math.sin(-radians)-1
    camera_y = math.cos(-radians)-1

    # Define the UTM projection (UTM zone 32N)
    utm_proj = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')

    # Define the WGS84 projection
    wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')

    # Example coordinate in UTM zone 32N (replace these values with your coordinates)
    utm_easting = lat_range  # example easting value
    utm_northing = lon_range  # example northing value

    # Reproject the coordinate
    lon_x, lat_x = pyproj.transform(utm_proj, wgs84_proj, utm_easting, np.ones_like(utm_easting) * utm_northing[0])
    lon_y, lat_y = pyproj.transform(utm_proj, wgs84_proj, np.ones_like(utm_northing) * utm_easting[0], utm_northing)

    # Output the WGS84 coordinate

    fig_dict = dict(
        data= [bedrock_fig, surface_fig],

        layout=dict(  # width=1800,
            height=800,
            margin=dict(l=0, r=0, t=30, b=0),
            title=title,
            font=dict(family="monospace", size=20),
            legend={"orientation": "h", "yanchor": "bottom", "xanchor": "left"},
            scene=dict(
                zaxis=dict(showbackground=True, showticklabels=False, title=""),
                xaxis=dict(
                    showbackground=False,
                    showticklabels=True,
                    visible=False,
                    range=[lat_range[0], lat_range[-1]],
                    tickvals=[ticks for ticks in lat_range[::42]],
                    ticktext=["%.2fE" % ticks for ticks in lon_x[::42]],

                    title="Longitude",

                ),
                yaxis=dict(
                    showbackground=False,
                    showticklabels=True,
                    visible=False,
                    range=[lon_range[0], lon_range[-1]],
                    title="Latitude",
                    tickvals=[ticks for ticks in lon_range[::42]],
                    ticktext=["%.2fN" % ticks for ticks in lat_y[::42]],

                ),
            ),
            scene_aspectratio=dict(x=1, y=ratio_y, z=ratio_z),
            scene_camera_eye=dict(x=camera_x, y=camera_y, z=0.8),
            scene_camera_center = dict(x=0.1, y=0, z=-0.1)

        ),
    )
    # create figure
    fig = go.Figure(fig_dict)
    fig.update_traces(showscale=False)
    fig.update_layout(title={'text': str(year_iterate), 'font': {'size': 50}, 'x': 0.5, 'y': 0.9})

    fig.write_image(f"Plots/3D/glacier_surface_{year}.png", width=1500, height=1200, scale=0.75)

for year in np.arange(2000, 2021)[:]:
    print(year)
    updata_graph("surface elevation [m]", year-2000, 1.5, year)