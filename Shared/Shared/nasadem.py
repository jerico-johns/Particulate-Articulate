# Databricks notebook source
!pip install datashader

# COMMAND ----------

!pip install pystac_client

# COMMAND ----------

!pip install xarray-spatial

# COMMAND ----------

!pip install planetary-computer

# COMMAND ----------

!pip install xarray

# COMMAND ----------

!pip install rasterio

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore') #ignoring all warnings since this is only an EDA notebook

# COMMAND ----------

from datashader.transfer_functions import shade, stack
from datashader.colors import Elevation
from pystac_client import Client
from xrspatial import hillshade

import planetary_computer as pc
import xarray as xr

from pyspark.sql.types import DoubleType


# COMMAND ----------

import numpy as np
import ast

# COMMAND ----------

grid_1X116 = {
    "type": "Polygon",
    "coordinates": [
        [
            [121.5257644471362, 24.97766123020391],
            [121.5257644471362, 25.01836939334328],
            [121.4808486829302, 25.01836939334328],
            [121.4808486829302, 24.97766123020391],
            [121.5257644471362, 24.97766123020391],
        ]
    ],
}

#Taipei
# POLYGON ((121.5257644471362 24.97766123020391, #top left
#           121.5257644471362 25.01836939334328, #bottom left
#           121.4808486829302 25.01836939334328, #bottom right
#           121.4808486829302 24.97766123020391, #top right
#           121.5257644471362 24.97766123020391)) #top left

#Delhi
# POLYGON ((77.30453178416276 28.54664454217707, 
#           77.30453178416276 28.58609243100243, 
#           77.25961601995678 28.58609243100243, 
#           77.25961601995678 28.54664454217707, 
#           77.30453178416276 28.54664454217707))

#LA
# POLYGON ((-117.3948356552278 33.98201108613195, 
#           -117.3948356552278 34.01924766510738, 
#           -117.4397514194338 34.01924766510738, 
#           -117.4397514194338 33.98201108613195, 
#           -117.3948356552278 33.98201108613195))

# COMMAND ----------

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
nasadem = catalog.search(collections=["nasadem"], intersects=grid_1X116)
# nasadem_bl = catalog.search(collections=["nasadem"], intersects=areas_of_interest_bl)
# nasadem_br = catalog.search(collections=["nasadem"], intersects=areas_of_interest_br)
# nasadem_tr = catalog.search(collections=["nasadem"], intersects=areas_of_interest_tr)

items = [item for item in nasadem.get_items()]
# items_bl = [item for item in nasadem_bl.get_items()]
# items_br = [item for item in nasadem_br.get_items()]
# items_tr = [item for item in nasadem_tr.get_items()]

print(f"Returned {len(items)} Items")
# print(f"Returned {len(items_bl)} Items")
# print(f"Returned {len(items_br)} Items")
# print(f"Returned {len(items_tr)} Items")

for item in items:
    print(f"{item.id}: {item.datetime}")


# COMMAND ----------

dir(items[0])

# COMMAND ----------

items[1].to_dict()

# COMMAND ----------

signed_asset = pc.sign(items[0].assets["elevation"])

da = (
    xr.open_rasterio(signed_asset.href)
    .squeeze()
    .drop("band")[:-1, :-1]
    .coarsen({"y": 5, "x": 5})
    .mean()
)

# COMMAND ----------

da

# COMMAND ----------

da_data = da.data
min_elevation = da_data.min()
max_elevation = da_data.max()
avg_elevation = da_data.mean()
pertl25_elevation = np.percentile(da_data, 25)
pertl75_elevation = np.percentile(da_data, 25)
median_elevation = np.percentile(da_data, 50)

# COMMAND ----------

print(min_elevation,max_elevation,avg_elevation,pertl25_elevation,pertl75_elevation,median_elevation)

# COMMAND ----------

# Render the hillshade with a coloramp of the values applied on top
shaded = hillshade(da, azimuth=100, angle_altitude=50)
stack(shade(shaded, cmap=["white", "gray"]), shade(da, cmap=Elevation, alpha=128))

# COMMAND ----------

import boto3
import pandas as pd

# COMMAND ----------

# Download training labels. 
file='train/grids_poly_coords.csv'
bucket='particulate-articulate-capstone'

#buffer = io.BytesIO()
s3_read_client = boto3.client('s3')
s3_tl_obj = s3_read_client.get_object(Bucket= bucket, Key= file)
#s3_tl_obj.download_fileobj(buffer)
train_labels = pd.read_csv(s3_tl_obj['Body'],header=0)

# COMMAND ----------

train_labels

# COMMAND ----------

def get_elevation_details(row):
    aoi = {
    "type": "Polygon",
    "coordinates": [
        ast.literal_eval(val)
    ],
    }
    print(aoi)
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    nasadem = catalog.search(collections=["nasadem"], intersects=aoi)

    items = [item for item in nasadem.get_items()]
    signed_asset = pc.sign(items[0].assets["elevation"])

    da = (
        xr.open_rasterio(signed_asset.href)
        .squeeze()
        .drop("band")[:-1, :-1]
        .coarsen({"y": 5, "x": 5})
        .mean()
    )
    da_data = da.data
    min_elevation = da_data.min()
    max_elevation = da_data.max()
    avg_elevation = da_data.mean()
    pertl25_elevation = np.percentile(da_data, 25)
    pertl75_elevation = np.percentile(da_data, 25)
    median_elevation = np.percentile(da_data, 50)
    
    return [min_elevation,max_elevation,avg_elevation,pertl25_elevation,pertl75_elevation,median_elevation]

# COMMAND ----------

train_labels['min_elevation'] = train_labels.apply(lambda row: get_elevation_details(row)[0], axis = 1)
train_labels['max_elevation'] = train_labels.apply(lambda row: get_elevation_details(row)[1], axis = 1)
train_labels['avg_elevation'] = train_labels.apply(lambda row: get_elevation_details(row)[2], axis = 1)
train_labels['pertl25_elevation'] = train_labels.apply(lambda row: get_elevation_details(row)[3], axis = 1)
train_labels['pertl75_elevation'] = train_labels.apply(lambda row: get_elevation_details(row)[4], axis = 1)
train_labels['median_elevation'] = train_labels.apply(lambda row: get_elevation_details(row)[5], axis = 1)

# COMMAND ----------

display(dbutils.fs.ls("/mnt/%s" % mount_name))

# COMMAND ----------

spark_df = spark.createDataFrame(train_labels)

# COMMAND ----------

spark_df.write.parquet("/mnt/capstone/train/elevation.parquet")

# COMMAND ----------

#@udf(returnType=DoubleType())
def get_avg_elevation(lat, lon):
    aoi = {
    "type": "Point",
    "coordinates": [
        lon,lat
    ],
    }
    print(aoi)
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    nasadem = catalog.search(collections=["nasadem"], intersects=aoi)

    items = [item for item in nasadem.get_items()]
    signed_asset = pc.sign(items[0].assets["elevation"])

    da = (
        xr.open_rasterio(signed_asset.href)
        .squeeze()
        .drop("band")[:-1, :-1]
        .coarsen({"y": 5, "x": 5})
        .mean()
    )
    da_data = da.data
    avg_elevation = da_data.mean()
#     pertl25_elevation = np.percentile(da_data, 25)
#     pertl75_elevation = np.percentile(da_data, 25)
#     median_elevation = np.percentile(da_data, 50)
    
    return avg_elevation#,pertl25_elevation,pertl75_elevation,median_elevation]

# COMMAND ----------

@udf(returnType=DoubleType())
def get_max_elevation(lat, lon):
    aoi = {
    "type": "Point",
    "coordinates": [
        lon,lat
    ],
    }
    print(aoi)
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    nasadem = catalog.search(collections=["nasadem"], intersects=aoi)

    items = [item for item in nasadem.get_items()]
    signed_asset = pc.sign(items[0].assets["elevation"])

    da = (
        xr.open_rasterio(signed_asset.href)
        .squeeze()
        .drop("band")[:-1, :-1]
        .coarsen({"y": 5, "x": 5})
        .mean()
    )
    da_data = da.data
    max_elevation = da_data.max()
    
    return max_elevation 

# COMMAND ----------

#@udf(returnType=DoubleType())
def get_elevation_details(lat, lon):
    aoi = {
    "type": "Point",
    "coordinates": [
        lon,lat
    ],
    }
    print(aoi)
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    nasadem = catalog.search(collections=["nasadem"], intersects=aoi)

    items = [item for item in nasadem.get_items()]
    signed_asset = pc.sign(items[0].assets["elevation"])

    da = (
        xr.open_rasterio(signed_asset.href)
        .squeeze()
        .drop("band")[:-1, :-1]
        .coarsen({"y": 5, "x": 5})
        .mean()
    )
    da_data = da.data
    min_elevation = da_data.min()
    max_elevation = da_data.max()
    avg_elevation = da_data.mean()
    return [min_elevation,max_elevation,avg_elevation]#,pertl25_elevation,pertl75_elevation,median_elevation]

# COMMAND ----------

get_elevation_details(25.062361,121.526528)

# COMMAND ----------

lat_lon_for_elev=spark.read.parquet("/mnt/capstone/lat_lon_for_elev.parquet")

# COMMAND ----------

lat_lon_for_elev =  lat_lon_for_elev.dropDuplicates(['latitude','longitude'])

# COMMAND ----------

lat_lon_for_elev.count()

# COMMAND ----------

lat_lon_for_elev_pd = lat_lon_for_elev.toPandas()

# COMMAND ----------

lat_lon_for_elev_pd['min_elevation'] = lat_lon_for_elev_pd.apply(lambda row: get_elevation_details(row.latitude,row.longitude)[0], axis = 1)
lat_lon_for_elev_pd['max_elevation'] = lat_lon_for_elev_pd.apply(lambda row: get_elevation_details(row.latitude,row.longitude)[1], axis = 1)
lat_lon_for_elev_pd['avg_elevation'] = lat_lon_for_elev_pd.apply(lambda row: get_elevation_details(row.latitude,row.longitude)[2], axis = 1)

# COMMAND ----------

elev_lat_lon_level = spark.createDataFrame(lat_lon_for_elev_pd)

# COMMAND ----------

elev_lat_lon_level.write.parquet("/mnt/capstone/df_elev_lat_lon_level.parquet")

# COMMAND ----------

