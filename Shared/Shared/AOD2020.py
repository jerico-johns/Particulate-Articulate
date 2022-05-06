# Databricks notebook source
!pip install cloudpathlib

# COMMAND ----------

!pip install pyhdf

# COMMAND ----------

!pip install pyproj

# COMMAND ----------

#Import Packages. 
import sys
import os
import re
import warnings
import glob
import time
from datetime import datetime
from datetime import timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from shapely.geometry import mapping, box
# import geopandas as gpd
# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep
# from osgeo import gdal
import pandas as pd
import pyproj

from cloudpathlib import S3Path, S3Client
from pyhdf.SD import SD, SDC

import boto3
import io
import pickle

# from profilehooks import profile

warnings.simplefilter('ignore')

# COMMAND ----------

#Getting latitudes and longitudes in a given HDF
def get_lat_lon(hdf, DATAFIELD_NAME='Optical_Depth_055'):
    # Construct the grid.  The needed information is in a global attribute
    # called 'StructMetadata.0'.  Use regular expressions to tease out the
    # extents of the grid.
    data3D = hdf.select(DATAFIELD_NAME)
    data = data3D[0,:,:].astype(np.double)
    
    fattrs = hdf.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                              (?P<upper_left_x>[+-]?\d+\.\d+)
                              ,
                              (?P<upper_left_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)

    match = ul_regex.search(gridmeta)
    x0 = np.float(match.group('upper_left_x'))
    y0 = np.float(match.group('upper_left_y'))

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                              (?P<lower_right_x>[+-]?\d+\.\d+)
                              ,
                              (?P<lower_right_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = np.float(match.group('lower_right_x'))
    y1 = np.float(match.group('lower_right_y'))

    nx, ny = data.shape
    x = np.linspace(x0, x1, nx, endpoint=False)
    y = np.linspace(y0, y1, ny, endpoint=False)
    xv, yv = np.meshgrid(x, y)

    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("+init=EPSG:4326") 
    lon, lat= pyproj.transform(sinu, wgs84, xv, yv)
    
    return (lon, lat)

# COMMAND ----------

#Getting readings a given dataset from given HDF
#Values set as _FillValue are marked as NaN to help distinguish later as those that need imputation
def get_sds_data(hdf, DATAFIELD_NAME, layer):

    # Read dataset.
    data3D = hdf.select(DATAFIELD_NAME)
    attrs = data3D.attributes()
    data = data3D[layer,:,:].astype(np.double)
    
    fillvalue = attrs['_FillValue']
    valid_range_low = attrs['valid_range'][0]
    valid_range_high = attrs['valid_range'][1]
    data[data==float(fillvalue)]=np.nan
    data[data<float(valid_range_low)]=np.nan
    data[data>float(valid_range_high)]=np.nan
        
    #Read attributes for scale factor and offset
    if 'scale_factor' in attrs and 'add_offset' in attrs:
        scale_factor = attrs["scale_factor"]
        add_offset = attrs["add_offset"]
        data=(data-add_offset)*scale_factor

            
#     if(DATAFIELD_NAME=='Optical_Depth_055' or DATAFIELD_NAME=='Optical_Depth_047'):
#         qa_data = extract_qa(hdf, i)
#         for i in range(len(data)):
#             for j in range(len(data[0])):
#                 if(qa_data[i][j]==0):
#                     data[i][j]=np.nan

#     all_data.append(data)
            
    average=np.nanmean(data)
    sd=np.nanstd(data)
    valid_range=attrs['valid_range']
    
    return (DATAFIELD_NAME,data)

# COMMAND ----------

#For a 1x1km square, get the readings for all given datasets
#Outputs a numpy array with longitudes, latitudes, and readings for every dataset in the argument list for that lon,lat
def get_1km_aod_data(hdf, datasets_to_read):
    
    lon, lat  = get_lat_lon(hdf, DATAFIELD_NAME='Optical_Depth_055')
    flat_lon, flat_lat = lon.ravel(), lat.ravel()
    
    all_layer_output = np.empty(len(datasets_to_read))
    
    data3D = hdf.select('Optical_Depth_055')
    num_overpasses = data3D[:,:,:].astype(np.double).shape[0]
    
    for layer in range(num_overpasses): 
        output = np.column_stack((flat_lon, flat_lat))
        for DATAFIELD_NAME in datasets_to_read[2:]: #2: onwards for datasets_to_read because it contains lon and lat as first two list elements
            data = get_sds_data(hdf, DATAFIELD_NAME, layer)[1]
            flat_data = data.ravel()
            output = np.column_stack((output, flat_data))
        all_layer_output = np.row_stack((all_layer_output,output))
    return all_layer_output

# COMMAND ----------

def get_aod_output(maiac_file_path, s3_cli):
    
    maiac_file = S3Path(maiac_file_path, client=s3_cli)
    file_name = maiac_file.fspath
    print("reading file", file_name)
    #common_part_name='/drivendata-competition-airathon-public-as/pm25/train/maiac/'
    hdf = SD(file_name, SDC.READ)
    #file_cols = ['year','month','date','hour','minute','second','product','location','reading_num']
    datasets_to_read = ['lon', 'lat', 'Optical_Depth_047', 'Optical_Depth_055',
       'AOD_Uncertainty', 'FineModeFraction', 'Column_WV', 'Injection_Height', 'AOD_QA']
    aod_data = get_1km_aod_data(hdf,datasets_to_read)

    aod_df = pd.DataFrame(aod_data, columns = datasets_to_read)
    
    return aod_df
    #np.savetxt(csv_filename, final_output, delimiter=",", fmt='%s')
    # upload without using disk
    #df.write.parquet("s3a://pollution-prediction/aod/test.parquet",mode="overwrite")

    #my_array_data = io.BytesIO()
    #pickle.dump(final_output, my_array_data)
    #my_array_data.seek(0)
    #s3_write_client.upload_fileobj(my_array_data, 'pollution-prediction', 'trial1maiac.pkl')


# COMMAND ----------

def aod_avg_5km(maiac_file_path, min_lon, max_lon, min_lat, max_lat, grid_id, aod_reading_end, pm25_reading_date, s3_cli):
    start_time = time.time()
    aod_df = get_aod_output(maiac_file_path, s3_cli)
    print("--- Time taken to get aod data for one file %s seconds ---" % (time.time() - start_time))

    
    aod_avg_5km_df = aod_df.loc[(aod_df['lon']>=float(min_lon)) & (aod_df['lat']>=float(min_lat))
                            & (aod_df['lon']<=float(max_lon)) & (aod_df['lat']<=float(max_lat))]
    print(len(aod_avg_5km_df))
    #aod_avg_5km_df_meaned = aod_avg_5km_df[aod_avg_5km_df.columns.tolist()].mean()
    aod_avg_5km_df['grid_id']=grid_id
    aod_avg_5km_df['aod_reading_end'] = aod_reading_end
    aod_avg_5km_df['pm25_reading_date'] = pm25_reading_date
    return aod_avg_5km_df#.to_frame().transpose()

# COMMAND ----------

# MAGIC %md
# MAGIC PM 25 Satellite Metadata

# COMMAND ----------

display(dbutils.fs.ls("/FileStore"))

# COMMAND ----------

pm25_sat_metadata = pd.read_csv('/dbfs/FileStore/pm25_satellite_metadata.csv')

# COMMAND ----------

pm25_sat_metadata = pm25_sat_metadata.rename(columns={"location": "loc"})

# COMMAND ----------

def utc_date_end(row):
    return row['time_end'][:row['time_end'].find(" ")]

# COMMAND ----------

pm25_sat_metadata['utc_date']=pm25_sat_metadata.apply(lambda row: utc_date_end(row), axis=1)

# COMMAND ----------

pm25_sat_metadata['time_end'].dtype

# COMMAND ----------

# MAGIC %md
# MAGIC Train Labels

# COMMAND ----------

train_labels = pd.read_csv('/dbfs/FileStore/train_labels.csv',delimiter='|')

# COMMAND ----------

def min_lon_polygon(row):
    bounds_str = row['wkt'].replace("POLYGON ((","").replace("))",",")
    bounds = bounds_str.split(",")
    lons=[]
    for b in bounds:
        lon_lat = b.strip().split(" ")
        if(len(lon_lat[0])>0):
            lons.append(float(lon_lat[0]))
    return min(lons)

def max_lon_polygon(row):
    bounds_str = row['wkt'].replace("POLYGON ((","").replace("))",",")
    bounds = bounds_str.split(",")
    lons=[]
    for b in bounds:
        lon_lat = b.strip().split(" ")
        if(len(lon_lat[0])>0):
            lons.append(float(lon_lat[0]))
    return max(lons)

# COMMAND ----------

def min_lat_polygon(row):
    bounds_str = row['wkt'].replace("POLYGON ((","").replace("))",",")
    bounds = bounds_str.split(",")
    lats=[]
    for b in bounds:
        lon_lat = b.strip().split(" ")
        if(len(b)>0 and len(lon_lat[1])>0):
            lats.append(float(lon_lat[1]))
    return min(lats)

def max_lat_polygon(row):
    bounds_str = row['wkt'].replace("POLYGON ((","").replace("))",",")
    bounds = bounds_str.split(",")
    lats=[]
    for b in bounds:
        lon_lat = b.strip().split(" ")
        if(len(b)>0 and len(lon_lat[1])>0):
            lats.append(float(lon_lat[1]))
    return max(lats)

# COMMAND ----------

def location_code(row):
    if(row['location']=='Los Angeles (SoCAB)'):
        return 'la'
    if(row['location']=='Delhi'):
        return 'dl'
    if(row['location']=='Taipei'):
        return 'tpe'

# COMMAND ----------

def utc_year(row):
    dt_str=row['datetime'][:row['datetime'].find("T")]
    dt= datetime.strptime(dt_str,"%Y-%m-%d")
    return dt.year
def utc_month(row):
    dt_str=row['datetime'][:row['datetime'].find("T")]
    dt= datetime.strptime(dt_str,"%Y-%m-%d")
    return dt.month
def utc_date(row):
    return row['datetime'][:row['datetime'].find("T")]

# COMMAND ----------

train_labels["min_lon"] = train_labels.apply(lambda row: min_lon_polygon(row), axis=1)
train_labels["max_lon"] = train_labels.apply(lambda row: max_lon_polygon(row), axis=1)
train_labels["min_lat"] = train_labels.apply(lambda row: min_lat_polygon(row), axis=1)
train_labels["max_lat"] = train_labels.apply(lambda row: max_lat_polygon(row), axis=1)
train_labels["loc"] = train_labels.apply(lambda row: location_code(row), axis=1)
train_labels["utc_date"] = train_labels.apply(lambda row: utc_date(row), axis=1)

# COMMAND ----------

train_labels[0:4]

# COMMAND ----------

# MAGIC %md
# MAGIC # Merge PM25 Satellite Metadata and Labels Data

# COMMAND ----------

#datetime(02/01 6am)

#time_end (02/01 noon)   -   datetime(02/02 6am)   YES


#time_end (02/02 noon)   -   datetime(02/02 6am)   NO

test=0
if(not test):
    pm25_sat_metadata = pm25_sat_metadata[pm25_sat_metadata['split']=='train']
    all_metadata_new = pd.merge(train_labels, pm25_sat_metadata, on=['loc'])
    all_metadata_new = all_metadata_new[(all_metadata_new['time_end'].astype('datetime64[ns]')<=
                                         all_metadata_new['datetime'].astype('datetime64[ns]')) &
                                        (all_metadata_new['time_end'].astype('datetime64[ns]') >=
                                         all_metadata_new['datetime'].astype('datetime64[ns]') - timedelta(days=1))
                                       ]

else:
    all_metadata = pm25_sat_metadata

# COMMAND ----------

all_metadata = all_metadata_new

# COMMAND ----------

all_metadata.rename(columns={'datetime': 'pm25_reading_date', 'time_end': 'aod_reading_end'}, inplace=True)


# COMMAND ----------

all_metadata['datetime_dt'] = pd.to_datetime(all_metadata['pm25_reading_date'], errors='coerce')
year_2018_metadata = all_metadata[all_metadata['datetime_dt'].dt.year==2018]

# COMMAND ----------

len(year_2018_metadata)

# COMMAND ----------

year_2019_metadata = all_metadata[all_metadata['datetime_dt'].dt.year==2019]

# COMMAND ----------

len(year_2019_metadata)

# COMMAND ----------

year_2020_metadata = all_metadata[all_metadata['datetime_dt'].dt.year==2020]

# COMMAND ----------

len(year_2020_metadata)

# COMMAND ----------

all_metadata['datetime_dt'].dt.year.unique()

# COMMAND ----------

subset = all_metadata[:3]

# COMMAND ----------

subset

# COMMAND ----------

from pyspark.sql import functions as sf


# COMMAND ----------

year_2020_metadata_sdf = spark.createDataFrame(year_2020_metadata)

# COMMAND ----------

def ifHDF(row_data, s3_cli):
    if(".hdf" in row_data['us_url']):
        return aod_avg_5km(aod_avg_5km(row_data['us_url'], 
                                  row_data['min_lon'], row_data['max_lon'], row_data['min_lat'], row_data['max_lat'], 
                                  row_data['grid_id'],row_data['aod_reading_end'],row_data['pm25_reading_date'], s3_cli))

# COMMAND ----------

udf_aod_avg_5km = sf.udf(lambda row:ifHDF(row))

# COMMAND ----------

eg5 = year_2020_metadata_sdf.take(3)

# COMMAND ----------

eg5_re = year_2020_metadata_sdf.rdd.map(lambda row:ifHDF(row))

# COMMAND ----------

year_2020_metadata_7 = all_metadata[(all_metadata['datetime_dt'].dt.year==2020) & 
                                    (all_metadata['datetime_dt'].dt.month==7)]

# COMMAND ----------

start_time = time.time()

all_5km_dfs = []

s3_cli = S3Client(no_sign_request=True)

for row in year_2020_metadata_7.iterrows():
    row_data = row[1]
    #print(row_data['us_url'])
    if(".hdf" in row_data['us_url']):
        aod_avg_5km_df = aod_avg_5km(row_data['us_url'], 
                                  row_data['min_lon'], row_data['max_lon'], row_data['min_lat'], row_data['max_lat'], 
                                  row_data['grid_id'],row_data['aod_reading_end'],row_data['pm25_reading_date'], s3_cli)
        all_5km_dfs.append(aod_avg_5km_df)
all_2020_aod_df = pd.concat(all_5km_dfs, axis=0)

all_2020_aod_df.to_parquet('/mnt/capstone/train/aod/aod_2020_7.parquet')

print("--- Time taken write 2020 month 7 grid level aod data to parquet - %s seconds ---" % (time.time() - start_time))

# COMMAND ----------

dbutils.fs.ls("/mnt/capstone/train/aod")

# COMMAND ----------

all_2020_aod_df.to_parquet('/dbfs/aod_2020_7.parquet')

# COMMAND ----------

spark.createDataFrame(all_2020_aod_df).coalesce(1).write.parquet("/mnt/capstone/train/aod/aod_2020_7.parquet")

# COMMAND ----------

