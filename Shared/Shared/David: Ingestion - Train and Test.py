# Databricks notebook source
# MAGIC %md
# MAGIC # Data Download & Modeling Notebook

# COMMAND ----------

# MAGIC %md Notebook to begin ingesting pipelined data, run some basic EDA, and begin to build some modelling infrastructure. 

# COMMAND ----------

#Import Packages. 
import sys
import os
import requests
import warnings
import glob
import time
import re
from pathlib import Path
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
import numpy as np
import pandas as pd

#For writing to s3
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import io
import pickle

#Spark
from pyspark.sql import functions as sf
from pyspark.sql.functions import col, lit,isnan, when, count, substring, date_format
from pyspark.sql.types import DoubleType, IntegerType, TimestampType, DateType, FloatType, LongType, StringType, StructField, StructType
from pyspark.sql.window import Window
from pyspark.ml.feature import Imputer
from datetime import datetime, timedelta
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import LinearRegression
from sparkdl.xgboost import XgboostRegressor
from pyspark.ml.feature import PCA

from pyspark.sql.functions import dayofyear
from pyspark.sql.functions import dayofmonth
from pyspark.sql.functions import dayofweek
from pyspark.sql.functions import weekofyear
from pyspark.sql.functions import month
from pyspark.sql.functions import year
#TODO: Import holidays package (https://towardsdatascience.com/5-minute-guide-to-detecting-holidays-in-python-c270f8479387) package to get country's holidays. 


warnings.simplefilter('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.) Download data from s3, format, and join. 

# COMMAND ----------

# MAGIC %md Define Helper Functions

# COMMAND ----------

def directory_to_sparkDF(directory, schema=None, drop_cols = None): 
    '''Iterate through all files in a provided directory (that are downloaded from s3 bucket) 
    and return a unioned Spark Dataframe.'''
    
    files = Path(directory).glob('*.parquet')
  
    first = True
    
    for file in files:
        # Must read into Pandas DF first so that we preserve the index column (i.e. grid_id for GFS data). 
        # TODO: Drop landn columns (can't include from 6/13/19 onwards). 
        df = pd.read_parquet(file, engine = 'pyarrow')
        df.reset_index(drop=False,inplace=True)
        if first: 
            df_out = spark.createDataFrame(df, schema=schema)
            if drop_cols: 
                df_out = df_out.drop(*drop_cols)
            first = False
        else: 
            # If files have mismatched column counts (a handful of files may be missing certain forecast times), 
            # continue without union (i.e. drop "corrupt" file from training). 
            try: 
                df_new = spark.createDataFrame(df, schema=schema)
                if drop_cols: 
                    df_new = df_new.drop(*drop_cols)
                df_out = df_new.union(df_out)
            except: 
                continue

    # Output shape so that we can quickly check that function works as intended. 
    print(f'Rows: {df_out.count()}, Columns: {len(df_out.columns)}')
    return df_out

# COMMAND ----------

# MAGIC %md  Install AWS CLI and Pull Data from Public s3 Bucket

# COMMAND ----------

# # # Comment out after first run. 
!curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
!unzip awscliv2.zip
!sudo ./aws/install
!aws --version

# #Download GFS data from s3 bucket to Databricks workspace. We will then load these files into a Spark DF. 
# !aws s3 cp s3://capstone-particulate-storage/GFS train/GFS/gfs --no-sign-request --recursive


# COMMAND ----------

# # Download AOD data from s3 bucket to Databricsk workspace. 
!aws s3 cp s3://particulate-articulate-capstone/train/aod train/AOD/ --no-sign-request --recursive

# COMMAND ----------

elevation_df = spark.read.parquet("dbfs:/mnt/capstone/train/elevation/elevation.parquet")

# COMMAND ----------

df_GFS = spark.read.parquet("dbfs:/mnt/capstone/train/df_GFS.parquet")

# COMMAND ----------

df_GFS = df_GFS.withColumn("gfs_date_d", sf.to_date(df_GFS['date']))
df_GFS = df_GFS.withColumnRenamed('grid_id', 'grid_id_gfs')
df_GFS = df_GFS.withColumnRenamed('date', 'gfs_date')

# COMMAND ----------

# Download training labels. 
file='meta_data/train_labels_grid.csv'
bucket='capstone-particulate-storage'

#buffer = io.BytesIO()
s3_read_client = boto3.client('s3')
s3_tl_obj = s3_read_client.get_object(Bucket= bucket, Key= file)
#s3_tl_obj.download_fileobj(buffer)
train_labels = pd.read_csv(s3_tl_obj['Body'],delimiter='|',header=0)
train_labels_df = spark.createDataFrame(train_labels)
train_labels_df = train_labels_df.withColumn("pm25_date_d", sf.to_date(train_labels_df['datetime']))
w = train_labels_df.withColumn("pm25_datetime_dt", sf.to_timestamp(train_labels_df['datetime']))

# COMMAND ----------

train_labels_df.show(1)

# COMMAND ----------

df_GFS_labels_join = train_labels_df.join(df_GFS, on=[train_labels_df.grid_id == df_GFS.grid_id_gfs,
                                                 ((train_labels_df.pm25_date_d == df_GFS.gfs_date_d) 
                                                   | (sf.date_add(train_labels_df.pm25_date_d,1) == df_GFS.gfs_date_d)),  
                                                 ],
                                                how="outer")

# COMMAND ----------

cols_00 = [col for col in df_GFS_labels_join.columns if '00' in col]
cols_06 = [col for col in df_GFS_labels_join.columns if '06' in col]
cols_12 = [col for col in df_GFS_labels_join.columns if '12' in col]
cols_18 = [col for col in df_GFS_labels_join.columns if '18' in col]

# COMMAND ----------

import math

# COMMAND ----------

for col in cols_00:
    df_GFS_labels_join = df_GFS_labels_join\
    .withColumn(col+'_new',when((((df_GFS_labels_join.pm25_date_d == df_GFS_labels_join.gfs_date_d) 
                                          & (sf.hour(df_GFS_labels_join.pm25_datetime_dt)<0)) | 
                                ((df_GFS_labels_join.pm25_date_d < df_GFS_labels_join.gfs_date_d) 
                                         & (sf.hour(df_GFS_labels_join.pm25_datetime_dt)>=0)))
        ,df_GFS_labels_join[col]).otherwise(-math.inf))\
    .drop(df_GFS_labels_join[col])

# COMMAND ----------

for col in cols_06:
    df_GFS_labels_join = df_GFS_labels_join\
    .withColumn(col+'_new',when((((df_GFS_labels_join.pm25_date_d == df_GFS_labels_join.gfs_date_d) 
                                          & (sf.hour(df_GFS_labels_join.pm25_datetime_dt)<6)) | 
                                ((df_GFS_labels_join.pm25_date_d < df_GFS_labels_join.gfs_date_d) 
                                         & (sf.hour(df_GFS_labels_join.pm25_datetime_dt)>=6)))
        ,df_GFS_labels_join[col]).otherwise(-math.inf))\
    .drop(df_GFS_labels_join[col])

# COMMAND ----------

for col in cols_12:
    df_GFS_labels_join = df_GFS_labels_join\
    .withColumn(col+'_new',when((((df_GFS_labels_join.pm25_date_d == df_GFS_labels_join.gfs_date_d) 
                                          & (sf.hour(df_GFS_labels_join.pm25_datetime_dt)<12)) | 
                                ((df_GFS_labels_join.pm25_date_d < df_GFS_labels_join.gfs_date_d) 
                                         & (sf.hour(df_GFS_labels_join.pm25_datetime_dt)>=12)))
        ,df_GFS_labels_join[col]).otherwise(-math.inf))\
    .drop(df_GFS_labels_join[col])


# COMMAND ----------

for col in cols_18:
    df_GFS_labels_join = df_GFS_labels_join\
    .withColumn(col+'_new',when((((df_GFS_labels_join.pm25_date_d == df_GFS_labels_join.gfs_date_d) 
                                          & (sf.hour(df_GFS_labels_join.pm25_datetime_dt)<18)) | 
                                ((df_GFS_labels_join.pm25_date_d < df_GFS_labels_join.gfs_date_d) 
                                         & (sf.hour(df_GFS_labels_join.pm25_datetime_dt)>=18)))
        ,df_GFS_labels_join[col]).otherwise(-math.inf))\
    .drop(df_GFS_labels_join[col])

# COMMAND ----------

display(df_GFS_labels_join)

# COMMAND ----------

display(df_GFS_labels_join[df_GFS_labels_join['pm25_date_d'] == '2020-12-31'])

# COMMAND ----------

display(df_GFS_labels_join[df_GFS_labels_join['location']=='Los Angeles (SoCAB)'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create ds_GFS_labels_final

# COMMAND ----------

df_GFS_labels_final = df_GFS_labels_join.groupby(["datetime","grid_id","value","location","tz","wkt","pm25_date_d","pm25_datetime_dt","latitude","longitude"]).max()

# COMMAND ----------

display(df_GFS_labels_final.where("grid_id = 'XNLVD'"))

# COMMAND ----------

df_GFS_labels_final[df_GFS_labels_final['value']!=df_GFS_labels_final['max(value)']].count()

# COMMAND ----------

df_GFS_labels_final[df_GFS_labels_final['latitude']!=df_GFS_labels_final['max(latitude)']].count()

# COMMAND ----------

df_GFS_labels_final[df_GFS_labels_final['longitude']!=df_GFS_labels_final['max(longitude)']].count()

# COMMAND ----------

df_GFS_labels_final = df_GFS_labels_final.drop(*['max(value)','max(latitude)','max(longitude)'])

# COMMAND ----------

display(df_GFS_labels_final)

# COMMAND ----------

#Add centroid to calculate distance to grid center for each row

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create shapely centroids and points

# COMMAND ----------

from shapely import wkt

# COMMAND ----------

def get_centroid(polygon): 
    g = wkt.loads(polygon)
    coords = g.centroid.wkt
    return coords

# COMMAND ----------

get_centr = sf.udf(lambda x:get_centroid(x))

train_labels_df_centr = train_labels_df.withColumn('centroid', get_centr(train_labels_df.wkt))

# COMMAND ----------

train_labels_df_centr = train_labels_df_centr.select('grid_id', 'centroid').distinct()

# COMMAND ----------

df_GFS_labels_final = df_GFS_labels_final.join(train_labels_df_centr, on = 'grid_id', how = 'left')

# COMMAND ----------

#Clean grid_id showing in latitude/longitude
df_GFS_labels_final = df_GFS_labels_final.where("grid_id IS NOT NULL")

# COMMAND ----------

display(df_GFS_labels_final)

# COMMAND ----------

from shapely.geometry import Point

# COMMAND ----------

def create_point(lon, lat):
    geo_point = str(Point(float(lon), float(lat)))
    return geo_point
    

# COMMAND ----------

def calc_distance(centroid, obs_point): 
    centroid_shp = wkt.loads(centroid)
    obs_point_shp = wkt.loads(obs_point)
    distance_btwn_points = centroid_shp.distance(obs_point_shp)
    return distance_btwn_points

# COMMAND ----------

get_point = sf.udf(lambda x, y:create_point(x, y))
calculate_distance = sf.udf(lambda x,y: calc_distance(x, y))

# COMMAND ----------

df_GFS_labels_final = df_GFS_labels_final.withColumn('observation_point', get_point(df_GFS_labels_final.longitude, df_GFS_labels_final.latitude))

# COMMAND ----------

df_GFS_labels_final_with_distance = df_GFS_labels_final.withColumn('distance_to_grid_center', calculate_distance(df_GFS_labels_final.centroid, df_GFS_labels_final.observation_point))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## End of disagg GFS Labels

# COMMAND ----------

display(df_GFS_labels_final_with_distance.where('value > 200'))

# COMMAND ----------

df_GFS_labels_final.write.mode("overwrite").parquet("/mnt/capstone/train/df_GFS_datevalidated.parquet")

# COMMAND ----------

df_GFS_labels_final = spark.read.parquet("/mnt/capstone/train/df_GFS_datevalidated.parquet")

# COMMAND ----------

df_GFS_agg = df_GFS_labels_final.groupBy("grid_id","datetime","value","location","tz","wkt","pm25_date_d","pm25_datetime_dt").mean()

# COMMAND ----------

df_GFS_agg[df_GFS_agg['value']!=df_GFS_agg['avg(value)']].count()

# COMMAND ----------

df_GFS_agg = df_GFS_agg.drop("avg(value)")

# COMMAND ----------

# MAGIC %md Now open downloaded GFS files and union into a full Spark Dataframe. 

# COMMAND ----------

# df_GFS = directory_to_sparkDF(directory = 'train/GFS/gfs', drop_cols = ['landn_surface00', 'landn_surface06', 'landn_surface12', 'landn_surface18'])
# # Group GFS Data by date, grid_id
# df_GFS_agg = df_GFS.groupBy("grid_id","date").mean()

# COMMAND ----------

AODCustomSchema = StructType([
        StructField("index", LongType(), True),
        StructField("lon", FloatType(), True),
        StructField("lat", FloatType(), True),
        StructField("Optical_Depth_047", FloatType(), True),
        StructField("Optical_Depth_055", FloatType(), True),
        StructField("AOD_Uncertainty", FloatType(), True),
        StructField("FineModeFraction", FloatType(), True),
        StructField("Column_WV", FloatType(), True),
        StructField("Injection_Height", FloatType(), True),
        StructField("AOD_QA", IntegerType(), True),
        StructField("grid_id", StringType(), True),
        StructField("aod_reading_end", StringType(), True),
        StructField("pm25_reading_date", StringType(), True)])
df_AOD = directory_to_sparkDF(directory = 'train/AOD/', schema=AODCustomSchema)

# COMMAND ----------

df_AOD.write.parquet("/mnt/capstone/train/df_AOD_datevalidated.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AOD Disagg Start

# COMMAND ----------

df_AOD = spark.read.parquet("/mnt/capstone/train/df_AOD_datevalidated.parquet")

# COMMAND ----------

df_AOD = df_AOD.withColumn('AOD_QA', df_AOD.AOD_QA.cast('int'))

# COMMAND ----------

df_AOD = df_AOD.drop(*['index']).distinct()

# COMMAND ----------

df_AOD = df_AOD.groupBy(*['grid_id', 'lon', 'lat', 'aod_reading_end', 'pm25_reading_date']).mean()

# COMMAND ----------

df_AOD = df_AOD.drop(*['avg(lon)', 'avg(lat)'])

# COMMAND ----------

# MAGIC %md AOD QA Engineering

# COMMAND ----------

def qa_format(val):
    if val:
        return '{0:016b}'.format(val)

# COMMAND ----------

def masks_to_int(s):
    if s:
        return int(s, 2)

# COMMAND ----------

udf_qa_format = sf.udf(lambda x:qa_format(x))
udf_mask_int = sf.udf(lambda x:masks_to_int(x),StringType() )

# COMMAND ----------

df_AOD = df_AOD.withColumnRenamed('avg(AOD_QA)', 'AOD_QA')
df_AOD = df_AOD.withColumn('AOD_QA', df_AOD.AOD_QA.cast('int'))
# df_AOD = df_AOD.withColumn('AOD_QA', df_AOD.AOD_QA.cast('string'))

# COMMAND ----------

#Recast columns
df_AOD = df_AOD.withColumn("AOD_qa_str",udf_qa_format(df_AOD["AOD_QA"]))
df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask_str', substring('AOD_qa_str', 0,3))\
    .withColumn('AOD_QA_LWS_Mask_str', substring('AOD_qa_str', 3,2))\
    .withColumn('AOD_QA_Adj_Mask_str', substring('AOD_qa_str', 5,3))\
    .withColumn('AOD_Level_str', substring('AOD_qa_str', 8,1))\
    .withColumn('Algo_init_str', substring('AOD_qa_str', 9,1))\
    .withColumn('BRF_over_snow_str', substring('AOD_qa_str', 10,1))\
    .withColumn('BRF_climatology_str', substring('AOD_qa_str', 11,1))\
    .withColumn('AOD_QA_SC_Mask_str', substring('AOD_qa_str', 12,3))
df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_LWS_Mask', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_Adj_Mask', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_Level', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('Algo_init', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('BRF_over_snow', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('BRF_climatology', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_SC_Mask', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))

qa_masks_str_cols = ('AOD_QA_Cloud_Mask_str',
                    'AOD_QA_LWS_Mask_str',
                    'AOD_QA_Adj_Mask_str',
                    'AOD_Level_str',
                    'Algo_init_str',
                    'BRF_over_snow_str',
                    'BRF_climatology_str',
                    'AOD_QA_SC_Mask_str')
df_AOD.drop(*qa_masks_str_cols)

df_AOD.registerTempTable("aod")

#AOD Lat-Lon pairs as list
df_AOD = df_AOD.withColumn("lon-lat-pair", sf.concat_ws('_',df_AOD.lon,df_AOD.lat))

lat_lon_list_df = df_AOD.groupBy("grid_id","aod_reading_end","pm25_reading_date")\
.agg(sf.collect_list("lon-lat-pair").alias("aod_lon_lat_list"))

# COMMAND ----------

# MAGIC %md Aggregate to grid level by taking summary statistics across grids. 

# COMMAND ----------

df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_LWS_Mask', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_Adj_Mask', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_Level', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('Algo_init', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('BRF_over_snow', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('BRF_climatology', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_SC_Mask', udf_mask_int(df_AOD["AOD_QA_Cloud_Mask_str"]))

# COMMAND ----------

df_AOD_filtered = df_AOD\
.withColumn('Optical_Depth_047_new',when(df_AOD.AOD_QA_Cloud_Mask == 1,df_AOD.Optical_Depth_047).otherwise(None))\
.drop(df_AOD.Optical_Depth_047)\
.withColumnRenamed('Optical_Depth_047_new', 'Optical_Depth_047')\
.withColumn('Optical_Depth_055_new',when(df_AOD.AOD_QA_Cloud_Mask == 1,df_AOD.Optical_Depth_055).otherwise(None))\
.drop(df_AOD.Optical_Depth_055)\
.withColumnRenamed('Optical_Depth_055_new', 'Optical_Depth_055')


# COMMAND ----------

display(df_AOD_with_distance)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ##AOD disagg start

# COMMAND ----------

df_AOD = df_AOD.join(train_labels_df_centr, on = 'grid_id', how = 'left')

# COMMAND ----------

df_AOD = df_AOD.withColumn('observation_point', get_point(df_AOD.lon, df_AOD.lat))

# COMMAND ----------

df_AOD_with_distance = df_AOD.withColumn('AOD_distance_to_grid_center', calculate_distance(df_AOD.centroid, df_AOD.observation_point))

# COMMAND ----------

# MAGIC %md
# MAGIC ## End of AOD Disagg

# COMMAND ----------

#df_AOD_with_distance = df_AOD_with_distance.groupBy('grid_id', 'lon', 'lat', 'pm25_reading_date').mean()

# COMMAND ----------

display(df_AOD_with_distance)

# COMMAND ----------

df_AOD_filtered.registerTempTable("aod_filtered")

# COMMAND ----------

df_AOD.registerTempTable("aod")

# COMMAND ----------

# df_aod_grid = spark.sql("SELECT grid_id, pm25_reading_date, \
#             lat, lon, \
#             min(Optical_Depth_047) as min_Optical_Depth_047,\
#             max(Optical_Depth_047) as max_Optical_Depth_047,\
#             percentile_approx(Optical_Depth_047, 0.5) as median_Optical_Depth_047,\
#             min(Optical_Depth_055) as min_Optical_Depth_055,\
#             max(Optical_Depth_055) as max_Optical_Depth_055,\
#             percentile_approx(Optical_Depth_055, 0.5) as median_Optical_Depth_055,\
#             min(AOD_Uncertainty) as min_AOD_Uncertainty,\
#             max(AOD_Uncertainty) as max_AOD_Uncertainty,\
#             percentile_approx(AOD_Uncertainty, 0.5) as median_AOD_Uncertainty,\
#             min(Column_WV) as min_Column_WV,\
#             max(Column_WV) as max_Column_WV,\
#             percentile_approx(Column_WV, 0.5) as median_Column_WV,\
#             min(AOD_QA_Cloud_Mask) as min_AOD_QA_Cloud_Mask,\
#             max(AOD_QA_Cloud_Mask) as max_AOD_QA_Cloud_Mask,\
#             percentile_approx(AOD_QA_Cloud_Mask, 0.5) as median_AOD_QA_Cloud_Mask,\
#             min(AOD_QA_LWS_Mask) as min_AOD_QA_LWS_Mask,\
#             max(AOD_QA_LWS_Mask) as max_AOD_QA_LWS_Mask,\
#             percentile_approx(AOD_QA_LWS_Mask, 0.5) as median_AOD_QA_LWS_Mask,\
#             min(AOD_QA_Adj_Mask) as min_AOD_QA_Adj_Mask,\
#             max(AOD_QA_Adj_Mask) as max_AOD_QA_Adj_Mask,\
#             percentile_approx(AOD_QA_Adj_Mask, 0.5) as median_AOD_QA_Adj_Mask,\
#             min(AOD_Level) as min_AOD_Level,\
#             max(AOD_Level) as max_AOD_Level,\
#             percentile_approx(AOD_Level, 0.5) as median_AOD_Level,\
#             min(Algo_init) as min_Algo_init,\
#             max(Algo_init) as max_Algo_init,\
#             percentile_approx(Algo_init, 0.5) as median_Algo_init,\
#             min(BRF_over_snow) as min_BRF_over_snow,\
#             max(BRF_over_snow) as max_BRF_over_snow,\
#             percentile_approx(BRF_over_snow, 0.5) as median_BRF_over_snow,\
#             min(BRF_climatology) as min_BRF_climatology,\
#             max(BRF_climatology) as max_BRF_climatology,\
#             percentile_approx(BRF_climatology, 0.5) as median_BRF_climatology,\
#             min(AOD_QA_SC_Mask) as min_AOD_QA_SC_Mask,\
#             max(AOD_QA_SC_Mask) as max_AOD_QA_SC_Mask,\
#             percentile_approx(AOD_QA_SC_Mask, 0.5) as median_AOD_QA_SC_Mask\
#             FROM aod group by grid_id, pm25_reading_date, lat, lon")

# df_aod_grid = df_aod_grid.join(lat_lon_list_df, on=[df_aod_grid.grid_id == lat_lon_list_df.grid_id,  
#                                                    df_aod_grid.pm25_reading_date == lat_lon_list_df.pm25_reading_date],
#                                                 how="left").drop(lat_lon_list_df.grid_id).drop(lat_lon_list_df.pm25_reading_date)

# COMMAND ----------

#previous df_aod_grid
#df_aod_grid.count()
display(df_AOD_with_distance)

# COMMAND ----------

#prev df_aod_grid
df_aod_grid.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in df_aod_grid.dtypes if c not in ('aod_lon_lat_list')]).toPandas()

# COMMAND ----------

df_aod_grid_filtered = spark.sql("SELECT grid_id, pm25_reading_date,
              \
            min(Optical_Depth_047) as min_Optical_Depth_047,\
            max(Optical_Depth_047) as max_Optical_Depth_047,\
            percentile_approx(Optical_Depth_047, 0.5) as median_Optical_Depth_047,\
            min(Optical_Depth_055) as min_Optical_Depth_055,\
            max(Optical_Depth_055) as max_Optical_Depth_055,\
            percentile_approx(Optical_Depth_055, 0.5) as median_Optical_Depth_055,\
            min(AOD_Uncertainty) as min_AOD_Uncertainty,\
            max(AOD_Uncertainty) as max_AOD_Uncertainty,\
            percentile_approx(AOD_Uncertainty, 0.5) as median_AOD_Uncertainty,\
            min(Column_WV) as min_Column_WV,\
            max(Column_WV) as max_Column_WV,\
            percentile_approx(Column_WV, 0.5) as median_Column_WV,\
            min(AOD_QA_Cloud_Mask) as min_AOD_QA_Cloud_Mask,\
            max(AOD_QA_Cloud_Mask) as max_AOD_QA_Cloud_Mask,\
            percentile_approx(AOD_QA_Cloud_Mask, 0.5) as median_AOD_QA_Cloud_Mask,\
            min(AOD_QA_LWS_Mask) as min_AOD_QA_LWS_Mask,\
            max(AOD_QA_LWS_Mask) as max_AOD_QA_LWS_Mask,\
            percentile_approx(AOD_QA_LWS_Mask, 0.5) as median_AOD_QA_LWS_Mask,\
            min(AOD_QA_Adj_Mask) as min_AOD_QA_Adj_Mask,\
            max(AOD_QA_Adj_Mask) as max_AOD_QA_Adj_Mask,\
            percentile_approx(AOD_QA_Adj_Mask, 0.5) as median_AOD_QA_Adj_Mask,\
            min(AOD_Level) as min_AOD_Level,\
            max(AOD_Level) as max_AOD_Level,\
            percentile_approx(AOD_Level, 0.5) as median_AOD_Level,\
            min(Algo_init) as min_Algo_init,\
            max(Algo_init) as max_Algo_init,\
            percentile_approx(Algo_init, 0.5) as median_Algo_init,\
            min(BRF_over_snow) as min_BRF_over_snow,\
            max(BRF_over_snow) as max_BRF_over_snow,\
            percentile_approx(BRF_over_snow, 0.5) as median_BRF_over_snow,\
            min(BRF_climatology) as min_BRF_climatology,\
            max(BRF_climatology) as max_BRF_climatology,\
            percentile_approx(BRF_climatology, 0.5) as median_BRF_climatology,\
            min(AOD_QA_SC_Mask) as min_AOD_QA_SC_Mask,\
            max(AOD_QA_SC_Mask) as max_AOD_QA_SC_Mask,\
            percentile_approx(AOD_QA_SC_Mask, 0.5) as median_AOD_QA_SC_Mask\
            FROM aod_filtered group by grid_id, pm25_reading_date")

df_aod_grid_filtered = df_aod_grid_filtered.join(lat_lon_list_df, on=[df_aod_grid_filtered.grid_id == lat_lon_list_df.grid_id,  
                                                   df_aod_grid_filtered.pm25_reading_date == lat_lon_list_df.pm25_reading_date],
                                                how="left").drop(lat_lon_list_df.grid_id).drop(lat_lon_list_df.pm25_reading_date)

# COMMAND ----------

df_aod_grid_filtered.count()

# COMMAND ----------

df_aod_grid_filtered.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in df_aod_grid_filtered.dtypes if c not in ('aod_lon_lat_list')]).toPandas()

# COMMAND ----------

df_GFS_agg.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Joins

# COMMAND ----------

df_GFS_labels_final_with_distance = df_GFS_labels_final_with_distance.withColumn('distance_rank', sf.dense_rank().over(Window.partitionBy('grid_id', 'datetime').orderBy('distance_to_grid_center')))

# COMMAND ----------

df_AOD_with_distance = df_AOD_with_distance.withColumn('AOD_distance_rank', sf.dense_rank().over(Window.partitionBy('grid_id', 'pm25_reading_date').orderBy('AOD_distance_to_grid_center')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter AOD to 5 closest observations and GFS to single closest

# COMMAND ----------

df_GFS_labels_final_with_distance = df_GFS_labels_final_with_distance.where('distance_rank = 1')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disagg join 

# COMMAND ----------

#Prejoin drop and rename columns for clean join
train_labels_df_PJ = train_labels_df.drop(*['tz', 'wkt', 'pm25_date_d'])
df_AOD_with_distance_PJ = df_AOD_with_distance.drop(*['lon', 'lat', 'aod_reading_end', 'lon-lat-pair', 'observation_point'])
df_GFS_labels_final_with_distance_PJ = df_GFS_labels_final_with_distance.drop(*['value', 'location', 'tz', 'wkt', 'pm25_date_d', 'latitude', 'longitude', 'centroid', 'observation_point', 'distance_to_grid_center', 'distance_rank'])
elevation_df_PJ = elevation_df.drop('polygon_coords')

# COMMAND ----------

#DISAGGREGATED JOIN ON label.date_time = df_AOD.pm25_reading_date & train_labels.pm25_datetime = GFS.pm25_datetime
stage_1 = train_labels_df_PJ.join(df_AOD_with_distance_PJ, on = [train_labels_df_PJ.grid_id == df_AOD_with_distance_PJ.grid_id, 
                                                                train_labels_df_PJ.datetime == df_AOD_with_distance_PJ.pm25_reading_date], how = 'left').drop(df_AOD_with_distance_PJ.grid_id).drop(df_AOD_with_distance_PJ.centroid)

# AOD_GFS_DISAGG_JOIN = df_AOD_with_distance.join(df_GFS_labels_final_with_distance, on = [df_AOD_with_distance.grid_id == df_GFS_labels_final_with_distance.grid_id, 
#                                  df_AOD_with_distance.pm25_reading_date == df_GFS_labels_final_with_distance.datetime, 
#                                  df_AOD_with_distance.AOD_distance_rank == df_GFS_labels_final_with_distance.distance_rank], how = 'left')

# COMMAND ----------

stage_2 = stage_1.join(df_GFS_labels_final_with_distance_PJ, on = [stage_1.grid_id == df_GFS_labels_final_with_distance_PJ.grid_id , 
                                                                  stage_1.pm25_datetime_dt == df_GFS_labels_final_with_distance_PJ.pm25_datetime_dt], how = 'left').drop(df_GFS_labels_final_with_distance_PJ.grid_id).drop(df_GFS_labels_final_with_distance_PJ.datetime).drop(df_GFS_labels_final_with_distance_PJ.pm25_datetime_dt)

# COMMAND ----------

stage_3 = stage_2.join(elevation_df_PJ, on = stage_2.grid_id == elevation_df_PJ.grid_id, how = "left").drop(elevation_df_PJ.grid_id)

# COMMAND ----------

stage_3.write.mode('overwrite').parquet("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_disaggregated.parquet")

# COMMAND ----------

display(stage_3)

# COMMAND ----------

display(elevation_df)

# COMMAND ----------

display(df_GFS_labels_final_with_distance)

# COMMAND ----------

# AOD + GFS
aod_filtered_gfs_joined = df_aod_grid_filtered\
.join(df_GFS_agg, on=[df_aod_grid_filtered.grid_id == df_GFS_agg.grid_id, 
                      df_aod_grid_filtered.pm25_reading_date == df_GFS_agg.pm25_datetime_dt],how="right")\
.drop(df_aod_grid_filtered.grid_id)#.drop(df_GFS_agg.date)
aod_filtered_gfs_joined_with_labels = aod_filtered_gfs_joined

# COMMAND ----------

display(AOD_GFS_DISAGG_JOIN.where('distance_to_grid_center IS NOT NULL')).drop(df_AOD_with_distance.grid_id)

# COMMAND ----------

AOD_GFS_DISAGG_JOIN = AOD_GFS_DISAGG_JOIN.where('distance_to_grid_center IS NOT NULL').drop(df_AOD_with_distance.grid_id)

# COMMAND ----------

display(df_GFS_labels_final_with_distance)


# COMMAND ----------

display(df_AOD_with_distance)

# COMMAND ----------

# MAGIC %md Calculate grid center lat / lon to calculate distances to grid center for each row. 

# COMMAND ----------

!pip install shapely

# COMMAND ----------

display(train_labels_df)
    

# COMMAND ----------

aod_gfs_joined = df_AOD\
.join(df_GFS_agg, on=[df_aod_grid.grid_id == df_GFS_agg.grid_id, 
                      df_aod_grid.pm25_reading_date == df_GFS_agg.pm25_datetime_dt],how="right")\
.drop(df_aod_grid.grid_id)#.drop(df_GFS_agg.date)
aod_gfs_joined_with_labels = aod_gfs_joined

# COMMAND ----------

aod_filtered_gfs_joined.count()

# COMMAND ----------

aod_gfs_joined_with_labels.count()

# COMMAND ----------

#Join with elevation
aod_filtered_gfs_elev_wlabels_joined = aod_filtered_gfs_joined_with_labels\
.join(elevation_df, on=aod_filtered_gfs_joined_with_labels.grid_id == elevation_df.grid_id,how="left")\
.drop(elevation_df.grid_id)

# COMMAND ----------

display(aod_filtered_gfs_elev_wlabels_joined)

# COMMAND ----------

#Join with elevation
aod_gfs_elev_wlabels_joined = aod_gfs_joined_with_labels\
.join(elevation_df, on=aod_gfs_joined_with_labels.grid_id == elevation_df.grid_id,how="left")\
.drop(elevation_df.grid_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join Elevation

# COMMAND ----------

display(aod_gfs_elev_wlabels_joined_disagg)

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined.select([count(when(aod_filtered_gfs_elev_wlabels_joined[c]==-math.inf, c)).alias(c) for (c,c_type) in aod_filtered_gfs_elev_wlabels_joined.dtypes if c not in ('aod_lon_lat_list','pm25_date_d','pm25_datetime_dt')]).toPandas()

# COMMAND ----------

col_for_inf_replacement = ['avg(max('+col+'_new))' for col in cols_00 + cols_06 +cols_12 + cols_18 ]

# COMMAND ----------

col_for_inf_replacement

# COMMAND ----------

col_for_inf_replacement = ['avg(max('+col+'_new))' for col in cols_00 + cols_06 +cols_12 + cols_18 ]for i in col_for_inf_replacement:
    aod_filtered_gfs_elev_wlabels_joined=aod_filtered_gfs_elev_wlabels_joined.withColumn(i,
                when((aod_filtered_gfs_elev_wlabels_joined[i]==-math.inf),None)\
                                                                                             .otherwise(aod_filtered_gfs_elev_wlabels_joined[i]))


# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined.select([count(when(aod_filtered_gfs_elev_wlabels_joined[c]==-math.inf, c)).alias(c) for (c,c_type) in aod_filtered_gfs_elev_wlabels_joined.dtypes if c not in ('aod_lon_lat_list','pm25_date_d','pm25_datetime_dt')]).toPandas()

# COMMAND ----------


aod_gfs_elev_wlabels_joined.select([count(when(aod_gfs_elev_wlabels_joined[c]==-math.inf, c)).alias(c) for (c,c_type) in aod_gfs_elev_wlabels_joined.dtypes if c not in ('aod_lon_lat_list','pm25_date_d','pm25_datetime_dt')]).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write

# COMMAND ----------

aod_gfs_elev_wlabels_joined_disagg = aod_gfs_elev_wlabels_joined_disagg.drop(df_GFS_labels_final_with_distance.centroid)
aod_gfs_elev_wlabels_joined_disagg = aod_gfs_elev_wlabels_joined_disagg.drop(df_GFS_labels_final_with_distance.observation_point)                                                                    

# COMMAND ----------

display(aod_gfs_elev_wlabels_joined_disagg)

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined.write.mode("overwrite").parquet("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined.parquet")

# COMMAND ----------

aod_gfs_elev_wlabels_joined.write.mode("overwrite").parquet("/mnt/capstone/train/aod_gfs_elev_wlabels_joined.parquet")

# COMMAND ----------

display(aod_filtered_gfs_elev_wlabels_joined)

# COMMAND ----------

df_GFS_agg.write.mode("overwrite").parquet("/mnt/capstone/train/df_GFS_agg.parquet")

# COMMAND ----------

trial_aod_gfs_elev_wlabels_joined = spark.read.parquet("/mnt/capstone/train/aod_gfs_elev_wlabels_joined.parquet")

# COMMAND ----------

IMPUTE_VALUES = False

# COMMAND ----------

if IMPUTE_VALUES:  
    df = aod_gfs_joined_with_labels

    imp_test = df.filter(df.min_Optical_Depth_047.isNull())
    imp_train = df.filter(df.min_Optical_Depth_047.isNotNull())


    #Remove troublesome features
    imp_drop_cols = ['aod_lon_lat_list', 'datetime','value','max_Optical_Depth_047','median_Optical_Depth_047']
    #Remove our train label (to impute). 
    imp_label = 'min_Optical_Depth_047'

    imp_features, imp_int_features, imp_str_features = get_cols(imp_train, imp_drop_cols, imp_label)

    imp_train=imp_train.drop(*imp_drop_cols)

# COMMAND ----------

# MAGIC %md Train/Val Split

# COMMAND ----------

if IMPUTE_VALUES:    
    # Add percent rank to aid in cross validation/splitting
    imp_train = imp_train.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('date')))

    # Get 10 folds (20 train + val splits) to generate distribution for statistical testing. 
    # Cache for faster access times. 
    imp_train = imp_train.withColumn("foldNumber", when((imp_train.rank < .66), lit(0)) \
                                               .otherwise(lit(1))).cache()

    imp_train_train = imp_train.where("foldNumber % 2 == 0")
    imp_train_val = imp_train.where("foldNumber % 2 != 0")

# COMMAND ----------

if IMPUTE_VALUES:  
     #For pipeline
    stages = []

    # Process strings into categorical vars first since we can't pass strings to VectorAssembler. 
    si = StringIndexer(
        inputCols=[col for col in imp_str_features], 
        outputCols=[col + "_classVec" for col in imp_str_features], 
        handleInvalid='keep'
    )
    #Add to pipeline. 
    stages += [si]

    #Indexed string features + non-processed int features. 
    assemblerInputs = [c + '_classVec' for c in imp_str_features] + imp_int_features

    #VectorAssembler to vectorize all features. 
    va = VectorAssembler(inputCols=assemblerInputs, outputCol="features_assembled", handleInvalid='keep')
    stages += [va]

    # Now use Vector Indexer to bin our categorical features into N bins. 
    vi = VectorIndexer(inputCol= 'features_assembled', outputCol= 'features_indexed', handleInvalid='keep', maxCategories = 4)
    #Add to pipeline. 
    stages += [vi]

    # Finally standardize features to limit size of feature space (i.e. Categorical Feature 4 has 1049 values which would require max bins = 1050). 
    # TODO: Investigate why VectorIndexer isn't binning into 4 categories already. 
    scaler = StandardScaler(inputCol="features_indexed", outputCol="features",
                            withStd=True, withMean=False)

    stages += [scaler]

    #Define model pipeline. 
    imp_pipeline = Pipeline(stages = stages)

    #Fit transform on train data (excl. validation data to prevent leakage). 
    #Transform both train_data and validation_data. 
    imp_pipelineModel = imp_pipeline.fit(imp_train_train)
    imp_train_df = imp_pipelineModel.transform(imp_train_train)
    imp_validation_df = imp_pipelineModel.transform(imp_train_val)
    
    imp_crossVal_df = imp_validation_df.union(imp_train_df)

# COMMAND ----------

if IMPUTE_VALUES: 
    lr_imp_train_train = imp_train_train.na.drop("any")
    lr_imp_train_val = imp_train_val.na.drop("any")

    lr_imp_pipelineModel = imp_pipeline.fit(lr_imp_train_train)
    lr_imp_train_df = lr_imp_pipelineModel.transform(lr_imp_train_train)
    lr_imp_validation_df = lr_imp_pipelineModel.transform(lr_imp_train_val)
    # The next step is to define the model training stage of the pipeline. 
    # The following command defines a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.
    # If you are running Databricks Runtime for Machine Learning 9.0 ML or above, you can set the `num_workers` parameter to leverage the cluster for distributed training.
    imp_xgb = XgboostRegressor(num_workers=3, featuresCol = "features", labelCol=imp_label)
    #Fit on train and predict (transform) on test. 
    imp_xgb = imp_xgb.fit(lr_imp_train_df)
    imputations = imp_xgb.transform(lr_imp_validation_df)

    #Show a couple predictions
    imputations.select("prediction", imp_label, "features").show(5)

    # Select (prediction, true label) and compute r2. 
    imp_evaluator = RegressionEvaluator(labelCol = imp_label, predictionCol = 'prediction')
    imp_r2 = imp_evaluator.setMetricName('r2').evaluate(imputations)
    imp_rmse = imp_evaluator.setMetricName('rmse').evaluate(imputations)
    imp_mae = imp_evaluator.setMetricName('mae').evaluate(imputations)

    print(f'R2 on test data = {imp_r2}')
    print(f'RMSE on test data = {imp_rmse}')
    print(f'MAE on test data = {imp_mae}')


# COMMAND ----------

if IMPUTE_VALUES: 
    #Define Decision Tree Regressor
    imp_gbt = GBTRegressor(featuresCol = "features", labelCol= imp_label, maxIter = 10)

    #Fit on train and predict (transform) on test. 
    imp_gbt = imp_gbt.fit(lr_imp_train_df)
    imputations = imp_gbt.transform(lr_imp_validation_df)

    #Show a couple predictions
    imputations.select("prediction", imp_label, "features").show(5)

    # Select (prediction, true label) and compute r2. 
    imp_evaluator = RegressionEvaluator(labelCol = imp_label, predictionCol = 'prediction')
    imp_r2 = imp_evaluator.setMetricName('r2').evaluate(imputations)
    imp_rmse = imp_evaluator.setMetricName('rmse').evaluate(imputations)
    imp_mae = imp_evaluator.setMetricName('mae').evaluate(imputations)

    print(f'R-Squared on test data = {imp_r2}')
    print(f'RMSE on test data = {imp_rmse}')
    print(f'MAE on test data = {imp_mae}')

    featureImportance = pd.DataFrame(list(zip(va.getInputCols(), imp_gbt.featureImportances)), columns = ['Feature', 'Importance']).sort_values(by = 'Importance', ascending = False).head(10)

# COMMAND ----------

# featureImportance

# COMMAND ----------

if IMPUTE_VALUES: 
    #Linear Regressor
    #For pipeline
    lr_imp_train_train = imp_train_train.na.drop("any")
    lr_imp_train_val = imp_train_val.na.drop("any")

    lr_imp_pipelineModel = imp_pipeline.fit(lr_imp_train_train)
    lr_imp_train_df = lr_imp_pipelineModel.transform(lr_imp_train_train)
    lr_imp_validation_df = lr_imp_pipelineModel.transform(lr_imp_train_val)

    #Fit on train and predict (transform) on test. 
    imp_lr = LinearRegression(featuresCol = "features", labelCol= imp_label, maxIter = 10)
    imp_lr = imp_lr.fit(lr_imp_train_df)
    imputations = imp_lr.transform(lr_imp_validation_df)

    #Show a couple predictions
    imputations.select("prediction", imp_label, "features").show(5)

    # Select (prediction, true label) and compute r2. 
    imp_evaluator = RegressionEvaluator(labelCol = imp_label, predictionCol = 'prediction')
    imp_r2 = imp_evaluator.setMetricName('r2').evaluate(imputations)
    imp_rmse = imp_evaluator.setMetricName('rmse').evaluate(imputations)
    imp_mae = imp_evaluator.setMetricName('mae').evaluate(imputations)

    print(f'R-Squared on test data = {imp_r2}')
    print(f'RMSE on test data = {imp_rmse}')
    print(f'MAE on test data = {imp_mae}')
    
    featureImportance = pd.DataFrame(list(zip(va.getInputCols(), lr.featureImportances)), columns = ['Feature', 'Importance']).sort_values(by = 'Importance', ascending = False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # TEST Ingestion

# COMMAND ----------

submission_format_csv = spark.read.option("header","true").csv("/mnt/capstone/test/submission_format.csv")

# COMMAND ----------

display(submission_format_csv)

# COMMAND ----------

submission_format_csv = submission_format_csv.withColumn("pm25_date_d", sf.to_date(submission_format_csv['datetime']))
submission_format_csv = submission_format_csv.withColumn("pm25_datetime_dt", sf.to_timestamp(submission_format_csv['datetime']))

# COMMAND ----------

submission_format_csv.count()

# COMMAND ----------

df_GFS = spark.read.parquet("dbfs:/mnt/capstone/train/df_GFS.parquet")

# COMMAND ----------

display(df_GFS.select(['latitude','longitude','grid_id']))

# COMMAND ----------

df_20190901 = spark.read.parquet("/mnt/capstone/train/gfs.0p25.20210901.f006.parquet")

# COMMAND ----------

!aws s3 cp s3://capstone-particulate-storage/GFS/201801/ train/GFS/201801/ --no-sign-request --recursive

# COMMAND ----------

df_GFS_201801 = directory_to_sparkDF(directory = 'train/GFS/201801/')

# COMMAND ----------

df_GFS = df_GFS.unionAll(df_20190901)

# COMMAND ----------

df_GFS = df_GFS.unionAll(df_GFS_201801)

# COMMAND ----------

df_GFS.write.mode("overwrite").parquet("dbfs:/mnt/capstone/train/df_GFS.parquet")

# COMMAND ----------

df_GFS = spark.read.parquet("dbfs:/mnt/capstone/train/df_GFS.parquet")

# COMMAND ----------

df_GFS = df_GFS.withColumn("gfs_date_d", sf.to_date(df_GFS['date']))
df_GFS = df_GFS.withColumn("latitude", df_GFS['latitude'].cast(DoubleType()))
df_GFS = df_GFS.withColumn("longitude", df_GFS['longitude'].cast(DoubleType()))
df_GFS = df_GFS.withColumnRenamed('grid_id', 'grid_id_gfs')
df_GFS = df_GFS.withColumnRenamed('date', 'gfs_date')

# COMMAND ----------

df_GFS.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in df_GFS.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','gfs_date_d')]).toPandas()

# COMMAND ----------

df_GFS_test_join = submission_format_csv.join(df_GFS, on=[submission_format_csv.grid_id == df_GFS.grid_id_gfs,
                                                 ((submission_format_csv.pm25_date_d == df_GFS.gfs_date_d) 
                                                   | (sf.date_add(submission_format_csv.pm25_date_d,1) == df_GFS.gfs_date_d)),  
                                                 ],
                                                how="left")

# COMMAND ----------

cols_00 = [col for col in df_GFS_test_join.columns if '00' in col]
cols_06 = [col for col in df_GFS_test_join.columns if '06' in col]
cols_12 = [col for col in df_GFS_test_join.columns if '12' in col]
cols_18 = [col for col in df_GFS_test_join.columns if '18' in col]

# COMMAND ----------

for col in cols_00:
    df_GFS_test_join = df_GFS_test_join\
    .withColumn(col+'_new',when((((df_GFS_test_join.pm25_date_d == df_GFS_test_join.gfs_date_d) 
                                          & (sf.hour(df_GFS_test_join.pm25_datetime_dt)<0)) | 
                                ((df_GFS_test_join.pm25_date_d < df_GFS_test_join.gfs_date_d) 
                                         & (sf.hour(df_GFS_test_join.pm25_datetime_dt)>=0)))
        ,df_GFS_test_join[col]).otherwise(-math.inf))\
    .drop(df_GFS_test_join[col])

# COMMAND ----------

for col in cols_06:
    df_GFS_test_join = df_GFS_test_join\
    .withColumn(col+'_new',when((((df_GFS_test_join.pm25_date_d == df_GFS_test_join.gfs_date_d) 
                                          & (sf.hour(df_GFS_test_join.pm25_datetime_dt)<6)) | 
                                ((df_GFS_test_join.pm25_date_d < df_GFS_test_join.gfs_date_d) 
                                         & (sf.hour(df_GFS_test_join.pm25_datetime_dt)>=6)))
        ,df_GFS_test_join[col]).otherwise(-math.inf))\
    .drop(df_GFS_test_join[col])

# COMMAND ----------

for col in cols_12:
    df_GFS_test_join = df_GFS_test_join\
    .withColumn(col+'_new',when((((df_GFS_test_join.pm25_date_d == df_GFS_test_join.gfs_date_d) 
                                          & (sf.hour(df_GFS_test_join.pm25_datetime_dt)<12)) | 
                                ((df_GFS_test_join.pm25_date_d < df_GFS_test_join.gfs_date_d) 
                                         & (sf.hour(df_GFS_test_join.pm25_datetime_dt)>=12)))
        ,df_GFS_test_join[col]).otherwise(-math.inf))\
    .drop(df_GFS_test_join[col])

# COMMAND ----------

for col in cols_18:
    df_GFS_test_join = df_GFS_test_join\
    .withColumn(col+'_new',when((((df_GFS_test_join.pm25_date_d == df_GFS_test_join.gfs_date_d) 
                                          & (sf.hour(df_GFS_test_join.pm25_datetime_dt)<18)) | 
                                ((df_GFS_test_join.pm25_date_d < df_GFS_test_join.gfs_date_d) 
                                         & (sf.hour(df_GFS_test_join.pm25_datetime_dt)>=18)))
        ,df_GFS_test_join[col]).otherwise(-math.inf))\
    .drop(df_GFS_test_join[col])

# COMMAND ----------

display(df_GFS_test_join_final)

# COMMAND ----------

display(train_labels_df_centr)

# COMMAND ----------

df_GFS_test_join_final = df_GFS_test_join.groupby(["datetime","grid_id","value","pm25_date_d","pm25_datetime_dt","latitude","longitude"]).max()

# COMMAND ----------

df_GFS_test_join_final = df_GFS_test_join_final.join(train_labels_df_centr, on = df_GFS_test_join_final.grid_id == train_labels_df_centr.grid_id, how = 'left').drop(train_labels_df_centr.grid_id)

# COMMAND ----------

df_GFS_test_join_final = df_GFS_test_join_final.drop(*['max(latitude)', 'max(longitude)'])

# COMMAND ----------

df_GFS_test_join_final = df_GFS_test_join_final.withColumn('observation_point', get_point(df_GFS_test_join_final.longitude, df_GFS_test_join_final.latitude))

# COMMAND ----------

df_GFS_test_join_final = df_GFS_test_join_final.withColumn('distance_to_grid_center', calculate_distance(df_GFS_test_join_final.centroid, df_GFS_test_join_final.observation_point))

# COMMAND ----------

# # Join elevation here since we appear to be missing in AOD
# df_GFS_test_join_final = df_GFS_test_join_final.join(elevation_df, on = df_GFS_test_join_final.grid_id == elevation_df.grid_id, how = "left").drop(df_GFS_test_join_final.grid_id)


# COMMAND ----------

display(df_GFS_test_join_final)

# COMMAND ----------

df_GFS_test_join_final.select([count(when(df_GFS_test_join_final[c]==-math.inf, c)).alias(c) for (c,c_type) in df_GFS_test_join_final.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

df_GFS_test_join_final.write.mode("overwrite").parquet("/mnt/capstone/test/df_GFS_test.parquet")

# COMMAND ----------

df_GFS_agg = df_GFS_test_join_final.groupBy("grid_id","datetime","value","pm25_date_d","pm25_datetime_dt").mean()

# COMMAND ----------

df_GFS_agg.select([count(when(df_GFS_agg[c]==-math.inf, c)).alias(c) for (c,c_type) in df_GFS_agg.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

df_GFS_agg.write.mode("overwrite").parquet("/mnt/capstone/test/df_GFS_agg.parquet")

# COMMAND ----------

display(df_GFS_agg)

# COMMAND ----------

df_GFS_agg.select("grid_id").where(df_GFS_agg["avg(max(t_surface00_new))"]==-math.inf).distinct().show()

# COMMAND ----------

df_GFS_agg.select("datetime").where(df_GFS_agg["avg(max(t_surface00_new))"]==-math.inf).distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## AOD 

# COMMAND ----------

# # Download AOD data from s3 bucket to Databricsk workspace. 
!aws s3 cp s3://particulate-articulate-capstone/test/aod test/AOD/ --no-sign-request --recursive

# COMMAND ----------

AODCustomSchema = StructType([
        StructField("index", LongType(), True),
        StructField("lon", FloatType(), True),
        StructField("lat", FloatType(), True),
        StructField("Optical_Depth_047", FloatType(), True),
        StructField("Optical_Depth_055", FloatType(), True),
        StructField("AOD_Uncertainty", FloatType(), True),
        StructField("FineModeFraction", FloatType(), True),
        StructField("Column_WV", FloatType(), True),
        StructField("Injection_Height", FloatType(), True),
        StructField("AOD_QA", IntegerType(), True),
        StructField("grid_id", StringType(), True),
        StructField("aod_reading_end", StringType(), True),
        StructField("location", StringType(), True)])
df_AOD = directory_to_sparkDF(directory = 'test/AOD/', schema=AODCustomSchema)

# COMMAND ----------

df_AOD = df_AOD.withColumn("aod_reading_end_dt", sf.to_timestamp(df_AOD['aod_reading_end']))

# COMMAND ----------

display(df_AOD)

# COMMAND ----------

display(submission_format_csv)

# COMMAND ----------

df_AOD = df_AOD.drop(*['index']).distinct()

# COMMAND ----------

display(df_AOD_test)

# COMMAND ----------

df_AOD_test = df_AOD.join(submission_format_csv, on=[submission_format_csv.grid_id == df_AOD.grid_id, 
                                                ((df_AOD.aod_reading_end_dt <= (sf.date_add(submission_format_csv.pm25_datetime_dt,1))) 
                                                   & (df_AOD.aod_reading_end_dt >= (sf.date_sub(submission_format_csv.pm25_datetime_dt, 1))))],
                                                how="left").drop(submission_format_csv.grid_id)

# COMMAND ----------

#Drop AOD observations that fall outside valid date range (24hrs) for a grid, datetime. 
df_AOD_test = df_AOD_test.where('lon IS NOT NULL')
df_AOD_test = df_AOD_test.where('datetime IS NOT NULL')

# COMMAND ----------

display(df_AOD_test)

# COMMAND ----------

df_AOD_test = df_AOD_test.drop(*['aod_reading_end', 'aod_reading_end_dt'])
df_AOD_test = df_AOD_test.groupBy(*['grid_id', 'lon', 'lat', 'location', 'datetime', 'pm25_datetime_dt']).mean()

# COMMAND ----------

df_AOD_test = df_AOD_test.drop(*['avg(lon)', 'avg(lat)'])

# COMMAND ----------

df_AOD_test = df_AOD_test.withColumnRenamed('avg(AOD_QA)', 'AOD_QA')
df_AOD_test = df_AOD_test.withColumn('AOD_QA', df_AOD_test.AOD_QA.cast('int'))

# COMMAND ----------

df_AOD_test.count()

# COMMAND ----------

df_AOD_test.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in df_AOD_test.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d','aod_reading_end_dt')]).toPandas()

# COMMAND ----------

#Recast columns
df_AOD_test = df_AOD_test.withColumn("AOD_qa_str",udf_qa_format(df_AOD_test["AOD_QA"]))
df_AOD_test = df_AOD_test.withColumn('AOD_QA_Cloud_Mask_str', substring('AOD_qa_str', 0,3))\
    .withColumn('AOD_QA_LWS_Mask_str', substring('AOD_qa_str', 3,2))\
    .withColumn('AOD_QA_Adj_Mask_str', substring('AOD_qa_str', 5,3))\
    .withColumn('AOD_Level_str', substring('AOD_qa_str', 8,1))\
    .withColumn('Algo_init_str', substring('AOD_qa_str', 9,1))\
    .withColumn('BRF_over_snow_str', substring('AOD_qa_str', 10,1))\
    .withColumn('BRF_climatology_str', substring('AOD_qa_str', 11,1))\
    .withColumn('AOD_QA_SC_Mask_str', substring('AOD_qa_str', 12,3))
df_AOD_test = df_AOD_test.withColumn('AOD_QA_Cloud_Mask', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_LWS_Mask', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_Adj_Mask', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_Level', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('Algo_init', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('BRF_over_snow', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('BRF_climatology', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_SC_Mask', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))

qa_masks_str_cols = ('AOD_QA_Cloud_Mask_str',
                    'AOD_QA_LWS_Mask_str',
                    'AOD_QA_Adj_Mask_str',
                    'AOD_Level_str',
                    'Algo_init_str',
                    'BRF_over_snow_str',
                    'BRF_climatology_str',
                    'AOD_QA_SC_Mask_str')
df_AOD_test.drop(*qa_masks_str_cols)

df_AOD_test.registerTempTable("aod")

#AOD Lat-Lon pairs as list
df_AOD_test = df_AOD_test.withColumn("lon-lat-pair", sf.concat_ws('_',df_AOD_test.lon,df_AOD_test.lat))

# lat_lon_list_df = df_AOD_test.groupBy("grid_id","aod_reading_end")\
# .agg(sf.collect_list("lon-lat-pair").alias("aod_lon_lat_list"))

df_AOD_test = df_AOD_test.withColumn('AOD_QA_Cloud_Mask', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_LWS_Mask', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_Adj_Mask', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_Level', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('Algo_init', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('BRF_over_snow', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('BRF_climatology', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))\
    .withColumn('AOD_QA_SC_Mask', udf_mask_int(df_AOD_test["AOD_QA_Cloud_Mask_str"]))

# COMMAND ----------

df_AOD_test = df_AOD_test.join(train_labels_df_centr, on = 'grid_id', how = 'left')
df_AOD_test = df_AOD_test.withColumn('observation_point', get_point(df_AOD_test.lon, df_AOD_test.lat))
df_AOD_with_distance_test = df_AOD_test.withColumn('AOD_distance_to_grid_center', calculate_distance(df_AOD_test.centroid, df_AOD_test.observation_point))

# COMMAND ----------

display(df_AOD_with_distance_test)

# COMMAND ----------

display(elevation_df)

# COMMAND ----------

display(df_GFS_test_join_final)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Test filter by distance

# COMMAND ----------

df_GFS_test_join_final = df_GFS_test_join_final.withColumn('distance_rank', sf.dense_rank().over(Window.partitionBy('grid_id', 'datetime').orderBy('distance_to_grid_center')))
df_AOD_with_distance_test = df_AOD_with_distance_test.withColumn('AOD_distance_rank', sf.dense_rank().over(Window.partitionBy('grid_id', 'datetime').orderBy('AOD_distance_to_grid_center')))

# COMMAND ----------

df_GFS_test_join_final = df_GFS_test_join_final.where('distance_rank = 1')

# COMMAND ----------

#Prejoin drop and rename columns for clean join
df_AOD_with_distance_test_PJ = df_AOD_with_distance_test.drop(*['aod_reading_end', 'lon-lat-pair', 'observation_point', 'lon', 'lat'])
df_GFS_test_join_final_PJ = df_GFS_test_join_final.drop(*['value', 'location', 'tz', 'pm25_date_d', 'latitude', 'longitude', 'centroid', 'observation_point', 'distance_to_grid_center', 'distance_rank'])
elevation_df_PJ = elevation_df.drop(*['polygon_cords'])

# COMMAND ----------

stage_1 = submission_format_csv.join(df_AOD_with_distance_test_PJ, on = ['grid_id', 'datetime'], how = 'left')
# stage_1 = stage_1.drop(df_AOD_with_distance_test_PJ.grid_id)
stage_1 = stage_1.drop(df_AOD_with_distance_test_PJ.pm25_datetime_dt)


# COMMAND ----------

stage_2 = stage_1.join(df_GFS_test_join_final_PJ, on = ['grid_id', 'datetime'], how = "left")
stage_2 = stage_2.drop(df_GFS_test_join_final_PJ.pm25_datetime_dt)

# COMMAND ----------

stage_3 = stage_2.join(elevation_df_PJ, on = 'grid_id', how = "left")
# stage_3 = stage_3.drop(elevation_df_PJ.grid_id)
stage_3 = stage_3.drop(elevation_df_PJ.polygon_coords)

# COMMAND ----------

#Finally we have 463 where location is null so we will join onto grid_id (for which we have in all observations). 
location_df = train_labels_df.select('grid_id', 'location').distinct()

# COMMAND ----------

stage_4 = stage_3.join(location_df, on = "grid_id", how = "left").drop(stage_3.location)

# COMMAND ----------

stage_4.write.mode("overwrite").parquet("/mnt/capstone/test/aod_gfs_elev_joined_disagg.parquet")

# COMMAND ----------

display(stage_4)

# COMMAND ----------

submission_format_csv.select('grid_id', 'datetime').distinct().count()

# COMMAND ----------

stage_4.select('grid_id', 'datetime').distinct().count()

# COMMAND ----------

stage_4.where('location IS NULL').count()

# COMMAND ----------

train = spark.read.parquet('dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_disaggregated.parquet')

# COMMAND ----------

display(train)

# COMMAND ----------

grid_metadata = spark.read.csv("/mnt/capstone/test/grid_metadata.csv",header=True)

# COMMAND ----------

aod_gfs_elev_wlabels_joined_disagg.write.parquet("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_disaggregated.parquet")

# COMMAND ----------

df_AOD_filtered_test = df_AOD_test\
.withColumn('Optical_Depth_047_new',when(df_AOD_test.AOD_QA_Cloud_Mask == 1,df_AOD_test.Optical_Depth_047).otherwise(None))\
.drop(df_AOD_test.Optical_Depth_047)\
.withColumnRenamed('Optical_Depth_047_new', 'Optical_Depth_047')\
.withColumn('Optical_Depth_055_new',when(df_AOD_test.AOD_QA_Cloud_Mask == 1,df_AOD_test.Optical_Depth_055).otherwise(None))\
.drop(df_AOD_test.Optical_Depth_055)\
.withColumnRenamed('Optical_Depth_055_new', 'Optical_Depth_055')


# COMMAND ----------

df_AOD_test.registerTempTable("aod_test")
df_AOD_filtered_test.registerTempTable("aod_filtered_test")

# COMMAND ----------

df_aod_grid_test = spark.sql("SELECT grid_id, pm25_datetime_dt,\
            min(Optical_Depth_047) as min_Optical_Depth_047,\
            max(Optical_Depth_047) as max_Optical_Depth_047,\
            percentile_approx(Optical_Depth_047, 0.5) as median_Optical_Depth_047,\
            min(Optical_Depth_055) as min_Optical_Depth_055,\
            max(Optical_Depth_055) as max_Optical_Depth_055,\
            percentile_approx(Optical_Depth_055, 0.5) as median_Optical_Depth_055,\
            min(AOD_Uncertainty) as min_AOD_Uncertainty,\
            max(AOD_Uncertainty) as max_AOD_Uncertainty,\
            percentile_approx(AOD_Uncertainty, 0.5) as median_AOD_Uncertainty,\
            min(Column_WV) as min_Column_WV,\
            max(Column_WV) as max_Column_WV,\
            percentile_approx(Column_WV, 0.5) as median_Column_WV,\
            min(AOD_QA_Cloud_Mask) as min_AOD_QA_Cloud_Mask,\
            max(AOD_QA_Cloud_Mask) as max_AOD_QA_Cloud_Mask,\
            percentile_approx(AOD_QA_Cloud_Mask, 0.5) as median_AOD_QA_Cloud_Mask,\
            min(AOD_QA_LWS_Mask) as min_AOD_QA_LWS_Mask,\
            max(AOD_QA_LWS_Mask) as max_AOD_QA_LWS_Mask,\
            percentile_approx(AOD_QA_LWS_Mask, 0.5) as median_AOD_QA_LWS_Mask,\
            min(AOD_QA_Adj_Mask) as min_AOD_QA_Adj_Mask,\
            max(AOD_QA_Adj_Mask) as max_AOD_QA_Adj_Mask,\
            percentile_approx(AOD_QA_Adj_Mask, 0.5) as median_AOD_QA_Adj_Mask,\
            min(AOD_Level) as min_AOD_Level,\
            max(AOD_Level) as max_AOD_Level,\
            percentile_approx(AOD_Level, 0.5) as median_AOD_Level,\
            min(Algo_init) as min_Algo_init,\
            max(Algo_init) as max_Algo_init,\
            percentile_approx(Algo_init, 0.5) as median_Algo_init,\
            min(BRF_over_snow) as min_BRF_over_snow,\
            max(BRF_over_snow) as max_BRF_over_snow,\
            percentile_approx(BRF_over_snow, 0.5) as median_BRF_over_snow,\
            min(BRF_climatology) as min_BRF_climatology,\
            max(BRF_climatology) as max_BRF_climatology,\
            percentile_approx(BRF_climatology, 0.5) as median_BRF_climatology,\
            min(AOD_QA_SC_Mask) as min_AOD_QA_SC_Mask,\
            max(AOD_QA_SC_Mask) as max_AOD_QA_SC_Mask,\
            percentile_approx(AOD_QA_SC_Mask, 0.5) as median_AOD_QA_SC_Mask\
            FROM aod_test group by grid_id, pm25_datetime_dt")


# COMMAND ----------

df_aod_filtered_grid_test = spark.sql("SELECT grid_id, pm25_datetime_dt,\
            min(Optical_Depth_047) as min_Optical_Depth_047,\
            max(Optical_Depth_047) as max_Optical_Depth_047,\
            percentile_approx(Optical_Depth_047, 0.5) as median_Optical_Depth_047,\
            min(Optical_Depth_055) as min_Optical_Depth_055,\
            max(Optical_Depth_055) as max_Optical_Depth_055,\
            percentile_approx(Optical_Depth_055, 0.5) as median_Optical_Depth_055,\
            min(AOD_Uncertainty) as min_AOD_Uncertainty,\
            max(AOD_Uncertainty) as max_AOD_Uncertainty,\
            percentile_approx(AOD_Uncertainty, 0.5) as median_AOD_Uncertainty,\
            min(Column_WV) as min_Column_WV,\
            max(Column_WV) as max_Column_WV,\
            percentile_approx(Column_WV, 0.5) as median_Column_WV,\
            min(AOD_QA_Cloud_Mask) as min_AOD_QA_Cloud_Mask,\
            max(AOD_QA_Cloud_Mask) as max_AOD_QA_Cloud_Mask,\
            percentile_approx(AOD_QA_Cloud_Mask, 0.5) as median_AOD_QA_Cloud_Mask,\
            min(AOD_QA_LWS_Mask) as min_AOD_QA_LWS_Mask,\
            max(AOD_QA_LWS_Mask) as max_AOD_QA_LWS_Mask,\
            percentile_approx(AOD_QA_LWS_Mask, 0.5) as median_AOD_QA_LWS_Mask,\
            min(AOD_QA_Adj_Mask) as min_AOD_QA_Adj_Mask,\
            max(AOD_QA_Adj_Mask) as max_AOD_QA_Adj_Mask,\
            percentile_approx(AOD_QA_Adj_Mask, 0.5) as median_AOD_QA_Adj_Mask,\
            min(AOD_Level) as min_AOD_Level,\
            max(AOD_Level) as max_AOD_Level,\
            percentile_approx(AOD_Level, 0.5) as median_AOD_Level,\
            min(Algo_init) as min_Algo_init,\
            max(Algo_init) as max_Algo_init,\
            percentile_approx(Algo_init, 0.5) as median_Algo_init,\
            min(BRF_over_snow) as min_BRF_over_snow,\
            max(BRF_over_snow) as max_BRF_over_snow,\
            percentile_approx(BRF_over_snow, 0.5) as median_BRF_over_snow,\
            min(BRF_climatology) as min_BRF_climatology,\
            max(BRF_climatology) as max_BRF_climatology,\
            percentile_approx(BRF_climatology, 0.5) as median_BRF_climatology,\
            min(AOD_QA_SC_Mask) as min_AOD_QA_SC_Mask,\
            max(AOD_QA_SC_Mask) as max_AOD_QA_SC_Mask,\
            percentile_approx(AOD_QA_SC_Mask, 0.5) as median_AOD_QA_SC_Mask\
            FROM aod_filtered_test group by grid_id, pm25_datetime_dt")


# COMMAND ----------

df_aod_filtered_grid_test.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in df_aod_filtered_grid_test.dtypes if c not in ('pm25_date_d','pm25_datetime_dt')]).toPandas()

# COMMAND ----------

df_aod_filtered_grid_test.count()

# COMMAND ----------

df_aod_grid_test.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in df_aod_grid_test.dtypes if c not in ('pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

df_aod_grid_test.count()

# COMMAND ----------

#AOD+GFS
aod_filtered_gfs_joined = df_aod_filtered_grid_test\
.join(df_GFS_agg, on=[df_aod_filtered_grid_test.grid_id == df_GFS_agg.grid_id, 
                      df_aod_filtered_grid_test.pm25_datetime_dt == df_GFS_agg.pm25_datetime_dt],how="right")\
.drop(df_aod_filtered_grid_test.grid_id).drop(df_GFS_agg.pm25_datetime_dt)
#aod_filtered_gfs_joined_with_labels = aod_filtered_gfs_joined

# COMMAND ----------

aod_filtered_gfs_elev_joined = aod_filtered_gfs_joined\
.join(elevation_df, on=aod_filtered_gfs_joined.grid_id == elevation_df.grid_id,how="left")\
.drop(elevation_df.grid_id)

# COMMAND ----------

col_for_inf_replacement = ['avg(max('+col+'_new))' for col in cols_00 + cols_06 +cols_12 + cols_18 ]
for i in col_for_inf_replacement:
    aod_filtered_gfs_elev_joined=aod_filtered_gfs_elev_joined.withColumn(i,
                when((aod_filtered_gfs_elev_joined[i]==-math.inf),None).otherwise(aod_filtered_gfs_elev_joined[i]))

# COMMAND ----------

aod_filtered_gfs_elev_joined.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in aod_filtered_gfs_elev_joined.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

aod_filtered_gfs_elev_joined.select([count(when(aod_filtered_gfs_elev_joined[c]==-math.inf, c)).alias(c) for (c,c_type) in aod_filtered_gfs_elev_joined.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

aod_filtered_gfs_elev_joined.write.mode("overwrite").parquet("/mnt/capstone/test/aod_filtered_gfs_elev_joined.parquet")

# COMMAND ----------

display(aod_filtered_gfs_elev_joined)

# COMMAND ----------

aod_filtered_gfs_elev_joined.count()

# COMMAND ----------

#AOD + GFS
aod_gfs_joined = df_aod_grid_test\
.join(df_GFS_agg, on=[df_aod_grid_test.grid_id == df_GFS_agg.grid_id, 
                      df_aod_grid_test.pm25_datetime_dt == df_GFS_agg.pm25_datetime_dt],how="right")\
.drop(df_aod_grid_test.grid_id).drop(df_aod_grid_test.pm25_datetime_dt )

# COMMAND ----------

#Join with elevation
aod_gfs_elev_joined = aod_gfs_joined\
.join(elevation_df, on=aod_gfs_joined.grid_id == elevation_df.grid_id,how="left")\
.drop(elevation_df.grid_id)

# COMMAND ----------

col_for_inf_replacement = ['avg(max('+col+'_new))' for col in cols_00 + cols_06 +cols_12 + cols_18 ]
for i in col_for_inf_replacement:
    aod_gfs_elev_joined=aod_gfs_elev_joined.withColumn(i,
                when((aod_gfs_elev_joined[i]==-math.inf),None).otherwise(aod_gfs_elev_joined[i]))

# COMMAND ----------

display(aod_gfs_elev_joined)

# COMMAND ----------

aod_gfs_elev_joined.count()

# COMMAND ----------

aod_gfs_elev_joined.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in aod_gfs_elev_joined.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

aod_gfs_elev_joined.select([count(when(aod_gfs_elev_joined[c]==-math.inf, c)).alias(c) for (c,c_type) in aod_gfs_elev_joined.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

aod_gfs_elev_joined.write.mode("overwrite").parquet("/mnt/capstone/test/aod_gfs_elev_joined.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding Location 

# COMMAND ----------

aod_gfs_elev_joined = spark.read.parquet("/mnt/capstone/test/aod_gfs_elev_joined.parquet")

# COMMAND ----------

aod_filtered_gfs_elev_joined = spark.read.parquet("/mnt/capstone/test/aod_filtered_gfs_elev_joined.parquet")

# COMMAND ----------

grid_metadata = spark.read.csv("/mnt/capstone/test/grid_metadata.csv",header=True)

# COMMAND ----------

aod_gfs_elev_joined = aod_gfs_elev_joined.join(grid_metadata, on=[aod_gfs_elev_joined.grid_id==grid_metadata.grid_id], how="left").drop(grid_metadata.grid_id)

# COMMAND ----------

aod_gfs_elev_joined.count()

# COMMAND ----------

aod_gfs_elev_joined.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in aod_gfs_elev_joined.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

aod_gfs_elev_joined.write.mode("overwrite").parquet("/mnt/capstone/test/aod_gfs_elev_joined.parquet")

# COMMAND ----------

aod_filtered_gfs_elev_joined = aod_filtered_gfs_elev_joined.join(grid_metadata, on=[aod_filtered_gfs_elev_joined.grid_id==grid_metadata.grid_id], how="left").drop(grid_metadata.grid_id)

# COMMAND ----------

aod_filtered_gfs_elev_joined.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in aod_filtered_gfs_elev_joined.dtypes if c not in ('aod_lon_lat_list','pm25_datetime_dt','pm25_date_d')]).toPandas()

# COMMAND ----------

aod_filtered_gfs_elev_joined.count()

# COMMAND ----------

aod_filtered_gfs_elev_joined.write.mode("overwrite").parquet("/mnt/capstone/test/aod_filtered_gfs_elev_joined.parquet")

# COMMAND ----------

# MAGIC %md 
# MAGIC # Adding MISR to aggregated final df

# COMMAND ----------

#All AOD
aod_gfs_elev_wlabels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_gfs_elev_wlabels_joined.parquet")

#Clean AOD 
aod_filtered_gfs_elev_wlabels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined.parquet")


# COMMAND ----------

!aws s3 cp s3://particulate-articulate-capstone/train/misr train/misr/ --no-sign-request --recursive

# COMMAND ----------

df_misr_train = directory_to_sparkDF(directory = 'train/misr')

# COMMAND ----------

df_misr_train.write.parquet("/mnt/capstone/train/all_misr_train.parquet")

# COMMAND ----------

aod_gfs_elev_misr_wlabels = aod_gfs_elev_wlabels.join(df_misr_train,
                                                      on[aod_gfs_elev_wlabels["pm25_reading_date"]==df_misr_train[]],
                                                      how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extra EDA

# COMMAND ----------

#All AOD
aod_gfs_elev_wlabels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_gfs_elev_wlabels_joined.parquet")

#Clean AOD 
aod_filtered_gfs_elev_wlabels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined.parquet")

# COMMAND ----------

display(aod_filtered_gfs_elev_wlabels)

# COMMAND ----------

aod_filtered_gfs_elev_wlabels.corr('min_elevation','value')

# COMMAND ----------

new_df = aod_filtered_gfs_elev_wlabels.withColumn('elev_diff', aod_filtered_gfs_elev_wlabels.max_elevation - aod_filtered_gfs_elev_wlabels.min_elevation)

# COMMAND ----------

new_df.corr('elev_diff','value')

# COMMAND ----------

