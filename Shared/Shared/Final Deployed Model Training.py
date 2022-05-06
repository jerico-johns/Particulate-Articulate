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
# from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
# from pyspark.ml.regression import DecisionTreeRegressor
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.regression import GBTRegressor
#from pyspark.ml.regression import LinearRegression
# from sparkdl.xgboost import XgboostRegressor
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

# MAGIC %md Notebook setup. 

# COMMAND ----------

access_key = "AKIA33CENIT6JFRXRNOE"
secret_key = dbutils.secrets.get(scope = "capstone-s3", key = access_key)
encoded_secret_key = secret_key.replace("/", "%2F")
aws_bucket_name = "capstone-particulate-storage"
mount_name = "capstone"

# dbutils.fs.mount("s3a://%s:%s@%s" % (access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)
display(dbutils.fs.ls("/mnt/%s" % mount_name))

# COMMAND ----------

print(secret_key)

# COMMAND ----------

#Set desired stages to True to run. 
INIT_DATASETS = False
RUN_EDA = False
USE_IMPUTE = False
GFS_TIME_AVG = False
FEATURE_ENG_TIME = True
FEATURE_ENG_AVG = False
FEATURE_ENG_TRAILING = False
FEATURE_ENG_DELTA = False
IMPUTE_VALUES = False
RUN_BASELINES = False
RUN_CROSSVAL = False
RUN_SPLIT = False
UPDATE_INF = True
SELECT_FEATURES = False

# COMMAND ----------

if INIT_DATASETS == False: 
    # All AOD with MISR
    #aod_gfs_joined_with_labels_read= spark.read.parquet("dbfs:/mnt/capstone/train/aod_gfs_elev_misr_wlabels.parquet")
#     # All AOD. 
    aod_gfs_joined_with_labels_read = spark.read.parquet("/mnt/capstone/lat_lon_level_gfs_elev_aod_labels.parquet")
    # Clean AOD 
#     aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined.parquet")
#     # Imputed AOD 
    #aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_wimputations.parquet")
    

# COMMAND ----------

#Try grouping by grid_id
# aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.where('grid_id IS NOT NULL').groupBy(['date_utc', 'datetime_utc', 'grid_id']).mean()

# COMMAND ----------

display(aod_gfs_joined_with_labels_read)

# COMMAND ----------

# #Replace the negative (measurement error) and outlier positive values (> 600) with 0 and 600 respectively. 
# aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.where(aod_gfs_joined_with_labels_read['value'] > 0) 
# aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.where(aod_gfs_joined_with_labels_read['value'] <= 600)


# COMMAND ----------

cols_00 = [col for col in aod_gfs_joined_with_labels_read.columns if '00' in col]
cols_06 = [col for col in aod_gfs_joined_with_labels_read.columns if '06' in col]
cols_12 = [col for col in aod_gfs_joined_with_labels_read.columns if '12' in col]
cols_18 = [col for col in aod_gfs_joined_with_labels_read.columns if '18' in col]
# if GFS_TIME_AVG:
#     def avg_gfs_time_cols(array):
#         if array:
#             return sum(filter(None, array))/len(array)
#         else:
#             return None 
#     avg_cols = udf(lambda array: sum(filter(None, array))/len(array), DoubleType())

#     #averaging gfs time based features

#     for i in range(len(cols_00)):
#         colName = cols_00[i].replace("00","")
#         aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.withColumn(colName, 
#                                                                                      avg_cols(sf.array(cols_00[i],cols_06[i],
#                                                                                                        cols_12[i],cols_18[i])))
#     aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.drop(*cols_00)
#     aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.drop(*cols_06)
#     aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.drop(*cols_12)
#     aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.drop(*cols_18)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.) If (INIT_DATASET), Download data from s3, format, and join. 

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

if INIT_DATASETS: 
#     # Comment out after first run. 
#     !curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#     !unzip awscliv2.zip
#     !sudo ./aws/install
#     !aws --version

    #Download GFS data from s3 bucket to Databricks workspace. We will then load these files into a Spark DF. 
    !aws s3 cp s3://capstone-particulate-storage/GFS/ train/GFS/ --no-sign-request --recursive
    #Download GFS data from s3 bucket to Databricks workspace. We will then load these files into a Spark DF. 
    !aws s3 cp s3://capstone-particulate-storage/geos/ train/GEOS/ --no-sign-request --recursive
    #Download AOD data from s3 bucket to Databricsk workspace. 
    !aws s3 cp s3://particulate-articulate-capstone/aod/ train/AOD/ --no-sign-request --recursive
    #Download Elevation data
    !aws s3 cp s3://particulate-articulate-capstone/elevation.parquet train/elevation/ --no-sign-request

# COMMAND ----------

if INIT_DATASETS:    
    # Download training labels. 
    file='meta_data/train_labels_grid.csv'
    bucket='capstone-particulate-storage'

    #buffer = io.BytesIO()
    s3_read_client = boto3.client('s3')
    s3_tl_obj = s3_read_client.get_object(Bucket= bucket, Key= file)
    #s3_tl_obj.download_fileobj(buffer)
    train_labels = pd.read_csv(s3_tl_obj['Body'],delimiter='|',header=0)
    train_labels_df = spark.createDataFrame(train_labels)
    train_labels_df = train_labels_df.withColumn("date", date_format(train_labels_df['datetime'],"yyyy-MM-dd"))

# COMMAND ----------

# MAGIC %md Now open downloaded GFS files and union into a full Spark Dataframe. 

# COMMAND ----------

if INIT_DATASETS:    
    drop_cols = ['landn_surface00', 'landn_surface06', 'landn_surface12', 'landn_surface18']
#     create GFS Spark DataFrame 
    df_GFS = directory_to_sparkDF(directory = 'train/GFS/', drop_cols = ['landn_surface00', 'landn_surface06', 'landn_surface12', 'landn_surface18'])
#     Group GFS Data by date, grid_id
    df_GFS_agg = df_GFS.groupBy("grid_id", "date", "latitude", "longitude").mean()

# COMMAND ----------

# df_GFS.write.parquet("/mnt/capstone/train/df_GFS.parquet") 
# df_GFS_agg.write.parquet("/mnt/capstone/train/df_GFS_agg.parquet") 

# COMMAND ----------

if INIT_DATASETS:   
    # create AOD Dataframe
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
            StructField("utc_date", StringType(), True)])

    df_AOD = directory_to_sparkDF(directory = 'train/AOD/train/aod', schema=AODCustomSchema)

# COMMAND ----------

if INIT_DATASETS:  
    # create GEOS DataFrame
    elevation_df = spark.read.parquet("dbfs:/mnt/capstone/train/elevation/elevation.parquet")

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

udf_qa_format = sf.udf(lambda x:qa_format(x),StringType() )
udf_mask_int = sf.udf(lambda x:masks_to_int(x),StringType() )

# COMMAND ----------

if INIT_DATASETS:    
    #Recast columns
    df_AOD=df_AOD.withColumn("AOD_qa_str",udf_qa_format(col("AOD_QA")))
    df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask_str', substring('AOD_qa_str', 0,3))\
        .withColumn('AOD_QA_LWS_Mask_str', substring('AOD_qa_str', 3,2))\
        .withColumn('AOD_QA_Adj_Mask_str', substring('AOD_qa_str', 5,3))\
        .withColumn('AOD_Level_str', substring('AOD_qa_str', 8,1))\
        .withColumn('Algo_init_str', substring('AOD_qa_str', 9,1))\
        .withColumn('BRF_over_snow_str', substring('AOD_qa_str', 10,1))\
        .withColumn('BRF_climatology_str', substring('AOD_qa_str', 11,1))\
        .withColumn('AOD_QA_SC_Mask_str', substring('AOD_qa_str', 12,3))
    df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('AOD_QA_LWS_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('AOD_QA_Adj_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('AOD_Level', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('Algo_init', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('BRF_over_snow', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('BRF_climatology', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('AOD_QA_SC_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))

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

    lat_lon_list_df = df_AOD.groupBy("grid_id","utc_date")\
    .agg(sf.collect_list("lon-lat-pair").alias("aod_lon_lat_list"))

# COMMAND ----------

df_AOD.write.parquet("/mnt/capstone/train/df_AOD.parquet") 

# COMMAND ----------

# MAGIC %md Aggregate to grid level by taking summary statistics across grids. 

# COMMAND ----------

if INIT_DATASETS:    
    df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('AOD_QA_LWS_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('AOD_QA_Adj_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('AOD_Level', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('Algo_init', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('BRF_over_snow', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('BRF_climatology', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
        .withColumn('AOD_QA_SC_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))

    df_aod_grid = spark.sql("SELECT grid_id, utc_date,\
                min(Optical_Depth_047) as min_Optical_Depth_047,\
                max(Optical_Depth_047) as max_Optical_Depth_047,\
                mean(Optical_Depth_047) as mean_Optical_Depth_047,\
                min(Optical_Depth_055) as min_Optical_Depth_055,\
                max(Optical_Depth_055) as max_Optical_Depth_055,\
                mean(Optical_Depth_055) as mean_Optical_Depth_055,\
                min(AOD_Uncertainty) as min_AOD_Uncertainty,\
                max(AOD_Uncertainty) as max_AOD_Uncertainty,\
                mean(AOD_Uncertainty) as mean_AOD_Uncertainty,\
                min(Column_WV) as min_Column_WV,\
                max(Column_WV) as max_Column_WV,\
                mean(Column_WV) as mean_Column_WV,\
                min(AOD_QA_Cloud_Mask) as min_AOD_QA_Cloud_Mask,\
                max(AOD_QA_Cloud_Mask) as max_AOD_QA_Cloud_Mask,\
                mean(AOD_QA_Cloud_Mask) as mean_AOD_QA_Cloud_Mask,\
                min(AOD_QA_LWS_Mask) as min_AOD_QA_LWS_Mask,\
                max(AOD_QA_LWS_Mask) as max_AOD_QA_LWS_Mask,\
                mean(AOD_QA_LWS_Mask) as mean_AOD_QA_LWS_Mask,\
                min(AOD_QA_Adj_Mask) as min_AOD_QA_Adj_Mask,\
                max(AOD_QA_Adj_Mask) as max_AOD_QA_Adj_Mask,\
                mean(AOD_QA_Adj_Mask) as mean_AOD_QA_Adj_Mask,\
                min(AOD_Level) as min_AOD_Level,\
                max(AOD_Level) as max_AOD_Level,\
                mean(AOD_Level) as mean_AOD_Level,\
                min(Algo_init) as min_Algo_init,\
                max(Algo_init) as max_Algo_init,\
                mean(Algo_init) as mean_Algo_init,\
                min(BRF_over_snow) as min_BRF_over_snow,\
                max(BRF_over_snow) as max_BRF_over_snow,\
                mean(BRF_over_snow) as mean_BRF_over_snow,\
                min(BRF_climatology) as min_BRF_climatology,\
                max(BRF_climatology) as max_BRF_climatology,\
                mean(BRF_climatology) as mean_BRF_climatology,\
                min(AOD_QA_SC_Mask) as min_AOD_QA_SC_Mask,\
                max(AOD_QA_SC_Mask) as max_AOD_QA_SC_Mask,\
                mean(AOD_QA_SC_Mask) as mean_AOD_QA_SC_Mask\
                FROM aod group by grid_id, utc_date WHERE AOD_QA_Cloud_Mask == 1")

    df_aod_grid = df_aod_grid.join(lat_lon_list_df, on=[df_aod_grid.grid_id == lat_lon_list_df.grid_id,  
                                                       df_aod_grid.utc_date == lat_lon_list_df.utc_date],
                                                    how="left").drop(lat_lon_list_df.grid_id).drop(lat_lon_list_df.utc_date)

# COMMAND ----------

# MAGIC %md Joins

# COMMAND ----------

if INIT_DATASETS:   
    # AOD + GFS
#     aod_gfs_joined = df_AOD.join(df_GFS, on=[df_AOD.grid_id == df_GFS.grid_id,  
#                                             df_AOD.utc_date == df_GFS.date, 
#                                             df_AOD.lat == df_GFS.latitude, 
#                                             df_AOD.lon == df_GFS.longitude],how="left").drop(df_GFS.grid_id).drop(df_GFS.date)
    aod_gfs_joined = df_aod_grid.join(df_GFS_agg, on=[df_aod_grid.grid_id == df_GFS_agg.grid_id,  
                                                df_aod_grid.utc_date == df_GFS_agg.date],how="inner").drop(df_GFS_agg.grid_id).drop(df_GFS_agg.date)

    # AOD + GFS + Labels 
    aod_gfs_joined_with_labels = aod_gfs_joined.join(train_labels_df, on=[aod_gfs_joined.grid_id == train_labels_df.grid_id,  
                                                aod_gfs_joined.utc_date == train_labels_df.date],how="inner").drop(aod_gfs_joined.grid_id).drop(aod_gfs_joined.utc_date)

# COMMAND ----------

if INIT_DATASETS:  
    #AOD + GFS + Labels + Elevation 
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.join(elevation_df, on = [aod_gfs_joined_with_labels.grid_id == elevation_df.grid_id], 
                                                                                   how = "left").drop(elevation_df.polygon_coords).drop(elevation_df.grid_id)


# COMMAND ----------

if INIT_DATASETS:    
    aod_gfs_joined_with_labels.write.parquet("/mnt/capstone/train/aod_gfs_joined_with_labels_clean_AOD.parquet") 

# COMMAND ----------

aod_gfs_joined_with_labels.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.) Feature Engineering

# COMMAND ----------

# MAGIC %md Date Components

# COMMAND ----------

def feature_eng_time(df):
    df = df.withColumn('day_of_month', dayofmonth(df.datetime_utc))
    df = df.withColumn('day_of_week', dayofweek(df.datetime_utc))
    df = df.withColumn('day_of_year', dayofyear(df.datetime_utc))
    df = df.withColumn('week_of_year', weekofyear(df.datetime_utc))
    df = df.withColumn('month', month(df.datetime_utc))
    df = df.withColumn('year', year(df.datetime_utc))
    return df

# COMMAND ----------

# def feature_eng_avg(df):
#     df = df.withColumn('avg_weekday_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'day_of_week')))
#     df = df.withColumn('avg_daily_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'day_of_year')))
#     df = df.withColumn('avg_weekly_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'week_of_year')))
#     df = df.withColumn('avg_monthly_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'month')))
#     df = df.withColumn('avg_yearly_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'year')))
#     return df

# COMMAND ----------

def trailing_features(df, cols):
    days = lambda i: i * 86400
    for col in cols:
        df = df.withColumn('trailing1d_'+col,sf.mean(df[col])\
                           .over(Window.partitionBy('grid_id').orderBy(df['datetime_utc'].cast("timestamp").cast("long"))\
                                 .rangeBetween(-days(1), 0)))
    return df

# COMMAND ----------

if FEATURE_ENG_TIME: 
    aod_gfs_joined_with_labels_read = feature_eng_time(aod_gfs_joined_with_labels_read)

# COMMAND ----------

# if FEATURE_ENG_AVG:   
#     aod_gfs_joined_with_labels_read = feature_eng_avg(aod_gfs_joined_with_labels_read)

# COMMAND ----------

cols_aod = [col for col in aod_gfs_joined_with_labels_read.columns if '_047' in col or '_055' in col]
# cols_aod = cols_aod + ['Aerosol_Optical_Depth','Absorption_Aerosol_Optical_Depth','Nonspherical_Aerosol_Optical_Depth',
#                  'Small_Mode_Aerosol_Optical_Depth','Medium_Mode_Aerosol_Optical_Depth','Large_Mode_Aerosol_Optical_Depth']

# COMMAND ----------

def aod_scale(x):
    if x:
        return (46.759*x)+7.1333
    else: 
        return None
aod_scale_udf = sf.udf(lambda x:aod_scale(x) ,DoubleType())

# COMMAND ----------

for col_aod in cols_aod:
    aod_gfs_joined_with_labels_read=aod_gfs_joined_with_labels_read.withColumn(col_aod+'_scaled',
                                                                               aod_scale_udf(aod_gfs_joined_with_labels_read[col_aod]))

# COMMAND ----------

aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.drop(*cols_aod)

# COMMAND ----------

if FEATURE_ENG_TRAILING: 
    aod_gfs_joined_with_labels_read = trailing_features(aod_gfs_joined_with_labels_read,['Optical_Depth_047_scaled'])

# COMMAND ----------

# aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.withColumn('wind_speed', 
#                    (((aod_gfs_joined_with_labels_read['avg(max(u_pbl_new))']**2)+
#                     (aod_gfs_joined_with_labels_read['avg(max(v_pbl_new))'])**2)**(1/2)))

# COMMAND ----------

import math

# COMMAND ----------

if UPDATE_INF:
    cols_to_update_inf = cols_00+cols_06+cols_12+cols_18
    for i in cols_to_update_inf:
        aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.withColumn(i,
                              when((aod_gfs_joined_with_labels_read[i]==-math.inf),None).otherwise(aod_gfs_joined_with_labels_read[i]))

# COMMAND ----------

#Drop the difflag columns 
# aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_read.drop(*['difflag', 'difflag_lag1day'])

# COMMAND ----------

if SELECT_FEATURES:
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels_read.select(*['trailing1d_min_Optical_Depth_047_scaled',
                                                                          'trailing1d_max_Optical_Depth_047_scaled',
                                                                          'trailing1d_median_Optical_Depth_047_scaled',
                                                                            'day_of_year',
                                                                            'month',
                                                                            'week_of_year',
                                                                            'Aerosol_Optical_Depth_scaled',
                                                                            'min_Optical_Depth_047_scaled',
                                                                            'max_Optical_Depth_047_scaled',
                                                                            'median_Optical_Depth_047_scaled',
                                                                            'min_Optical_Depth_055_scaled',
                                                                            'max_Optical_Depth_055_scaled',
                                                                            'median_Optical_Depth_055_scaled',
                                                                            'Angstrom_Exponent_550_860nm',
                                                                            'min_elevation',
                                                                            'max_elevation',
                                                                            'avg_elevation',
                                                                            'avg(max(t_surface_new))',
                                                                            'avg(max(pbl_surface_new))',
                                                                            'avg(max(hindex_surface_new))',
                                                                            'avg(max(gust_surface_new))',
                                                                            'avg(max(r_atmosphere_new))',
                                                                            'avg(max(pwat_atmosphere_new))',
                                                                            'avg(max(vrate_pbl_new))',
                                                                            'wind_speed',
                                                                            'value',
                                                                            'pm25_date_d'
                                                                            ])

                                                                    
                                                                    
    #                                                                     ['location','datetime','value',
    #                                                                  'avg(AOD_distance_rank)','avg(AOD_distance_to_grid_center)',
    #                                                                  'avg(min_elevation)',
    #                                                                  'avg(max_elevation)','avg(avg_elevation)','avg(max(t_surface_new))',
    #                                                                  'avg(max(pbl_surface_new))','avg(max(hindex_surface_new))',
    #                                                                  'avg(max(gust_surface_new))','avg(max(r_atmosphere_new))',
    #                                                                  'avg(max(pwat_atmosphere_new))','avg(max(u_pbl_new))',
    #                                                                  'avg(max(v_pbl_new))','avg(max(vrate_pbl_new))',
    #                                                                  'month','day_of_year','day_of_week',
    #                                                                  'Angstrom_Exponent_550_860nm',
    #                                                                  'avg(avg(Optical_Depth_047))_scaled',
    #                                                                  'avg(avg(Optical_Depth_055))_scaled',
    #                                                                  'wind_speed','date']
else:
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels_read

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Split Strategy

# COMMAND ----------

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import pandas as pd
import numpy as np
np.random.seed(0)


import os
#import wget
from pathlib import Path

# COMMAND ----------

display(aod_gfs_joined_with_labels_test)

# COMMAND ----------

#INITAL SPLIT FOR TRAIN (will be broken for tuning) and TEST (unseen till final pred generation)
aod_gfs_joined_with_labels_train = aod_gfs_joined_with_labels_read.where("date_utc < '2021-04-01'")
aod_gfs_joined_with_labels_test = aod_gfs_joined_with_labels_read.where("date_utc >= '2021-04-01'")
# aod_gfs_joined_with_labels_train = aod_gfs_joined_with_labels_read.where("(date_utc >= '2018-02-01') and (date_utc <= '2020-12-31')")
# aod_gfs_joined_with_labels_test = aod_gfs_joined_with_labels_read.where("((date_utc >= '2017-01-01') and (date_utc <= '2018-01-31')) or ((date_utc >= '2021-01-01') and (date_utc <= '2021-08-31'))")

# COMMAND ----------

aod_gfs_joined_with_labels_test.agg({'date_utc':'max'}).show()

# COMMAND ----------

display(df_trailing)

# COMMAND ----------

#Plot trailing 10 day by location
df_trailing = aod_gfs_joined_with_labels_read.where("location == 'Santa Clarita' and date_utc >= date_sub(current_date(), 20)").sort('date_utc', ascending = False)
df_trailing = df_trailing.select('date_utc', 'value')
df_trailing = df_trailing.toPandas()
plt.bar(df_trailing['date_utc'], df_trailing['value'])

# COMMAND ----------

split_strategy = 'time'

# COMMAND ----------

if split_strategy == 'time':
    tabnet_df = aod_gfs_joined_with_labels_train.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('date_utc')))

    #5-5-80-5-5 split
    tabnet_df = tabnet_df.withColumn("Set", when((tabnet_df.rank < .8), "train") \
                                                  .when((tabnet_df.rank < .9), "valid") \
                                                   .otherwise("test")).cache()

    #For 8-1-1 split: 
#     tabnet_df = tabnet_df.withColumn("Set", when((tabnet_df.rank < .08), "train") \
#                                                   .when((tabnet_df.rank < .09), "valid") \
#                                                  .when((tabnet_df.rank < .10), "test") \
#                                                  .when((tabnet_df.rank < .18), "train") \
#                                                  .when((tabnet_df.rank < .19), "valid") \
#                                                .when((tabnet_df.rank < .20), "test") \
#                                                .when((tabnet_df.rank < .28), "train") \
#                                                .when((tabnet_df.rank < .29), "valid") \
#                                                .when((tabnet_df.rank < .30), "test") \
#                                                .when((tabnet_df.rank < .38), "train") \
#                                                .when((tabnet_df.rank < .39), "valid") \
#                                                .when((tabnet_df.rank < .40), "test") \
#                                                .when((tabnet_df.rank < .48), "train") \
#                                                .when((tabnet_df.rank < .49), "valid") \
#                                                .when((tabnet_df.rank < .50), "test") \
#                                                .when((tabnet_df.rank < .58), "train") \
#                                                .when((tabnet_df.rank < .59), "valid") \
#                                                .when((tabnet_df.rank < .60), "test") \
#                                                .when((tabnet_df.rank < .68), "train") \
#                                                .when((tabnet_df.rank < .69), "valid") \
#                                                .when((tabnet_df.rank < .70), "test") \
#                                                .when((tabnet_df.rank < .78), "train") \
#                                                .when((tabnet_df.rank < .79), "valid") \
#                                                .when((tabnet_df.rank < .80), "test") \
#                                                .when((tabnet_df.rank < .88), "train") \
#                                                .when((tabnet_df.rank < .89), "valid") \
#                                                .when((tabnet_df.rank < .90), "test") \
#                                                .when((tabnet_df.rank < .98), "train") \
#                                                .when((tabnet_df.rank < .99), "valid") \
#                                                .otherwise("test")).cache()
    
    

    tabnet_df = tabnet_df.drop(*['datetime_utc', 'rank'])
                               #['aod_lon_lat_list', 'pm25_date_d', 'pm25_datetime_dt', 'rank', 'aod_reading_end',
                                # 'datetime','pm25_reading_date', 'tz'])
    tabnet_df = tabnet_df.withColumn('year',sf.year(tabnet_df['date_utc']))\
                         .withColumn('month',sf.month(tabnet_df['date_utc']))\
                         .withColumn('day',sf.month(tabnet_df['date_utc']))
    tabnet_df = tabnet_df.drop(*['date_utc'])
    train = tabnet_df.toPandas()
    target = 'value'

    train_indices = train[train.Set=="train"].index
    valid_indices = train[train.Set=="valid"].index
    test_indices = train[train.Set=="test"].index

# COMMAND ----------

if split_strategy == '2_fold': 
    tabnet_df = aod_gfs_joined_with_labels_train.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('date_utc')))
    
    #5-5-80-5-5 split
    tabnet_df_1 = tabnet_df.withColumn("Set", when((tabnet_df.rank < .4), "train") \
                                                  .when((tabnet_df.rank < .45), "valid") \
                                                  .when((tabnet_df.rank <= .5), "test"))
    
    tabnet_df_2 = tabnet_df.withColumn("Set", when((tabnet_df.rank < .8), "train") \
                                                  .when((tabnet_df.rank < .9), "valid") \
                                                  .when((tabnet_df.rank <= 1.0), "test"))
    tabnet_df = tabnet_df.drop(*['datetime_utc','date_utc', 'rank'])                          
    tabnet_df_1 = tabnet_df_1.drop(*['datetime_utc','date_utc', 'rank'])
    tabnet_df_2 = tabnet_df_2.drop(*['datetime_utc','date_utc', 'rank'])
                                       
    target = 'value'                          
                                   
                                       
    train_1 = tabnet_df_1.toPandas()
    train_2 = tabnet_df_2.toPandas()

    train_indices_1 = train_1[train_1.Set=="train"].index
    valid_indices_1 = train_1[train_1.Set=="valid"].index
    test_indices_1 = train_1[train_1.Set=="test"].index   
                                       
    train_indices_2 = train_2[train_2.Set=="train"].index
    valid_indices_2 = train_1[train_2.Set=="valid"].index
    test_indices_2 = train_2[train_2.Set=="test"].index       

# COMMAND ----------

if split_strategy == 'random_day':
    train_split, val_split, test_split = aod_gfs_joined_with_labels_train.select("date_utc").distinct().randomSplit(weights=[0.8, 0.1, 0.1], seed = 43)

    train_split = train_split.withColumn("Set", lit("train"))
    val_split = val_split.withColumn("Set", lit("valid"))
    test_split = test_split.withColumn("Set", lit("test"))

    sets = train_split.union(val_split)
    sets = sets.union(test_split)
    
    tabnet_df = aod_gfs_joined_with_labels_train.join(sets, on = "date_utc", how = "left")
    tabnet_df = tabnet_df.drop(*['datetime_utc','date_utc', 'rank'])
    
    train = tabnet_df.toPandas()
    target = 'value'

    train_indices = train[train.Set=="train"].index
    valid_indices = train[train.Set=="valid"].index
    test_indices = train[train.Set=="test"].index

# COMMAND ----------

if split_strategy=='random':
    tabnet_df = aod_gfs_joined_with_labels
    tabnet_df = tabnet_df.drop(*['aod_lon_lat_list', 'pm25_date_d', 'pm25_datetime_dt', 'rank', 'aod_reading_end',
                                 'datetime','pm25_reading_date', 'tz'])
    train = tabnet_df.toPandas()
    target = 'value'
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

    train_indices = train[train.Set=="train"].index
    valid_indices = train[train.Set=="valid"].index
    test_indices = train[train.Set=="test"].index

# COMMAND ----------

# For locations that have lagging data available, we use ground truth as one feature in model to improve performance where possible. 
USE_LAG = True

# COMMAND ----------

#https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

# COMMAND ----------

if split_strategy != '2_fold': 
    if USE_LAG: 
        lag_feats = []
    else: 
        lag_feats = [c for c in train.columns if 'lag' in c]

    for col in train.columns[train.dtypes != object]:
        if col != target:
            train[col].fillna(train.loc[train_indices, col].mean(), inplace=True)

    categorical_columns = []
    # categorical_dims =  {}
    for col in train.columns[train.dtypes == object]:
        l_enc = LabelEncoderExt()
        print(col, train[col].nunique())
        train[col] = train[col].fillna("VV_likely")
        l_enc.fit(train[col].values)
        train[col] = l_enc.transform(train[col].values)
        categorical_columns.append(col)
    #     categorical_dims[col] = len(l_enc.classes_)

    unused_feat = ['Set', 'rank', 'location', 'parameter', 'difflag', 'day'] + lag_feats

    #Drop all categorical features as location based features can be learned through lat, lon. 
    # features = [ col for col in train.columns if col not in unused_feat+categorical_columns+[target]] 
    features = [ col for col in train.columns if col not in unused_feat+[target]]
else: 
    #Train 1
    if USE_LAG: 
        lag_feats = []
    else: 
        lag_feats = [c for c in train_1.columns if 'lag' in c]

    for col in train_1.columns[train_1.dtypes != object]:
        if col != target:
            train_1[col].fillna(train_1.loc[train_indices_1, col].mean(), inplace=True)

    categorical_columns = []
    # categorical_dims =  {}
    for col in train_1.columns[train_1.dtypes == object]:
        l_enc = LabelEncoderExt()
        print(col, train_1[col].nunique())
        train_1[col] = train_1[col].fillna("VV_likely")
        l_enc.fit(train_1[col].values)
        train_1[col] = l_enc.transform(train_1[col].values)
        categorical_columns.append(col)
    #     categorical_dims[col] = len(l_enc.classes_)

    unused_feat = ['Set', 'rank', 'location', 'parameter', 'difflag', 'day'] + lag_feats

    #Drop all categorical features as location based features can be learned through lat, lon. 
    # features = [ col for col in train_1.columns if col not in unused_feat+categorical_columns+[target]] 
    features_1 = [ col for col in train_1.columns if col not in unused_feat+[target]]
    
    # Train 2
    if USE_LAG: 
        lag_feats = []
    else: 
        lag_feats = [c for c in train_2.columns if 'lag' in c]

    for col in train_2.columns[train_2.dtypes != object]:
        if col != target:
            train_2[col].fillna(train_2.loc[train_indices_2, col].mean(), inplace=True)

    categorical_columns = []
    # categorical_dims =  {}
    for col in train_2.columns[train_2.dtypes == object]:
        l_enc = LabelEncoderExt()
        print(col, train_2[col].nunique())
        train_2[col] = train_2[col].fillna("VV_likely")
        l_enc.fit(train_2[col].values)
        train_2[col] = l_enc.transform(train_2[col].values)
        categorical_columns.append(col)
    #     categorical_dims[col] = len(l_enc.classes_)

    unused_feat = ['Set', 'rank', 'location', 'parameter', 'difflag', 'day'] + lag_feats

    #Drop all categorical features as location based features can be learned through lat, lon. 
    # features = [ col for col in train_2.columns if col not in unused_feat+categorical_columns+[target]] 
    features_2 = [ col for col in train_2.columns if col not in unused_feat+[target]]


#One hot encode categorical variables
# train = l_enc.fit_transform(train)

# COMMAND ----------

features

# COMMAND ----------

# unused_feat = ['Set', 'rank', 'location', 'parameter']

# features = [ col for col in train.columns if col not in unused_feat+[target]] 

# cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

# cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# # define your embedding sizes : here just a random choice
# cat_emb_dim = cat_dims

# COMMAND ----------

if split_strategy != '2_fold':
    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices].reshape(-1, 1)

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices].reshape(-1, 1)

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices].reshape(-1, 1)
else: 
    X_train = train_1[features_1].values[train_indices_1]
    y_train = train_1[target].values[train_indices_1].reshape(-1, 1)

    X_valid = train_1[features_1].values[valid_indices_1]
    y_valid = train_1[target].values[valid_indices_1].reshape(-1, 1)

    X_test = train_1[features_1].values[test_indices_1]
    y_test = train_1[target].values[test_indices_1].reshape(-1, 1)


    X_train_2 = train_2[features_2].values[train_indices_2]
    y_train_2 = train_2[target].values[train_indices_2].reshape(-1, 1)

    X_valid_2 = train_2[features_2].values[valid_indices_2]
    y_valid_2 = train_2[target].values[valid_indices_2].reshape(-1, 1)

    X_test_2 = train_2[features_2].values[test_indices_2]
    y_test_2 = train_2[target].values[test_indices_2].reshape(-1, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimensionality Reduction

# COMMAND ----------

from sklearn.decomposition import PCA

# COMMAND ----------

def pca_features(train, val, test):
    
    pca = PCA(n_components = 'mle', svd_solver = 'full').fit(train)

    train_transformed_nmf = pca.transform(train)
    new_colnames = ['pca_' + 'gram_f_'+str(i+1) for i in range(train_transformed_nmf.shape[1])]
    df_train = pd.DataFrame(train_transformed_nmf, columns=new_colnames)
    
    if(val is not None):
        val_transformed_nmf = pca.transform(val)
        df_val = pd.DataFrame(val_transformed_nmf, columns=new_colnames)
    else:
        df_val = None
    if(test is not None):
        test_transformed_nmf = pca.transform(test)
        df_test = pd.DataFrame(test_transformed_nmf, columns=new_colnames)
    else:
        df_test = None
    return df_train, df_val, df_test

# COMMAND ----------

X_train, X_valid, X_test = pca_features(X_train, X_valid, X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # MODELS

# COMMAND ----------

!pip install hyperopt

# COMMAND ----------

!pip install pytorch-tabnet

# COMMAND ----------

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from hyperopt import SparkTrials, STATUS_OK, Trials, fmin, hp, tpe

# COMMAND ----------

# MAGIC %md
# MAGIC ##Setup for Ensemble

# COMMAND ----------

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn import linear_model 

from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
import tensorflow as tf

# COMMAND ----------

#for tabnet
max_epochs = 1000 if not os.getenv("CI", False) else 2
batch_size = 1024

def get_model_classes():
    model_classes = {}
    model_classes['knn'] = KNeighborsRegressor(algorithm='ball_tree',leaf_size=28.0, metric='jaccard', n_jobs=-1, 
                                               n_neighbors=7, weights='uniform')
    model_classes['svm'] = SVR() #Notes: Very slow, warnings about running on samples of 10,000+ 
    model_classes['rf'] = RandomForestRegressor(bootstrap=True, max_depth=9.0, max_features=15,
                                                min_samples_split=6, n_estimators=1000)
    model_classes['dt'] = DecisionTreeRegressor(max_depth=2.0, max_features=None, max_leaf_nodes=8, min_samples_leaf=6,
                                                min_weight_fraction_leaf=0.1, splitter='best')
    model_classes['et'] = ExtraTreesRegressor(max_depth=75.0, min_samples_leaf=45, min_samples_split=25, max_features=0.30000000000000004)
    model_classes['xgb'] = XGBRegressor(#booster='max_features',
                colsample_bylevel=0.9711210698757635,
                colsample_bynode=0.8189616926583123,
                colsample_bytree=0.5280616128778559,# enable_categorical=False,
                gamma=85.20914214282189, gpu_id=-1, importance_type=None,
                interaction_constraints=None, learning_rate=0.4537384532200076,
                max_delta_step=7, max_depth=6,
                min_child_weight=0.18345411371256903,
                monotone_constraints=None, n_estimators=700,# n_jobs=8,
                num_parallel_tree=None, predictor=None, random_state=0,
                reg_alpha=60.0, reg_lambda=246.0, scale_pos_weight=1,
                subsample=0.6329777403826279, tree_method='approx',
                validate_parameters=1, verbosity=None)
    model_classes['ada'] = AdaBoostRegressor(learning_rate=0.012332486258756601, n_estimators=200, random_state=1)
    model_classes['gbm'] = GradientBoostingRegressor(ccp_alpha=0.09438419349933552, criterion='friedman_mse',
                             learning_rate=0.07129112745042249,
                             max_depth=5.367967051232338, # max_features='log2',
                             min_impurity_decrease=0.19010339880135252,
                             min_samples_split=0.9, n_estimators=700, random_state=1,
                             subsample=0.6097113225774524)
    model_classes['lr'] = LinearRegression()
    model_classes['nb'] = GaussianNB() #Notes: not suited for our feature space
    model_classes['nn'] = MLPRegressor(activation='tanh', alpha=0.1665264267040525,
                beta_1=0.9437180891012691, beta_2=0.9983706527475311,
                epsilon=0.0008675215392752822, hidden_layer_sizes=(500, 1000, 500),
                learning_rate='adaptive', learning_rate_init=0.061284425608596774,
                momentum=0.6552790893692165)
    model_classes['bag'] = BaggingRegressor(bootstrap=True, bootstrap_features=False, oob_score=True,
                                            max_features=0.2220044064208537, max_samples=0.978651699251462, n_estimators=900)
    model_classes['tn'] = TabNetRegressor(cat_dims=cat_dims, 
                                         cat_emb_dim=cat_emb_dim, 
                                         cat_idxs=cat_idxs, 
                                         optimizer_fn=torch.optim.Adam, # Any optimizer works here
                                         optimizer_params=dict(lr=2e-2),
                                         scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
                                         scheduler_params={"is_batch_level":True,
                                                         "max_lr":5e-2,
                                                         "steps_per_epoch":int(train.shape[0] / batch_size)+1,
                                                         "epochs":max_epochs
                                                          },
                                         mask_type='entmax')
    return model_classes

# COMMAND ----------

def get_param_grids():
    param_grids = {}
    param_grids['svm'] = {
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'C': [0.1, 1, 100] ,
        'gamma': [0.0001, 0.001, 0.1, 1, 5]}

    param_grids['knn'] = {
        'n_neighbors':hp.quniform('n_neighbors',5,12,1),
        'leaf_size': hp.quniform('leaf_size', 5, 30, 1),
        'metric': hp.choice('metric',['minkowski','manhattan','euclidean','cosine','jaccard','hamming']),
        'weights':hp.choice('weights',['uniform', 'distance']),
        'algorithm': hp.choice('algorithm',['auto', 'ball_tree','kd_tree','brute']),
        'n_jobs':hp.choice('n_jobs',[-1])}

    param_grids['rf'] = {
        'n_estimators': hp.choice('n_estimators', [int(x) for x in np.linspace(200, 1000, 5)]),
        'max_depth': hp.quniform('max_depth', 1, 15,1),
        'criterion': hp.choice('criterion', ['squared_error', 'absolute_error', 'poisson']),
        'max_features': hp.choice('max_features', ['auto', 'log2', 8, 10, 15, 20]),
        'min_samples_split': hp.choice('min_samples_split',[int(x) for x in np.linspace(4, 10, 7)]),
        'bootstrap': hp.choice('bootstrap', [True, False]),} 
        #'class_weight': hp.choice('class_weight',[None, 'balanced', 'balanced_subsample'])}

    param_grids['dt'] = {
        'splitter': hp.choice('splitter',["best","random"]),
        'max_depth' : hp.quniform('max_depth',1,15,1),
        'min_samples_leaf': hp.quniform('min_samples_leaf',1,10,1), 
        'min_weight_fraction_leaf': hp.quniform('min_weight_fraction_leaf',0.1,1,0.1),
        'max_features': hp.choice('max_features',["auto","log2","sqrt",None]),
        'max_leaf_nodes': hp.choice('max_leaf_nodes',[None,10,20,30,40,50,60,70,80,90]) }
    
    param_grids['et'] = {
        'n_estimators': hp.quniform('max_depth',50,126,25),
        'max_features': hp.quniform('max_features',0,1,0.1),
        'min_samples_leaf': hp.quniform('min_samples_leaf',20,50,5),
        'min_samples_split': hp.quniform('min_samples_split',15,36,5)}

    param_grids['xgb'] = {
        'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'objective': hp.choice('objective', ['reg:squarederror', 'reg:linear']), 
        'booster': hp.choice('booster', ['c', 'gblinear','dart']), 
        'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),
        'subsample': hp.uniform('subsample', 0, 1), 
        'gamma': hp.uniform('gamma', 80,100),
        'reg_alpha' : hp.quniform('reg_alpha', 0,120,1),
        'reg_lambda' : hp.quniform('reg_lambda', 100,250, 1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.2,1),
        'colsample_bylevel' : hp.uniform('colsample_bylevel', 0.2,1), 
        'colsample_bynode' : hp.uniform('colsample_bynode', 0.2,1), 
        'min_child_weight' : hp.uniform('min_child_weight', 0, 1),
        'n_estimators': hp.quniform("n_estimators", 100, 1200, 100),
        'learning_rate': hp.uniform("learning_rate", 0.01, 1), 
        'tree_method': hp.choice('tree_method', ['exact', 'approx']),
        'seed': 0
    }

    param_grids['ada'] = {
        'n_estimators': hp.quniform('n_estimators',10,500,50),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1.0))}

    param_grids['gbm'] = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
        'learning_rate': hp.uniform('learning_rate', 0.0, 0.1),
        'subsample': hp.uniform('subsample', 0.2, 1.0),
        'criterion': hp.choice('criterion', 
            ['friedman_mse', 'squared_error', 'mse']), 
        'min_samples_split': hp.quniform('min_samples_split', 0, 1, 0.1), 
        'max_depth': hp.uniform('max_depth', 3, 12), 
        'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 1.0), 
        'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']), 
        'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 0.1)
    }

    param_grids['lr'] = {
        'normalize': hp.choice('normalize', ['True', 'False'])
    }
#     param_grids['nb'] = {
#         'var_smoothing': hp.loguniform('var_smoothing', -9, 0)}

    param_grids['nn'] = {
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes',
            ['(10,30,10)', '(20, 60, 20)', '(100,)', '(100, 200, 100)', '(500, 1000, 500)']), 
        'activation': hp.choice('activation', ['tanh', 'relu']), 
        'solver': hp.choice('solver', ['sgd', 'adam']),
        'alpha': hp.uniform('alpha', 0, 1),
        'learning_rate': hp.choice('learning_rate', ['constant','adaptive']), 
        'learning_rate_init': hp.uniform('learning_rate_init', 0.0, 0.1), 
        'shuffle': hp.choice('shuffle', ['True', 'False']), 
        'momentum': hp.uniform('momentum', 0, 1), 
        'beta_1': hp.uniform('beta_1', 0.9, 1.0), 
        'beta_2': hp.uniform('beta_2', 0.99, 1.00), 
        'epsilon': hp.loguniform('epsilon', -9, 0)
        }

    param_grids['bag'] = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
        'max_samples': hp.uniform('max_samples', 0, 1), 
        'max_features': hp.uniform('max_features', 0, 1), 
        'bootstrap': hp.choice('bootstrap', ['True', 'False']),
        'bootstrap_features': hp.choice('bootstrap_features', ['True', 'False']), 
        'oob_score': hp.choice('oob_score', ['True', 'False'])    
        }
    
    param_grids['tn'] = {
        'mask_type': hp.choice('mask_type',['entmax','sparsemax'])
    }
    
    return param_grids

# COMMAND ----------

def build_tuned_model(X_train, X_valid, X_test, y_train, y_valid, y_test, model_class, mdl_early_stopping_rounds, tune_max_evals):
    model = get_model_classes()[model_class]
    space = get_param_grids()[model_class]
    
    trials = Trials()

    def objective(space):

        evaluation_all = [(X_train, y_train), (X_valid, y_valid)]
        if(model_class=='xgb'):
            model.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="rmse", 
                    early_stopping_rounds=mdl_early_stopping_rounds, 
                    verbose = 100)
        if(model_class=='tn'):
            model.fit(
                    X_train=X_train, y_train=y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)],
                    eval_name=['train', 'valid'],
                    eval_metric=['rmse', 'mae'],
                    max_epochs=max_epochs,
                    patience=50,
                    batch_size=1024, virtual_batch_size=128,
                    num_workers=0,
                    drop_last=False, 
                    #Load pretrained model
                #     from_unsupervised=loaded_pretrain
                    ) 
        else:
            model.fit(X_train, y_train)

        pred = np.array(model.predict(X_train))
        train_r2 = r2_score(y_pred=pred, y_true=y_train)
        pred = np.array(model.predict(X_valid))
        valid_r2 = r2_score(y_pred=pred, y_true=y_valid)
        pred = np.array(model.predict(X_test))
        test_r2 = r2_score(y_pred=pred, y_true=y_test)
#         print("Train Score:", train_r2)
#         print("Valid Score:", valid_r2)
#         print("Test Score:", test_r2)
#         print()
        return {'loss': -(test_r2- (train_r2 - test_r2)), 'model': model, 'status': STATUS_OK }

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = tune_max_evals,
                            trials = trials)

    print("The best hyperparameters for ", model_class, " are : ")
    print(best_hyperparams)
    best_model_new = trials.results[np.argmin([r['loss'] for r in  trials.results])]['model']
    return (model_class,best_model_new)

# COMMAND ----------

def build_model(X_train, X_valid, X_test, y_train, y_valid, y_test, model_class, mdl_early_stopping_rounds, tune_max_evals):
    model = get_model_classes()[model_class]
    space = get_param_grids()[model_class]
    
    if(model_class=='xgb'):
        model.fit(X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse", 
                early_stopping_rounds=mdl_early_stopping_rounds, 
                verbose = 100)
    if(model_class=='tn'):
        model.fit(
                X_train=X_train, y_train=y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_name=['train', 'valid'],
                eval_metric=['rmse', 'mae'],
                max_epochs=max_epochs,
                patience=50,
                batch_size=1024, virtual_batch_size=128,
                num_workers=0,
                drop_last=False, 
                #Load pretrained model
            #     from_unsupervised=loaded_pretrain
                ) 
    else:
        model.fit(X_train, y_train)

    pred = np.array(model.predict(X_train))
    train_r2 = r2_score(y_pred=pred, y_true=y_train)
    pred = np.array(model.predict(X_valid))
    valid_r2 = r2_score(y_pred=pred, y_true=y_valid)
    pred = np.array(model.predict(X_test))
    test_r2 = r2_score(y_pred=pred, y_true=y_test)
#         print("Train Score:", train_r2)
#         print("Valid Score:", valid_r2)
#         print("Test Score:", test_r2)
#         print()

    return (model_class,model)

# COMMAND ----------

def models_to_include(X_train, X_valid, X_test, y_train, y_valid, y_test, model_types_to_include,
                      mdl_early_stopping_rounds, tune_max_evals, tune=False):
    models = []
    for mt in model_types_to_include:
        if tune:
            models.append(build_tuned_model(X_train, X_valid, X_test, y_train, y_valid, y_test, mt, 
                                       mdl_early_stopping_rounds, tune_max_evals))
        else:
            models.append(build_model(X_train, X_valid, X_test, y_train, y_valid, y_test, mt, 
                                       mdl_early_stopping_rounds, tune_max_evals))
    return models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensemble

# COMMAND ----------

def get_stacking(base_models, meta_learner=None):
	# define the stacking ensemble
    if meta_learner==None:
        model = StackingRegressor(estimators=base_models)
    else:
        model = StackingRegressor(estimators=base_models,final_estimator=meta_learner)
    return model

# COMMAND ----------

def build_ensemble(X_train, X_valid, X_test, y_train, y_valid, y_test,
                   model_types_to_include=['xgb'], mdl_early_stopping_rounds=40, tune_max_evals=40, tune=False):
                    # model_types_to_include=['svm','knn','rf','et','xgb','ada','gbm','lr','nb','nn','bag']
    models = models_to_include(X_train, X_valid, X_test, y_train, y_valid, y_test, model_types_to_include,
                              mdl_early_stopping_rounds, tune_max_evals, tune=tune)

    return get_stacking(models)#,get_model_classes()['et'])

# COMMAND ----------

final_ensemble_model = build_ensemble(X_train, X_valid, X_test, y_train, y_valid, y_test, 
                                      #Beware: svm, 
                                      #model_types_to_include=['knn'],
                                      model_types_to_include=['knn','rf','dt','et','xgb','ada','gbm','bag'],
                                      mdl_early_stopping_rounds=40, tune_max_evals=40, tune=False)#,
#                    model_types_to_include=['svm','knn','rf','et','xgb','ada','gbm','lr','nb','nn','bag'])

# COMMAND ----------

final_ensemble_model.get_params()

# COMMAND ----------

final_ensemble_model.fit(X_train, y_train)
pred = np.array(final_ensemble_model.predict(X_train))
train_r2 = r2_score(y_pred=pred, y_true=y_train)
pred = np.array(final_ensemble_model.predict(X_valid))
valid_r2 = r2_score(y_pred=pred, y_true=y_valid)
pred = np.array(final_ensemble_model.predict(X_test))
test_r2 = r2_score(y_pred=pred, y_true=y_test)
print("Train Score:", train_r2)
print("Valid Score:", valid_r2)
print("Test Score:", test_r2)

# COMMAND ----------

features

# COMMAND ----------

plt.hist(y_test, bins='auto')

# COMMAND ----------

plt.hist(pred, bins='auto')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TabNet

# COMMAND ----------

!pip install pytorch-tabnet

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
import tensorflow as tf


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Pretrain

# COMMAND ----------

# TabNetPretrainer
unsupervised_model = TabNetPretrainer(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax', # "sparsemax",
#     n_shared_decoder=1, # nb shared glu for decoding
#     n_indep_decoder=1, # nb independent glu for decoding
)

# COMMAND ----------

max_epochs = 1000 if not os.getenv("CI", False) else 2

# COMMAND ----------

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_valid],
    max_epochs=max_epochs , patience=50,
    batch_size=2048, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.8,
) 

# COMMAND ----------

# # Make reconstruction from a dataset
# reconstructed_X, embedded_X = unsupervised_model.predict(X_valid)
# assert(reconstructed_X.shape==embedded_X.shape)

# COMMAND ----------

# unsupervised_explain_matrix, unsupervised_masks = unsupervised_model.explain(X_valid)

# COMMAND ----------

# fig, axs = plt.subplots(1, 3, figsize=(20,20))

# for i in range(3):
#     axs[i].imshow(unsupervised_masks[i][:50])
#     axs[i].set_title(f"mask {i}")

# COMMAND ----------

# unsupervised_model.save_model('/mnt/capstone/model/test_pretrain')
# loaded_pretrain = TabNetPretrainer()
# loaded_pretrain.load_model('/mnt/capstone/model/test_pretrain.zip')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Supervised 

# COMMAND ----------

max_epochs = 1000 if not os.getenv("CI", False) else 2
batch_size = 1024

# COMMAND ----------

tabnet = TabNetRegressor(cat_dims=cat_dims, 
                         cat_emb_dim=cat_emb_dim, 
                         cat_idxs=cat_idxs, 
                         optimizer_fn=torch.optim.Adam, # Any optimizer works here
                         optimizer_params=dict(lr=2e-2),
                         scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
                         scheduler_params={"is_batch_level":True,
                                         "max_lr":5e-2,
                                         "steps_per_epoch":int(train.shape[0] / batch_size)+1,
                                         "epochs":max_epochs
                                          },
                         mask_type='entmax',seed=0) # "sparsemax",)

# COMMAND ----------

tabnet.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmse'],
    max_epochs=max_epochs,
    patience=50,
    batch_size=256, virtual_batch_size=128,
    num_workers=0,
    drop_last=False, 
    #Load pretrained model
#     from_unsupervised=loaded_pretrain
) 

# COMMAND ----------

preds = tabnet.predict(X_test)

y_true = y_test

test_score = r2_score(y_pred=preds, y_true=y_true)
valid_score = r2_score(y_pred=tabnet.predict(X_valid), y_true=y_valid)
train_score = r2_score(y_pred=tabnet.predict(X_train), y_true=y_train)
print(f"TRAIN SCORE FOR : {train_score}")
print(f"VALID SCORE FOR : {valid_score}")
print(f"TEST SCORE FOR: {test_score}")

# COMMAND ----------

#tabnet.save_model('/mnt/capstone/model/tabnet_03_13')

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost

# COMMAND ----------

from xgboost import XGBRegressor

# COMMAND ----------

clf_xgb = XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=0.3959137968655604, colsample_bynode=0.38427772092378626, colsample_bytree=0.7366289714335655, enable_categorical=False, gamma=14.0281111983136, gpu_id=-1, importance_type=None, interaction_constraints='', learning_rate=0.5386955173692471, max_delta_step=6, max_depth=8, min_child_weight=0.4016920074761443, monotone_constraints='()', n_estimators=900, n_jobs=8, num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=22, reg_lambda=204, scale_pos_weight=1, subsample=0.8783345980525217, tree_method='exact', validate_parameters=1, verbosity=None)


# XGBRegressor(base_score=0.5, 
#                        booster='gbtree',
#                        colsample_bylevel=0.4572943401260166,
#                        colsample_bynode=0.9817802074404147,
#                        colsample_bytree=0.385678979854995, 
#                        enable_categorical=False,
#                        gamma=100,   # 4.263935519930179, 
#                        gpu_id=-1, 
#                        importance_type=None,
#                        interaction_constraints='', 
#                        learning_rate=0.1716282906637911,
#                        max_delta_step=10, 
#                        max_depth=5,  #11
#                        min_child_weight=0.4733776751261309, 
#                        #missing=nan,
#                        monotone_constraints='()', 
#                        n_estimators=500, 
#                        n_jobs=8,
#                        num_parallel_tree=1, 
#                        objective='reg:linear', 
#                        predictor='auto',
#                        random_state=0, 
#                        reg_alpha=100, 
#                        reg_lambda=200, #111
#                        scale_pos_weight=1,
#                        subsample=0.8941730932476202, 
#                        tree_method='approx',
#                        validate_parameters=1, 
#                        verbosity=None)


clf_xgb.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=40,
        verbose=10)


# COMMAND ----------

preds = np.array(clf_xgb.predict(X_train))
train_r2 = r2_score(y_pred=preds, y_true=y_train)
print(train_r2)

preds = np.array(clf_xgb.predict(X_valid))
valid_r2 = r2_score(y_pred=preds, y_true=y_valid)
print(valid_r2)

preds = np.array(clf_xgb.predict(X_test))
test_r2 = r2_score(y_pred=preds, y_true=y_test)
print(test_r2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Tuning

# COMMAND ----------

!pip install hyperopt

# COMMAND ----------

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from hyperopt import SparkTrials, STATUS_OK, Trials, fmin, hp, tpe
from sklearn import base


# COMMAND ----------

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
       'objective': hp.choice('objective', ['reg:squarederror', 'reg:linear']), 
       'booster': hp.choice('booster', ['gbtree','dart']), #'gblinear' doesnt work with shap for explainability
       'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),
       'subsample': hp.uniform('subsample', 0, 1), 
        'gamma': hp.uniform('gamma', 0,120),
        'reg_alpha' : hp.quniform('reg_alpha', 0,120,1),
        'reg_lambda' : hp.quniform('reg_lambda', 100,250, 1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.2,1),
        'colsample_bylevel' : hp.uniform('colsample_bylevel', 0.2,1), 
        'colsample_bynode' : hp.uniform('colsample_bynode', 0.2,1), 
        'min_child_weight' : hp.uniform('min_child_weight', 0, 1),
        'n_estimators': hp.quniform("n_estimators", 100, 1200, 100),
        'learning_rate': hp.uniform("learning_rate", 0.01, 1), 
        'tree_method': hp.choice('tree_method', ['exact', 'approx']),
        'seed': 0
    }

# COMMAND ----------

def objective(space):
    clf=XGBRegressor(
                    max_depth = int(space['max_depth']), 
                    objective = space['objective'], 
                    booster = space['booster'], 
                    max_delta_step = int(space['max_delta_step']), 
                    subsample = space['subsample'], 
                    gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),
                    reg_lambda = int(space['reg_lambda']), 
                    colsample_bytree= space['colsample_bytree'], 
                    colsample_bylevel= space['colsample_bylevel'], 
                    colsample_bynode = space['colsample_bynode'], 
                    min_child_weight= space['min_child_weight'],
                    n_estimators =int(space['n_estimators']),
                    learning_rate = space['learning_rate'], 
                    tree_method = space['tree_method']
                    )
    
    evaluation_all = [( X_train, y_train), ( X_valid, y_valid)]
    
    clf.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse", 
            early_stopping_rounds=40, 
            verbose = 100)
    

    pred = np.array(clf.predict(X_test))
    test_r2 = r2_score(y_pred=pred, y_true=y_test)
    print("SCORE:", test_r2)
    return {'loss': -test_r2, 'model': clf, 'status': STATUS_OK }

# COMMAND ----------

def objective_CV(space):
    clf=XGBRegressor(
                    max_depth = int(space['max_depth']), 
                    objective = space['objective'], 
                    booster = space['booster'], 
                    max_delta_step = int(space['max_delta_step']), 
                    subsample = space['subsample'], 
                    gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),
                    reg_lambda = int(space['reg_lambda']), 
                    colsample_bytree= space['colsample_bytree'], 
                    colsample_bylevel= space['colsample_bylevel'], 
                    colsample_bynode = space['colsample_bynode'], 
                    min_child_weight= space['min_child_weight'],
                    n_estimators =int(space['n_estimators']),
                    learning_rate = space['learning_rate'], 
                    )
    
    clf_2 = base.clone(clf)

    evaluation_all = [( X_train, y_train), ( X_valid, y_valid), 
                     (X_train_2, y_train_2), (X_valid_2, y_valid_2)]

    clf.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse", 
            early_stopping_rounds=40, 
            verbose = 100)
    
    train_pred = np.array(clf.predict(X_train))
    pred = np.array(clf.predict(X_test))

    clf_2.fit(X_train_2, y_train_2,
            eval_set=[(X_valid_2, y_valid_2)],
            eval_metric="rmse", 
            early_stopping_rounds=40, 
            verbose = 100)
    
    train_pred_2 = np.array(clf_2.predict(X_train_2))
    pred_2 = np.array(clf_2.predict(X_test_2))
    

    
    train_r2 = r2_score(y_pred=train_pred, y_true=y_train)
    train_r2_2 = r2_score(y_pred=train_pred_2, y_true=y_train_2)
    
    test_r2 = r2_score(y_pred=pred, y_true=y_test)
    test_r2_2 = r2_score(y_pred=pred_2, y_true=y_test_2)
    
    avg_train_r2 = (train_r2 + train_r2_2) / 2
    avg_r2 = (test_r2 + test_r2_2) / 2
    
    expected = test_r2 - (train_r2 - test_r2)
    expected_2 = test_r2_2 - (train_r2_2 - test_r2_2)
    
    avg_expected = (expected + expected_2) / 2

    print(f'TRAIN SCORE1: {train_r2} TEST SCORE1: {test_r2} VARIANCE1: {test_r2 - train_r2}  EXPECTED1: {expected}, TRAIN SCORE2: {train_r2_2} TEST SCORE2: {test_r2_2} VARIANCE2: {test_r2_2 - train_r2_2} EXPECTED2: {expected_2}')
    return {'loss': -avg_r2, 'model1': clf, 'model2':clf_2, 'status': STATUS_OK }

# COMMAND ----------

if split_strategy != '2_fold': 
    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 40,
                            trials = trials)

    print("The best hyperparameters are : ","\n")
    print(best_hyperparams)

    best_model_new = trials.results[np.argmin([r['loss'] for r in  trials.results])]['model']
else: 
    trials = Trials()

    best_hyperparams = fmin(fn = objective_CV,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 40,
                            trials = trials)

    print("The best hyperparameters are : ","\n")
    print(best_hyperparams)

    best_model_new = trials.results[np.argmin([r['loss'] for r in  trials.results])]['model1']

    best_model_new_2 = trials.results[np.argmin([r['loss'] for r in  trials.results])]['model2']

# COMMAND ----------

best_model_new

# COMMAND ----------

if split_strategy != '2_fold':   
    preds = np.array(best_model_new.predict(X_train))
    train_r2 = r2_score(y_pred=preds, y_true=y_train)
    print(train_r2)

    preds = np.array(best_model_new.predict(X_valid))
    valid_r2 = r2_score(y_pred=preds, y_true=y_valid)
    print(valid_r2)

    preds = np.array(best_model_new.predict(X_test))
    test_r2 = r2_score(y_pred=preds, y_true=y_test)
    print(test_r2)
else: 
    clf_xgb = best_model_new
    clf_xgb_2 = best_model_new_2

    clf_xgb.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=40,
            verbose=10)

    clf_xgb_2.fit(X_train_2, y_train_2,
            eval_set=[(X_valid_2, y_valid_2)],
            early_stopping_rounds=40,
            verbose=10)
    
    train_preds = np.array(clf_xgb.predict(X_train))
    train_preds_2 = np.array(clf_xgb_2.predict(X_train_2))
    train_r2 = r2_score(y_pred=train_preds, y_true=y_train)
    train_r2_2 = r2_score(y_pred=train_preds_2, y_true=y_train_2)
    print(f'1: {train_r2}, 2: {train_r2_2}')
    print((train_r2 + train_r2_2) / 2)

    val_preds = np.array(clf_xgb.predict(X_valid))
    val_preds_2 = np.array(clf_xgb_2.predict(X_valid_2))
    valid_r2 = r2_score(y_pred=val_preds, y_true=y_valid)
    valid_r2_2 = r2_score(y_pred=val_preds_2, y_true=y_valid_2)
    print(f'1: {valid_r2}, 2: {valid_r2_2}')
    print((valid_r2 + valid_r2_2) / 2)

    test_preds = np.array(clf_xgb.predict(X_test))
    test_preds_2 = np.array(clf_xgb_2.predict(X_test_2))
    test_r2 = r2_score(y_pred=test_preds, y_true=y_test)
    test_r2_2 = r2_score(y_pred=test_preds_2, y_true=y_test_2)
    print(f'1: {test_r2}, 2: {test_r2_2}')
    print((test_r2 + test_r2_2) / 2)

# COMMAND ----------

# best_model_new.save_model('/mnt/capstone/model/aod_filtered_gfs_elev_wlabels_joined_randomsplit.json')

# COMMAND ----------

# saved_model = XGBRegressor()
# saved_model.load_model('dbfs:/mnt/capstone/model/aod_filtered_gfs_elev_wlabels_joined_prepostsplit.json')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Explainability

# COMMAND ----------

import shap

# COMMAND ----------

shap_values = shap.TreeExplainer(best_model_new).shap_values(train_2[features_2])

# COMMAND ----------

shap.summary_plot(shap_values, train_2[features_2])

# COMMAND ----------

shap.plots.force(shap_values)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Submission

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Retrain Best Model on all Train Data

# COMMAND ----------

test_aod_gfs_joined_with_labels_read = aod_gfs_joined_with_labels_test

test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.drop(*['datetime_utc'])
test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn('year',sf.year(test_aod_gfs_joined_with_labels_read['date_utc']))\
                         .withColumn('month',sf.month(test_aod_gfs_joined_with_labels_read['date_utc']))\
                         .withColumn('day',sf.month(test_aod_gfs_joined_with_labels_read['date_utc']))
test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.drop(*['date_utc'])

# COMMAND ----------

#Comparison point: Calculate R2 using trailing1day PM2.5 value as prediction 
preds_mean = test_aod_gfs_joined_with_labels_read.select(sf.avg('value_lag1day')).toPandas().values[0][0]
actual_mean = test_aod_gfs_joined_with_labels_read.select(sf.avg('value')).toPandas().values[0][0]

preds = np.array(test_aod_gfs_joined_with_labels_read.select('value_lag1day').fillna(preds_mean).toPandas())
y_true = np.array(test_aod_gfs_joined_with_labels_read.select('value').fillna(actual_mean).toPandas())
test_r2 = r2_score(y_pred=preds, y_true=y_true)
print(test_r2)

# COMMAND ----------

preds_mean

# COMMAND ----------

display(test_aod_gfs_joined_with_labels_read)

# COMMAND ----------

#averaging gfs time based features
# cols_00 = [col for col in test_aod_gfs_joined_with_labels_read.columns if '00' in col]
# cols_06 = [col for col in test_aod_gfs_joined_with_labels_read.columns if '06' in col]
# cols_12 = [col for col in test_aod_gfs_joined_with_labels_read.columns if '12' in col]
# cols_18 = [col for col in test_aod_gfs_joined_with_labels_read.columns if '18' in col]
# for i in range(len(cols_00)):
#     colName = cols_00[i].replace("00","")
#     test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn(colName, avg_cols(sf.array(cols_00[i], cols_06[i],cols_12[i],cols_18[i])))

# COMMAND ----------

# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.drop(*cols_00)
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.drop(*cols_06)
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.drop(*cols_12)
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.drop(*cols_18)

# COMMAND ----------

# if FEATURE_ENG_TIME: 
#     test_aod_gfs_joined_with_labels_read = feature_eng_time(test_aod_gfs_joined_with_labels_read)

# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn('day_of_month', dayofmonth(test_aod_gfs_joined_with_labels_read.datetime))
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn('day_of_week', dayofweek(test_aod_gfs_joined_with_labels_read.datetime))
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn('day_of_year', dayofyear(test_aod_gfs_joined_with_labels_read.datetime))
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn('week_of_year', weekofyear(test_aod_gfs_joined_with_labels_read.datetime))
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn('month', month(test_aod_gfs_joined_with_labels_read.datetime))
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn('year', year(test_aod_gfs_joined_with_labels_read.datetime))

# COMMAND ----------

# test_cols_aod = [col for col in test_aod_gfs_joined_with_labels_read.columns if '_047' in col or '_055' in col]
# test_cols_aod = test_cols_aod + ['Aerosol_Optical_Depth','Absorption_Aerosol_Optical_Depth','Nonspherical_Aerosol_Optical_Depth',
#                  'Small_Mode_Aerosol_Optical_Depth','Medium_Mode_Aerosol_Optical_Depth','Large_Mode_Aerosol_Optical_Depth']

# for col_aod in test_cols_aod:
#     test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn(col_aod+'_scaled',aod_scale_udf(test_aod_gfs_joined_with_labels_read[col_aod]))
# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.drop(*test_cols_aod)

# COMMAND ----------

# if FEATURE_ENG_TRAILING: 
#     test_aod_gfs_joined_with_labels_read = trailing_features(test_aod_gfs_joined_with_labels_read,['min_Optical_Depth_047_scaled',
#                                                                                'max_Optical_Depth_047_scaled',
#                                                                                'median_Optical_Depth_047_scaled',
#                                                                                'min_Optical_Depth_055_scaled',
#                                                                                'max_Optical_Depth_055_scaled',
#                                                                                'median_Optical_Depth_055_scaled'])

# COMMAND ----------

# test_aod_gfs_joined_with_labels_read = test_aod_gfs_joined_with_labels_read.withColumn('wind_speed', 
#                    (((test_aod_gfs_joined_with_labels_read['avg(max(u_pbl_new))']**2)+
#                     (test_aod_gfs_joined_with_labels_read['avg(max(v_pbl_new))'])**2)**(1/2)))

# COMMAND ----------

# test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels_read.select(*['trailing1d_min_Optical_Depth_047_scaled',
#                                                                       'trailing1d_max_Optical_Depth_047_scaled',
#                                                                       'trailing1d_median_Optical_Depth_047_scaled',
#                                                                         'day_of_year',
#                                                                         'month',
#                                                                         'week_of_year',
#                                                                         'Aerosol_Optical_Depth_scaled',
#                                                                         'min_Optical_Depth_047_scaled',
#                                                                         'max_Optical_Depth_047_scaled',
#                                                                         'median_Optical_Depth_047_scaled',
#                                                                         'min_Optical_Depth_055_scaled',
#                                                                         'max_Optical_Depth_055_scaled',
#                                                                         'median_Optical_Depth_055_scaled',
#                                                                         'Angstrom_Exponent_550_860nm',
#                                                                         'min_elevation',
#                                                                         'max_elevation',
#                                                                         'avg_elevation',
#                                                                         'avg(max(t_surface_new))',
#                                                                         'avg(max(pbl_surface_new))',
#                                                                         'avg(max(hindex_surface_new))',
#                                                                         'avg(max(gust_surface_new))',
#                                                                         'avg(max(r_atmosphere_new))',
#                                                                         'avg(max(pwat_atmosphere_new))',
#                                                                         'avg(max(vrate_pbl_new))',
#                                                                         'wind_speed',
#                                                                         'value',
#                                                                         'datetime',
#                                                                         'grid_id'
#                                                                         ])

# COMMAND ----------

test_full = test_aod_gfs_joined_with_labels_read.toPandas()
features_clean = [ col for col in test_full.columns ] 
test_clean = test_full[features_clean].values
# aod_gfs_joined_with_labels_test = aod_gfs_joined_with_labels_test.drop(*['datetime_utc','date_utc','grid_id','difflag'])

# COMMAND ----------

test_full.columns

# COMMAND ----------

train = tabnet_df.drop(*['Set']).toPandas()
test = test_aod_gfs_joined_with_labels_read.toPandas()
target = 'value'

# COMMAND ----------

USE_LAG = True

# COMMAND ----------

# Whether or not to use trailingPM2.5 data. 
if USE_LAG: 
    lag_feats = []
else: 
    lag_feats = [c for c in train.columns if 'lag' in c]

for col in train.columns[train.dtypes != object]:
    if col != target :#and col not in ['year','month','day']:
        train[col].fillna(train[col].mean(), inplace=True)
        test[col].fillna(train[col].mean(), inplace=True)
        
categorical_columns = []
# categorical_dims =  {}
for col in train.columns[train.dtypes == object]:
    l_enc = LabelEncoderExt()
    print(col, train[col].nunique())
    train[col] = train[col].fillna("VV_likely")
    test[col] = test[col].fillna("VV_likely")
    l_enc.fit(train[col])
    train[col] = l_enc.transform(train[col])
    test[col] = l_enc.transform(test[col])
    categorical_columns.append(col)
#     categorical_dims[col] = len(l_enc.classes_)

unused_feat = ['Set', 'rank', 'location', 'parameter', 'difflag'] + lag_feats

#Drop all categorical features as location based features can be learned through lat, lon. 
# features = [ col for col in train.columns if col not in unused_feat+categorical_columns+[target]] 
features = [ col for col in train.columns if col not in unused_feat+[target]] 

# categorical_columns = []
# categorical_dims =  {}
# for col in train.columns[train.dtypes == object]:
#     print(col, train[col].nunique())
#     l_enc = LabelEncoder()
#     train[col] = train[col].fillna("VV_likely")
#     if col != 'Set' and col != 'pm25_reading_date':
#         test[col] = test[col].fillna("VV_likely")
#     train[col] = l_enc.fit_transform(train[col].values)
#     if col != 'Set' and col != 'pm25_reading_date': 
#         test[col] = l_enc.transform(test[col].values)
#     categorical_columns.append(col)
#     categorical_dims[col] = len(l_enc.classes_)

# for col in train.columns[train.dtypes == 'float64']:
#     train.fillna(train[col].mean(), inplace=True)
#     test.fillna(train[col].mean(), inplace = True)

# COMMAND ----------

# unused_feat = ['Set', 'rank', 'location', 'parameter']

# features = [ col for col in train.columns if col not in unused_feat+[target]] 

# cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

# cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# # define your embedding sizes : here just a random choice
# cat_emb_dim = cat_dims

# COMMAND ----------

features

# COMMAND ----------

X_train = train[features].values
y_train = train[target].values.reshape(-1, 1)

X_test = test[features].values
y_test = test[target].values.reshape(-1, 1)

# COMMAND ----------

clf_xgb = XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=0.3959137968655604, colsample_bynode=0.38427772092378626, colsample_bytree=0.7366289714335655, enable_categorical=False, gamma=14.0281111983136, gpu_id=-1, importance_type=None, interaction_constraints='', learning_rate=0.5386955173692471, max_delta_step=6, max_depth=8, min_child_weight=0.4016920074761443, monotone_constraints='()', n_estimators=900, n_jobs=8, num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=22, reg_lambda=204, scale_pos_weight=1, subsample=0.8783345980525217, tree_method='exact', validate_parameters=1, verbosity=None)

# COMMAND ----------

clf_xgb.fit(X_train, y_train,
        verbose=10)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make predictions on test data. 

# COMMAND ----------

preds = np.array(clf_xgb.predict(X_train))
train_r2 = r2_score(y_pred=preds, y_true=y_train)
print(train_r2)

preds = np.array(clf_xgb.predict(X_test))
test_r2 = r2_score(y_pred=preds, y_true=y_test)
print(test_r2)

# COMMAND ----------

len(X_test)

# COMMAND ----------

len(preds)

# COMMAND ----------

plt.hist(preds, bins='auto')

# COMMAND ----------

df = pd.DataFrame(data = X_test, columns = features)

# COMMAND ----------

df['value'] = preds
df['actual'] = y_test
#df['date_utc'] = '2022-03-31'
df

# COMMAND ----------

df.sort_values(by = 'actual')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Calculate AQI Classification Error

# COMMAND ----------

def getAQIClassification(value): 
    if value <= 30.0: 
        aqi = 'Good'
    elif value <= 60.0: 
        aqi = 'Satisfactory'
    elif value <= 90.0: 
        aqi = 'Moderately Polluted'
    elif value <= 120.0: 
        aqi = 'Poor'
    elif value <= 250.0: 
        aqi = 'Very Poor'
    else: 
        aqi = 'Severe'
    
    return aqi 

# COMMAND ----------

actual_aqi = []
pred_aqi = []
aqi_hit = [] 

for index, row in df.iterrows():
    pred = row['prediction']
    actual = row['actual']
    
    act_aqi_class = getAQIClassification(actual)
    pred_aqi_class = getAQIClassification(pred)
    
    if act_aqi_class == pred_aqi_class: 
        aqi_class_hit = 1
    else: 
        aqi_class_hit = 0
   
    actual_aqi.append(act_aqi_class)
    pred_aqi.append(pred_aqi_class)
    aqi_hit.append(aqi_class_hit)
    
df['actual_aqi'] = actual_aqi
df['pred_aqi'] = pred_aqi
df['aqi_hit'] = aqi_hit
df['count'] = 1

# COMMAND ----------

# import datetime

# dates = []
# for index, row in df.iterrows():
#     dates.append(datetime.datetime(int(row['year']), int(row['month']), int(row['day'])))
    
# df['date'] = dates

# COMMAND ----------

df

# COMMAND ----------

spark_df = spark.createDataFrame(df) 

# COMMAND ----------

#Plot trailing 10 day by location
def get_past_10_days(lat, lon):
    where_condition = "round(latitude,2) =="+str(round(lat,2))+" and round(longitude,2) =="+str(round(lon,2))+ " and date >= date_sub(current_date(), 20)"
    df_trailing = spark_df.where(where_condition).sort('date', ascending = False)
    df_trailing = df_trailing.select('date', 'actual')
    df_trailing = df_trailing.toPandas()
    plt.bar(df_trailing['date'], df_trailing['actual'])

# COMMAND ----------

where_condition = "round(latitude,2) =="+str(round(33.830,2))+" and round(longitude,2) =="+str(round(-117.94,2))#+ " and date >= date_sub(current_date(), 20)"
spark_df.where(where_condition).agg({'date':'max'}).show()#count()

# COMMAND ----------

spark_df.agg({'date':'max'}).show()

# COMMAND ----------

get_past_10_days(33.830,	-117.94)

# COMMAND ----------

df_mean = df[['prediction', 'actual']].mean()
df_mean['rmse'] = ((df_mean['prediction'] - df_mean['actual'])**2)**0.5
df_mean

# COMMAND ----------

df_hits = df[['actual_aqi', 'aqi_hit', 'count']].groupby('actual_aqi').sum()
df_hits['percentage'] = df_hits['aqi_hit'] / df_hits['count']
df_hits

# COMMAND ----------

import seaborn as sns

# COMMAND ----------

plt.scatter(df['actual'], df['prediction'])


# COMMAND ----------

prediction = df['prediction']
actual = df['actual']

# COMMAND ----------

fig = sns.jointplot(x = actual, y = prediction, xlim = [0, 550], ylim = [0, 550])
fig.set_xlabel('Actual PM2.5')
fig.set_ylabel('Predicted PM2.5')
plt.ax_joint
plt.show()

# COMMAND ----------

plt.boxplot(df['actual'])

# COMMAND ----------

submission_df = df[['latitude', 'longitude', 'prediction']]
submission_df = submission_df.sort_values(by = ['datetime', 'grid_id'])

# COMMAND ----------

submission_df = df
submission_df_sp = spark.createDataFrame(submission_df)

# COMMAND ----------

# submission_df_sp = spark.createDataFrame(submission_df)

# COMMAND ----------

display(submission_df_sp)

# COMMAND ----------

# Filter for last day and only latitude and longitude. 
#submission_df = spark_df.where("day_of_month == 31.0 and month == 3.0 and year == 2022.0")

# COMMAND ----------

display(submission_df)

# COMMAND ----------

submission_df_sp.coalesce(1).write.option("header",True).mode("overwrite").csv("/mnt/capstone/test/submission_0421.csv") 

# COMMAND ----------

def generate_pred(lat, lon):
    try:
        gen_df = df
        gen_df = gen_df.astype({"latitude": float, "longitude": float})
        gen_df = gen_df.loc[(round(gen_df['latitude'],1)==round(lat,1)) & (round(gen_df['longitude'],1)==round(lon,1))]
        maxdate = gen_df['date_utc'].max()
        df_maxd = gen_df.loc[gen_df['date_utc']==maxdate]
        pred_value = df_maxd['value'].iloc[0]
        return pred_value
    except:
        return "we don't have data for your area yet, please check back in a few months."

# COMMAND ----------

generate_pred(34.0522342, -118.2436849)

# COMMAND ----------

gen_df = df
gen_df = gen_df.astype({"latitude": float, "longitude": float})
gen_df = gen_df.loc[(round(gen_df['latitude'],2)==round(lat,2)) & (round(gen_df['longitude'],2)==round(lon,2))]
maxdate = gen_df['date_utc'].max()
df_maxd = gen_df.loc[gen_df['date_utc']==maxdate]
pred_value = df_maxd['value'].iloc[0]
print(pred_value)

# COMMAND ----------

