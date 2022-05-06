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
# from pyspark.ml.regression import DecisionTreeRegressor
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.regression import GBTRegressor
#from pyspark.ml.regression import LinearRegression
# from sparkdl.xgboost import XgboostRegressor
# from pyspark.ml.feature import PCA

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

#Set desired stages to True to run. 
INIT_DATASETS = False
RUN_EDA = False
USE_IMPUTE = False
FEATURE_ENG_TIME = True
FEATURE_ENG_AVG = True
FEATURE_ENG_TRAILING = True
FEATURE_ENG_DELTA = True
IMPUTE_VALUES = False
RUN_BASELINES = False
RUN_CROSSVAL = False
RUN_SPLIT = False
disagg = True

# COMMAND ----------

if INIT_DATASETS == False and not disagg: 
#     # All AOD. 
    aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_gfs_elev_wlabels_joined.parquet")
    # Clean AOD 
#     aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined.parquet")
#     # Imputed AOD 
    #aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_wimputations.parquet")
    

# COMMAND ----------

if INIT_DATASETS == False and disagg: 
    aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_disaggregated.parquet")
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.where('AOD_distance_rank <= 20')
    
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA', aod_gfs_joined_with_labels.AOD_QA.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_qa_str', aod_gfs_joined_with_labels.AOD_qa_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA_Cloud_Mask_str', aod_gfs_joined_with_labels.AOD_QA_Cloud_Mask_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA_LWS_Mask_str', aod_gfs_joined_with_labels.AOD_QA_LWS_Mask_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA_Adj_Mask_str', aod_gfs_joined_with_labels.AOD_QA_Adj_Mask_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_Level_str', aod_gfs_joined_with_labels.AOD_Level_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('BRF_over_snow_str', aod_gfs_joined_with_labels.BRF_over_snow_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('BRF_climatology_str', aod_gfs_joined_with_labels.BRF_climatology_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA_SC_Mask_str', aod_gfs_joined_with_labels.AOD_QA_SC_Mask_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('Algo_init_str', aod_gfs_joined_with_labels.Algo_init_str.cast('int'))

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_distance_to_grid_center', aod_gfs_joined_with_labels.AOD_distance_to_grid_center.cast('float'))

    str_cols = [item[0] for item in aod_gfs_joined_with_labels.dtypes if item[1].startswith('string')]
    str_cols.append('value')

    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.groupBy(*str_cols).mean()
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.drop(*['avg(value)'])

# COMMAND ----------



# COMMAND ----------

df_misr_train = spark.read.parquet("/mnt/capstone/train/all_misr.parquet").drop(*['Month','Day','Year','Day_Of_Year','Hour','Minute'])

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.join(df_misr_train,
                                                      on=[aod_gfs_joined_with_labels.grid_id==df_misr_train.grid_id,
                                                        aod_gfs_joined_with_labels.datetime==df_misr_train.pm25_reading_date  ],
                                                      how="outer").drop(df_misr_train.grid_id).drop(df_misr_train.pm25_reading_date)

# COMMAND ----------

#averaging gfs time based features
cols_00 = [col for col in aod_gfs_joined_with_labels.columns if '00' in col]
cols_06 = [col for col in aod_gfs_joined_with_labels.columns if '06' in col]
cols_12 = [col for col in aod_gfs_joined_with_labels.columns if '12' in col]
cols_18 = [col for col in aod_gfs_joined_with_labels.columns if '18' in col]

# COMMAND ----------

def avg_gfs_time_cols(array):
    if array:
        return sum(filter(None, array))/len(array)
    else:
        return None 
avg_cols = udf(lambda array: sum(filter(None, array))/len(array), DoubleType())

for i in range(len(cols_00)):
    colName = cols_00[i].replace("00","")
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn(colName, avg_cols(sf.array(cols_00[i], cols_06[i],cols_12[i],cols_18[i])))

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.drop(*cols_00)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.drop(*cols_06)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.drop(*cols_12)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.drop(*cols_18)

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
    df = df.withColumn('day_of_month', dayofmonth(df.datetime))
    df = df.withColumn('day_of_week', dayofweek(df.datetime))
    df = df.withColumn('day_of_year', dayofyear(df.datetime))
    df = df.withColumn('week_of_year', weekofyear(df.datetime))
    df = df.withColumn('month', month(df.datetime))
    df = df.withColumn('year', year(df.datetime))
    return df

# COMMAND ----------

def feature_eng_avg(df):
    df = df.withColumn('avg_weekday_Optical_Depth_047', sf.mean('avg(avg(Optical_Depth_047))').over(Window.partitionBy('grid_id', 'day_of_week')))
    df = df.withColumn('avg_daily_Optical_Depth_047', sf.mean('avg(avg(Optical_Depth_047))').over(Window.partitionBy('grid_id', 'day_of_year')))
    df = df.withColumn('avg_weekly_Optical_Depth_047', sf.mean('avg(avg(Optical_Depth_047))').over(Window.partitionBy('grid_id', 'week_of_year')))
    df = df.withColumn('avg_monthly_Optical_Depth_047', sf.mean('avg(avg(Optical_Depth_047))').over(Window.partitionBy('grid_id', 'month')))
    df = df.withColumn('avg_yearly_Optical_Depth_047', sf.mean('avg(avg(Optical_Depth_047))').over(Window.partitionBy('grid_id', 'year')))
    return df

# COMMAND ----------

if FEATURE_ENG_TIME: 
    aod_gfs_joined_with_labels = feature_eng_time(aod_gfs_joined_with_labels)

# COMMAND ----------

if FEATURE_ENG_AVG:   
    aod_gfs_joined_with_labels = feature_eng_avg(aod_gfs_joined_with_labels)

# COMMAND ----------

cols_aod = [col for col in aod_gfs_joined_with_labels.columns if '_047' in col or '_055' in col]
cols_aod = cols_aod + ['Aerosol_Optical_Depth','Absorption_Aerosol_Optical_Depth','Nonspherical_Aerosol_Optical_Depth',
                 'Small_Mode_Aerosol_Optical_Depth','Medium_Mode_Aerosol_Optical_Depth','Large_Mode_Aerosol_Optical_Depth']

# COMMAND ----------

def aod_scale(x):
    if x:
        return (46.759*x)+7.1333
    else: 
        return None
aod_scale_udf = sf.udf(lambda x:aod_scale(x) ,DoubleType())

# COMMAND ----------

for col_aod in cols_aod:
    aod_gfs_joined_with_labels=aod_gfs_joined_with_labels.withColumn(col_aod+'_scaled',aod_scale_udf(aod_gfs_joined_with_labels[col_aod]))

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.drop(*cols_aod)

# COMMAND ----------

days = lambda i: i * 86400 

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing90d_Optical_Depth_047', sf.mean(aod_gfs_joined_with_labels['avg(avg(Optical_Depth_047))_scaled']).over(Window.partitionBy('grid_id').orderBy(aod_gfs_joined_with_labels['datetime'].cast("timestamp").cast("long")).rangeBetween(-days(90), 0)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing60d_Optical_Depth_047', sf.mean(aod_gfs_joined_with_labels['avg(avg(Optical_Depth_047))_scaled']).over(Window.partitionBy('grid_id').orderBy(aod_gfs_joined_with_labels['datetime'].cast("timestamp").cast("long")).rangeBetween(-days(60), 0)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing30d_Optical_Depth_047', sf.mean(aod_gfs_joined_with_labels['avg(avg(Optical_Depth_047))_scaled']).over(Window.partitionBy('grid_id').orderBy(aod_gfs_joined_with_labels['datetime'].cast("timestamp").cast("long")).rangeBetween(-days(30), 0)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing15d_Optical_Depth_047', sf.mean(aod_gfs_joined_with_labels['avg(avg(Optical_Depth_047))_scaled']).over(Window.partitionBy('grid_id').orderBy(aod_gfs_joined_with_labels['datetime'].cast("timestamp").cast("long")).rangeBetween(-days(15), 0)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing7d_Optical_Depth_047', sf.mean(aod_gfs_joined_with_labels['avg(avg(Optical_Depth_047))_scaled']).over(Window.partitionBy('grid_id').orderBy(aod_gfs_joined_with_labels['datetime'].cast("timestamp").cast("long")).rangeBetween(-days(7), 0)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing5d_Optical_Depth_047', sf.mean(aod_gfs_joined_with_labels['avg(avg(Optical_Depth_047))_scaled']).over(Window.partitionBy('grid_id').orderBy(aod_gfs_joined_with_labels['datetime'].cast("timestamp").cast("long")).rangeBetween(-days(5), 0)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing3d_Optical_Depth_047', sf.mean(aod_gfs_joined_with_labels['avg(avg(Optical_Depth_047))_scaled']).over(Window.partitionBy('grid_id').orderBy(aod_gfs_joined_with_labels['datetime'].cast("timestamp").cast("long")).rangeBetween(-days(3), 0)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing1d_Optical_Depth_047', sf.mean(aod_gfs_joined_with_labels['avg(avg(Optical_Depth_047))_scaled']).over(Window.partitionBy('grid_id').orderBy(aod_gfs_joined_with_labels['datetime'].cast("timestamp").cast("long")).rangeBetween(-days(1), 0)))

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed', 
                   (((aod_gfs_joined_with_labels['avg(max(u_pbl_new))']**2)+
                    (aod_gfs_joined_with_labels['avg(max(v_pbl_new))'])**2)**(1/2)))

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(-math.inf, np.nan)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(math.inf, np.nan)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(-np.inf, np.nan)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(np.nan, np.nan)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(math.nan, np.nan)

# COMMAND ----------

aod_gfs_joined_with_labels_STATIC = aod_gfs_joined_with_labels.withColumn('date', sf.to_date('datetime'))

# COMMAND ----------

aod_gfs_joined_with_labels_STATIC.dtypes

# COMMAND ----------

aod_gfs_joined_with_labels_stable = aod_gfs_joined_with_labels_STATIC.select(*['date',
'value',
'Aerosol_Optical_Depth_scaled',
'avg(min_elevation)',
'avg(max_elevation)',
'avg(avg_elevation)',
'avg(max(t_surface_new))',
'avg(max(pbl_surface_new))',
'avg(max(hindex_surface_new))',
'avg(max(gust_surface_new))',
'avg(max(r_atmosphere_new))',
'avg(max(pwat_atmosphere_new))',
'avg(max(vrate_pbl_new))', 
'wind_speed'
                                                                              ])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Split Strategy

# COMMAND ----------

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np
np.random.seed(0)


import os
#import wget
from pathlib import Path

# COMMAND ----------

split_strategy = 'random_day'

# COMMAND ----------

if split_strategy == 'time':
    tabnet_df = aod_gfs_joined_with_labels.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('pm25_reading_date')))

    #5-5-80-5-5 split
    # tabnet_df = tabnet_df.withColumn("Set", when((tabnet_df.rank < .08), "test") \
    #                                               .when((tabnet_df.rank < .16), "valid") \
    #                                                .when((tabnet_df.rank < .90), "train") \
    #                                                .when((tabnet_df.rank < .95), "valid") \
    #                                                .otherwise("test")).cache()

    #For 8-1-1 split: 
    tabnet_df = tabnet_df.withColumn("Set", when((tabnet_df.rank < .08), "train") \
                                                  .when((tabnet_df.rank < .09), "valid") \
                                                 .when((tabnet_df.rank < .10), "test") \
                                                 .when((tabnet_df.rank < .18), "train") \
                                                 .when((tabnet_df.rank < .19), "valid") \
                                               .when((tabnet_df.rank < .20), "test") \
                                               .when((tabnet_df.rank < .28), "train") \
                                               .when((tabnet_df.rank < .29), "valid") \
                                               .when((tabnet_df.rank < .30), "test") \
                                               .when((tabnet_df.rank < .38), "train") \
                                               .when((tabnet_df.rank < .39), "valid") \
                                               .when((tabnet_df.rank < .40), "test") \
                                               .when((tabnet_df.rank < .48), "train") \
                                               .when((tabnet_df.rank < .49), "valid") \
                                               .when((tabnet_df.rank < .50), "test") \
                                               .when((tabnet_df.rank < .58), "train") \
                                               .when((tabnet_df.rank < .59), "valid") \
                                               .when((tabnet_df.rank < .60), "test") \
                                               .when((tabnet_df.rank < .68), "train") \
                                               .when((tabnet_df.rank < .69), "valid") \
                                               .when((tabnet_df.rank < .70), "test") \
                                               .when((tabnet_df.rank < .78), "train") \
                                               .when((tabnet_df.rank < .79), "valid") \
                                               .when((tabnet_df.rank < .80), "test") \
                                               .when((tabnet_df.rank < .88), "train") \
                                               .when((tabnet_df.rank < .89), "valid") \
                                               .when((tabnet_df.rank < .90), "test") \
                                               .when((tabnet_df.rank < .98), "train") \
                                               .when((tabnet_df.rank < .99), "valid") \
                                               .otherwise("test")).cache()

# COMMAND ----------

if split_strategy == 'random_day':
    train_split, val_split, test_split = aod_gfs_joined_with_labels.select("date").distinct().randomSplit(weights=[0.8, 0.1, 0.1], seed = 43)

    train_split = train_split.withColumn("Set", lit("train"))
    val_split = val_split.withColumn("Set", lit("valid"))
    test_split = test_split.withColumn("Set", lit("test"))

    sets = train_split.union(val_split)
    sets = sets.union(test_split)
    
    tabnet_df = aod_gfs_joined_with_labels.join(sets, on = "date", how = "left")
    tabnet_df = tabnet_df.drop(*['aod_lon_lat_list', 'pm25_date_d', 'pm25_datetime_dt', 'rank', 'aod_reading_end',
                                 'datetime','pm25_reading_date', 'tz'])
    
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

categorical_columns = []
categorical_dims =  {}
for col in train.columns[train.dtypes == object]:
    print(col, train[col].nunique())
    l_enc = LabelEncoder()
    train[col] = train[col].fillna("VV_likely")
    train[col] = l_enc.fit_transform(train[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

for col in train.columns[train.dtypes != object]:
    if col != target:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)

# COMMAND ----------

aod_gfs_joined_with_labels_STATIC.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features to Test

# COMMAND ----------

features_to_test = 
['trailing1d_Optical_Depth_047',
'day_of_year',
'month',
'week_of_year',
'Aerosol_Optical_Depth_scaled'
'avg(avg(Optical_Depth_047))_scaled',
'avg(avg(Optical_Depth_055))_scaled',
'Angstrom_Exponent_550_860nm',
'avg(min_elevation)',
'avg(max_elevation)',
'avg(avg_elevation)',
'avg(max(t_surface_new))',
'avg(max(pbl_surface_new))',
'avg(max(hindex_surface_new))',
'avg(max(gust_surface_new))',
'avg(max(r_atmosphere_new))',
'avg(max(pwat_atmosphere_new))',
'avg(max(vrate_pbl_new))',
'wind_speed'
]

# COMMAND ----------

features = [ col for col in train.columns if col in features_to_test] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# define your embedding sizes : here just a random choice
cat_emb_dim = cat_dims

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices].reshape(-1, 1)

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices].reshape(-1, 1)

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices].reshape(-1, 1)


clf_xgb = XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=0.2484906463769062, colsample_bynode=0.9465727754932802, colsample_bytree=0.9249323530026272, enable_categorical=False, gamma=96.89886840840305, gpu_id=-1, importance_type=None, interaction_constraints='', learning_rate=0.5180448254093463, max_delta_step=10, max_depth=8, min_child_weight=0.07919184774601618, monotone_constraints='()', n_estimators=1200, n_jobs=8, num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=22, reg_lambda=167, scale_pos_weight=1, subsample=0.7134807755048374, tree_method='exact', validate_parameters=1, verbosity=None)

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



# COMMAND ----------

print(np.any(X_train==-math.inf))
print(np.any(X_valid==-math.inf))
print(np.any(X_test==-math.inf))
print(np.any(X_train==math.inf))
print(np.any(X_valid==math.inf))
print(np.any(X_test==math.inf))
print(np.any(X_train==np.nan))
print(np.any(X_valid==np.nan))
print(np.any(X_test==np.nan))
print(np.any(X_train==math.nan))
print(np.any(X_valid==math.nan))
print(np.any(X_test==math.nan))

# COMMAND ----------

print(np.any(y_train==-math.inf))
print(np.any(y_valid==-math.inf))
print(np.any(y_test==-math.inf))
print(np.any(y_train==math.inf))
print(np.any(y_valid==math.inf))
print(np.any(y_test==math.inf))
print(np.any(y_train==np.nan))
print(np.any(y_valid==np.nan))
print(np.any(y_test==np.nan))
print(np.any(y_train==math.nan))
print(np.any(y_valid==math.nan))
print(np.any(y_test==math.nan))

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

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from hyperopt import SparkTrials, STATUS_OK, Trials, fmin, hp, tpe

# COMMAND ----------

#for tabnet
max_epochs = 1000 if not os.getenv("CI", False) else 2
batch_size = 1024

def get_model_classes():
    model_classes = {}
    model_classes['knn'] = KNeighborsRegressor(algorithm = 'kd_tree', leaf_size = 10.0, metric = 'minkowski', 
                                               n_jobs = -1, n_neighbors = 10, weights = 'uniform')
    model_classes['svm'] = SVR() #Notes: Very slow, warnings about running on samples of 10,000+ 
    model_classes['rf'] = RandomForestRegressor(bootstrap=False, max_depth=10, 
                                                max_features=10, min_samples_split=6, n_estimators=1000)
    model_classes['dt'] = DecisionTreeRegressor(max_depth=3, max_features='log2', max_leaf_nodes=None, min_samples_leaf=7,
                                                min_weight_fraction_leaf=0.2, splitter='best')
    model_classes['et'] = ExtraTreesRegressor(max_depth=50, max_features = 0.6000000000000001, min_samples_leaf=35, min_samples_split=35)
    model_classes['xgb'] = XGBRegressor(booster='gbtree', colsample_bylevel=0.743079371939747, colsample_bynode=0.7394557416435046, 
                                        colsample_bytree=0.7884353161840154, gamma=82.75826646341258, 
                                        learning_rate=0.6387558131747334, max_delta_step=2, max_depth=3, 
                                        min_child_weight=0.6946157171640037, n_estimators=500, objective='reg:linear', 
                                        reg_alpha=36.0, reg_lambda=136.0, subsample=0.27536487629412465, 
                                        tree_method='approx', random_state=0)
    model_classes['ada'] = AdaBoostRegressor(learning_rate=0.07035918296230455, n_estimators=250, random_state=1)
    model_classes['gbm'] = GradientBoostingRegressor(ccp_alpha=0.06917074266241187, criterion='friedman_mse', 
                                                     learning_rate=0.0630618724403663, max_depth=4.324893248455548, 
                                                     max_features='log2', min_impurity_decrease=0.5237959697127571, min_samples_split=0.1,
                                                     n_estimators=400, subsample=0.7331159527517004, random_state=1)
    model_classes['lr'] = LinearRegression(normalize=True)
    model_classes['nb'] = GaussianNB() #Notes: not suited for our feature space
    model_classes['nn'] = MLPRegressor(activation='relu', alpha=0.18378581181719267, beta_1=0.9694838581230079, beta_2=0.9947866263919846,
                                       epsilon=0.11479126244480711, hidden_layer_sizes=(100, 200, 100),
                                       learning_rate='constant', learning_rate_init=0.06684280999788848, momentum=0.5383856147932831,
                                       shuffle=True, solver='sgd')
    model_classes['bag'] = BaggingRegressor(bootstrap=True, bootstrap_features=False, max_features=0.8973578908000865,
                                            max_samples=0.34318668319252643, n_estimators=300, oob_score=False)
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
        'booster': hp.choice('booster', ['gbtree', 'gblinear','dart']), 
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

    return get_stacking(models)

# COMMAND ----------

final_ensemble_model_nn = build_ensemble(X_train, X_valid, X_test, y_train, y_valid, y_test, 
                                      #Beware: svm, 
                                      #model_types_to_include=['nn'],
                                      model_types_to_include=['rf','dt','et','xgb','ada','gbm','bag'],
                                      mdl_early_stopping_rounds=40, tune_max_evals=40, tune=False)#,
#                    model_types_to_include=['svm','knn','rf','et','xgb','ada','gbm','lr','nb','nn','bag'])

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

display(dbutils.fs.ls("/mnt/capstone/model"))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Pretrain

# COMMAND ----------

# # TabNetPretrainer
# unsupervised_model = TabNetPretrainer(
#     cat_idxs=cat_idxs,
#     cat_dims=cat_dims,
#     cat_emb_dim=3,
#     optimizer_fn=torch.optim.Adam,
#     optimizer_params=dict(lr=2e-2),
#     mask_type='entmax', # "sparsemax",
# #     n_shared_decoder=1, # nb shared glu for decoding
# #     n_indep_decoder=1, # nb independent glu for decoding
# )

# COMMAND ----------

# max_epochs = 1000 if not os.getenv("CI", False) else 2

# COMMAND ----------

# unsupervised_model.fit(
#     X_train=X_train,
#     eval_set=[X_valid],
#     max_epochs=max_epochs , patience=50,
#     batch_size=2048, virtual_batch_size=128,
#     num_workers=0,
#     drop_last=False,
#     pretraining_ratio=0.8,
# ) 

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
                         mask_type='entmax') # "sparsemax",)

# COMMAND ----------

tabnet.fit(
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

# COMMAND ----------

preds = tabnet.predict(X_test)

y_true = y_test

test_score = r2_score(y_pred=preds, y_true=y_true)
valid_score = r2_score(y_pred=tabnet.predict(X_valid), y_true=y_valid)

print(f"VALID SCORE FOR 6 Folds (~2018-2019): {valid_score}")
print(f"TEST SCORE FOR ~ 2020: {test_score}")

# COMMAND ----------

#tabnet.save_model('/mnt/capstone/model/tabnet_03_13')

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost

# COMMAND ----------

from xgboost import XGBRegressor

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


# COMMAND ----------

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
       'objective': hp.choice('objective', ['reg:squarederror', 'reg:linear']), 
       'booster': hp.choice('booster', ['gbtree', 'gblinear','dart']), 
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

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 40,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

best_model_new = trials.results[np.argmin([r['loss'] for r in  trials.results])]['model']

# COMMAND ----------

best_model_new

# COMMAND ----------

preds = np.array(best_model_new.predict(X_train))
train_r2 = r2_score(y_pred=preds, y_true=y_train)
print(train_r2)

preds = np.array(best_model_new.predict(X_valid))
valid_r2 = r2_score(y_pred=preds, y_true=y_valid)
print(valid_r2)

preds = np.array(best_model_new.predict(X_test))
test_r2 = r2_score(y_pred=preds, y_true=y_test)
print(test_r2)

# COMMAND ----------

best_model_new.save_model('/mnt/capstone/model/aod_filtered_gfs_elev_wlabels_joined_randomsplit.json')

# COMMAND ----------

saved_model = XGBRegressor()
saved_model.load_model('dbfs:/mnt/capstone/model/aod_filtered_gfs_elev_wlabels_joined_prepostsplit.json')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Submission

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Retrain Best Model on all Train Data

# COMMAND ----------

test_aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/test/aod_gfs_elev_joined_disagg.parquet")

submission = test_aod_gfs_joined_with_labels.select('datetime', 'grid_id')

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*['aod_lon_lat_list', 'pm25_date_d', 'pm25_datetime_dt', 'aod_reading_end', 'rank'])

target = 'value'

# COMMAND ----------

display(test_aod_gfs_joined_with_labels )

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.where('AOD_distance_rank <= 20 or AOD_distance_rank IS NULL')

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA', test_aod_gfs_joined_with_labels.AOD_QA.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_qa_str', test_aod_gfs_joined_with_labels.AOD_qa_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_Cloud_Mask_str', test_aod_gfs_joined_with_labels.AOD_QA_Cloud_Mask_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_LWS_Mask_str', test_aod_gfs_joined_with_labels.AOD_QA_LWS_Mask_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_Adj_Mask_str', test_aod_gfs_joined_with_labels.AOD_QA_Adj_Mask_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_Level_str', test_aod_gfs_joined_with_labels.AOD_Level_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('BRF_over_snow_str', test_aod_gfs_joined_with_labels.BRF_over_snow_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('BRF_climatology_str', test_aod_gfs_joined_with_labels.BRF_climatology_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_SC_Mask_str', test_aod_gfs_joined_with_labels.AOD_QA_SC_Mask_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('Algo_init_str', test_aod_gfs_joined_with_labels.Algo_init_str.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_Cloud_Mask', test_aod_gfs_joined_with_labels.AOD_QA_Cloud_Mask.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_LWS_Mask', test_aod_gfs_joined_with_labels.AOD_QA_LWS_Mask.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_Adj_Mask', test_aod_gfs_joined_with_labels.AOD_QA_Adj_Mask.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_Level', test_aod_gfs_joined_with_labels.AOD_Level.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('BRF_over_snow', test_aod_gfs_joined_with_labels.BRF_over_snow.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('BRF_climatology', test_aod_gfs_joined_with_labels.BRF_climatology.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_SC_Mask', test_aod_gfs_joined_with_labels.AOD_QA_SC_Mask.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('Algo_init', test_aod_gfs_joined_with_labels.Algo_init.cast('int'))

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_distance_to_grid_center', test_aod_gfs_joined_with_labels.AOD_distance_to_grid_center.cast('float'))

str_cols = [item[0] for item in test_aod_gfs_joined_with_labels.dtypes if item[1].startswith('string')]
str_cols.append('value')

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.groupBy(*str_cols).mean()
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*['avg(value)'])

# COMMAND ----------

#averaging gfs time based features
cols_00 = [col for col in test_aod_gfs_joined_with_labels.columns if '00' in col]
cols_06 = [col for col in test_aod_gfs_joined_with_labels.columns if '06' in col]
cols_12 = [col for col in test_aod_gfs_joined_with_labels.columns if '12' in col]
cols_18 = [col for col in test_aod_gfs_joined_with_labels.columns if '18' in col]

# COMMAND ----------

def avg_gfs_time_cols(array):
    if array:
        return sum(filter(None, array))/len(array)
    else:
        return None 
avg_cols = udf(lambda array: sum(filter(None, array))/len(array), DoubleType())

for i in range(len(cols_00)):
    colName = cols_00[i].replace("00","")
    test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn(colName, avg_cols(sf.array(cols_00[i], cols_06[i],cols_12[i],cols_18[i])))

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*cols_00)
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*cols_06)
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*cols_12)
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*cols_18)

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_month', dayofmonth(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_week', dayofweek(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_year', dayofyear(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('week_of_year', weekofyear(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('month', month(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('year', year(test_aod_gfs_joined_with_labels.datetime))

# COMMAND ----------

cols_aod = [col for col in test_aod_gfs_joined_with_labels.columns if '_047' in col or '_055' in col]
cols_aod
for col_aod in cols_aod:
    test_aod_gfs_joined_with_labels=test_aod_gfs_joined_with_labels.withColumn(col_aod+'_scaled',
                                                                               aod_scale_udf(test_aod_gfs_joined_with_labels[col_aod]))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*cols_aod)

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('wind_speed', 
                   (((test_aod_gfs_joined_with_labels['avg(max(u_pbl_new))']**2)+
                     (test_aod_gfs_joined_with_labels['avg(max(v_pbl_new))'])**2)**(1/2)))

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_Cloud_Mask', test_aod_gfs_joined_with_labels['avg(AOD_QA_Cloud_Mask)'].cast('string')).drop(*['avg(AOD_QA_Cloud_Mask)'])

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_LWS_Mask', test_aod_gfs_joined_with_labels['avg(AOD_QA_LWS_Mask)'].cast('string')).drop(*['avg(AOD_QA_LWS_Mask)'])

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_Adj_Mask', test_aod_gfs_joined_with_labels['avg(AOD_QA_Adj_Mask)'].cast('string')).drop(*['avg(AOD_QA_Adj_Mask)'])

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_Level', test_aod_gfs_joined_with_labels['avg(AOD_Level)'].cast('string')).drop(*['avg(AOD_Level)'])

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('BRF_over_snow', test_aod_gfs_joined_with_labels['avg(BRF_over_snow)'].cast('string')).drop(*['avg(BRF_over_snow)'])

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('BRF_climatology', test_aod_gfs_joined_with_labels['avg(BRF_climatology)'].cast('string')).drop(*['avg(BRF_climatology)'])

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('AOD_QA_SC_Mask', test_aod_gfs_joined_with_labels['avg(AOD_QA_SC_Mask)'].cast('string')).drop(*['avg(AOD_QA_SC_Mask)'])

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('Algo_init', test_aod_gfs_joined_with_labels['avg(Algo_init)'].cast('string')).drop(*['avg(Algo_init)'])

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.replace(-math.inf, np.nan)
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.replace(math.inf, np.nan)
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.replace(-np.inf, np.nan)
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.replace(np.nan, np.nan)
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.replace(math.nan, np.nan)

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('date', sf.to_date('datetime'))

# COMMAND ----------

test_aod_gfs_joined_with_labels.select("grid_id", "date").distinct().count()

# COMMAND ----------

test_aod_gfs_joined_with_labels.count()

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop('value')

# COMMAND ----------

test_full = test_aod_gfs_joined_with_labels.toPandas()
features_clean = [ col for col in test_full.columns ] 
test_clean = test_full[features_clean].values
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*['datetime'])

# COMMAND ----------

checkdf = test_aod_gfs_joined_with_labels.groupby('grid_id', 'date').count()

# COMMAND ----------

checkdf[checkdf['count']>1].show()

# COMMAND ----------

train = tabnet_df.toPandas()
test = test_aod_gfs_joined_with_labels.toPandas()
target = 'value'

# COMMAND ----------

len(test)

# COMMAND ----------

categorical_columns = []
categorical_dims =  {}
for col in train.columns[train.dtypes == object]:
    print(col, train[col].nunique())
    l_enc = LabelEncoder()
    train[col] = train[col].fillna("VV_likely")
    if col != 'Set' and col != 'pm25_reading_date':
        test[col] = test[col].fillna("VV_likely")
    train[col] = l_enc.fit_transform(train[col].values)
    if col != 'Set' and col != 'pm25_reading_date': 
        test[col] = l_enc.fit_transform(test[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

for col in train.columns[train.dtypes != object]: 
    if col != target:
        train.fillna(train[col].mean(), inplace=True)
        test.fillna(train[col].mean(), inplace = True)

# COMMAND ----------

unused_feat = ['Set']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# define your embedding sizes : here just a random choice
cat_emb_dim = cat_dims

# COMMAND ----------

X_train = train[features].values
y_train = train[target].values.reshape(-1, 1)

X_test = test[features].values
# y_test = test[target].values.reshape(-1, 1)

# COMMAND ----------

final_ensemble_model.get_params()

# COMMAND ----------

# Fit tuned best model on full train data. 
final_ensemble_model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make predictions on test data. 

# COMMAND ----------

preds = np.array(final_ensemble_model.predict(X_test))

# COMMAND ----------

len(preds)

# COMMAND ----------

plt.hist(preds, bins='auto')

# COMMAND ----------

df = pd.DataFrame(data = test_clean, columns = features_clean)

# COMMAND ----------

df['value'] = preds
df

# COMMAND ----------

submission_df = df[['datetime', 'grid_id', 'value']]
submission_df = submission_df.sort_values(by = ['datetime', 'grid_id'])

# COMMAND ----------

submission_df

# COMMAND ----------

submission_df_sp = spark.createDataFrame(submission_df)

# COMMAND ----------

display(submission_df_sp)

# COMMAND ----------

submission_df_sp.coalesce(1).write.option("header",True).mode("overwrite").csv("/mnt/capstone/test/submission_03_20_ens_disagg_dist20_randomday.csv") 

# COMMAND ----------

