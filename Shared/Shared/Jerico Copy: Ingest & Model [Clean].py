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
FEATURE_ENG_AVG = False
FEATURE_ENG_TRAILING = False
FEATURE_ENG_DELTA = False
IMPUTE_VALUES = False
RUN_BASELINES = False
RUN_CROSSVAL = False
RUN_SPLIT = False

# COMMAND ----------

if INIT_DATASETS == False: 
#     # All AOD. 
    aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_gfs_elev_wlabels_joined.parquet")
#     aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_gfs_joined_with_labels.parquet")
    # Clean AOD 
#     aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined.parquet")
#     # Imputed AOD 
#     aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_wimputations.parquet")
    

# COMMAND ----------

GEO_df = spark.read.parquet("dbfs:/mnt/capstone/train/df_GEOS_agg.parquet")

# COMMAND ----------

GEO_df = GEO_df.withColumnRenamed('datetime', 'geo_datetime')
GEO_df = GEO_df.withColumnRenamed('grid_id', 'geo_grid_id')

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.join(GEO_df, on = [aod_gfs_joined_with_labels.grid_id == GEO_df.geo_grid_id, aod_gfs_joined_with_labels.datetime == GEO_df.geo_datetime], how = "left")

# df_GEOS_joined = train_labels_df.join(df_GEOS, on=[train_labels_df.grid_id == df_GEOS.grid_id,
#                              ((train_labels_df.datetime >= df_GEOS.time) 
#                                & (sf.date_sub(train_labels_df.datetime, 1) <= df_GEOS.time)),  
#                              ],
#                             how="left").cache()

# COMMAND ----------

display(aod_gfs_joined_with_labels)

# COMMAND ----------

aod_gfs_joined_with_labels.write.parquet("/mnt/capstone/train/aod_gfs_geo_joined_with_labels.parquet") 

# COMMAND ----------

# MAGIC %md
# MAGIC Add grid center coordinates so we can calculate distance to grid center for each row. 

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

#     #Download GFS data from s3 bucket to Databricks workspace. We will then load these files into a Spark DF. 
#     !aws s3 cp s3://capstone-particulate-storage/GFS/ train/GFS/ --no-sign-request --recursive
#     #Download GFS data from s3 bucket to Databricks workspace. We will then load these files into a Spark DF. 
#     !aws s3 cp s3://capstone-particulate-storage/geos/ train/GEOS/ --no-sign-request --recursive
#     #Download AOD data from s3 bucket to Databricsk workspace. 
#     !aws s3 cp s3://particulate-articulate-capstone/aod/ train/AOD/ --no-sign-request --recursive
#     #Download Elevation data
#     !aws s3 cp s3://particulate-articulate-capstone/elevation.parquet train/elevation/ --no-sign-request

# COMMAND ----------

!aws s3 cp s3://capstone-particulate-storage/GEOS/ train/GEOS/ --no-sign-request --recursive

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

display(train_labels_df)

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

# MAGIC %md
# MAGIC Open downloaded GEOS files and union into full Spark Dataframe. 

# COMMAND ----------

df_GEOS = directory_to_sparkDF(directory = 'train/GEOS/')

# COMMAND ----------

display(df_GEOS)

# COMMAND ----------

df_GEOS.write.parquet("/mnt/capstone/train/df_GEOS.parquet") 

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

df_GEOS = df_GEOS.withColumn('date', to_date(df_GEOS.time))

# COMMAND ----------

display(df_GEOS)

# COMMAND ----------

df_GEOS_joined = train_labels_df.join(df_GEOS, on=[train_labels_df.grid_id == df_GEOS.grid_id,
                             ((train_labels_df.datetime >= df_GEOS.time) 
                               & (sf.date_sub(train_labels_df.datetime, 1) <= df_GEOS.time)),  
                             ],
                            how="left").cache()

# COMMAND ----------



# COMMAND ----------

df_GEOS_joined = df_GEOS_joined.drop(*['value', 'location', 'tz', 'wkt', 'date', 'time', 'lev', 'lat', 'lon'])

# COMMAND ----------

df_GEOS_joined = df_GEOS_joined.drop(df_GEOS.grid_id)

# COMMAND ----------

display(df_GEOS_joined)

# COMMAND ----------

df_GEOS_agg = df_GEOS_joined.groupBy('datetime', 'grid_id').mean()

# COMMAND ----------

display(df_GEOS_agg)

# COMMAND ----------

df_GEOS_agg.write.parquet("/mnt/capstone/train/df_GEOS_agg.parquet") 

# COMMAND ----------

#Extract distinct grid_id, datetime from our clean dataframe to align times. 

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

# df_AOD.write.parquet("/mnt/capstone/train/df_AOD.parquet") 

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
# MAGIC ### 2.) Exploratory Data Analysis

# COMMAND ----------

def get_cols(df, drop_cols, label_col):      
    ''' Get all features, and categorize int_vars or str_vars for easy transformations in our Pipeline.
    Return all three lists of features. Drop any specified features and the label column.'''
    all_vars = []
    str_vars = []
    int_vars = []

    for col in df.dtypes:
        if col[0] not in drop_cols and col[0] not in [label_col]: 
            all_vars.append(col[0])
            if col[1] == 'string': 
                str_vars.append(col[0])
            else:
                #Drop lists from features 
                if 'array' not in col[1]: 
                    int_vars.append(col[0])
                    
    return all_vars, int_vars, str_vars

# COMMAND ----------

# Function Source: https://gist.github.com/cameres/bc24ac6711c9e537dd20be47b2a83558
def compute_correlation_matrix(df, method='pearson'):
    # wrapper around
    # https://forums.databricks.com/questions/3092/how-to-calculate-correlation-matrix-with-all-colum.html
    df_rdd = df.rdd.map(lambda row: row[0:])
    corr_mat = Statistics.corr(df_rdd, method=method)
    corr_mat_df = pd.DataFrame(corr_mat,
                    columns=df.columns, 
                    index=df.columns)
    return corr_mat_df

# COMMAND ----------

def plot_corr_matrix(correlations,attr,fig_no):
    fig=plt.figure(fig_no)
    ax=fig.add_subplot(111)
    ax.set_title("Correlation Matrix for Specified Attributes")
    ax.set_xticklabels(['']+attr)
    ax.set_yticklabels(['']+attr)
    cax=ax.matshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    plt.show()

# COMMAND ----------

if RUN_EDA: 
    # Get Correlation Matrix. 
    corr_mat_df = compute_correlation_matrix(df_GFS.select(num_vars), method = 'pearson')
    corr_mat_df

# COMMAND ----------

if RUN_EDA: 
    #Plot Correlation Matrix. 
    sns.set(rc = {'figure.figsize':(20,20)})
    plot_corr_matrix(corr_mat_df, (num_vars[:9]), 234)

# COMMAND ----------

if RUN_EDA: 
    #Create Tableau like dashboard to quickly explore all variables
    display(aod_gfs_joined_with_labels.select(features))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.) Feature Engineering

# COMMAND ----------

# MAGIC %md Date Components

# COMMAND ----------

if FEATURE_ENG_TIME:   
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_month', dayofmonth(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_week', dayofweek(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_year', dayofyear(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('week_of_year', weekofyear(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('month', month(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('year', year(aod_gfs_joined_with_labels.pm25_reading_date))

# COMMAND ----------

# MAGIC %md
# MAGIC Daily, Weekly, Monthly, Yearly Averages by Grid ID. 

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql import functions as F

if FEATURE_ENG_AVG:   
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('avg_weekday_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'day_of_week')))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('avg_daily_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'day_of_year')))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('avg_weekly_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'week_of_year')))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('avg_monthly_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'month')))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('avg_yearly_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id', 'year')))

# COMMAND ----------

# MAGIC %md
# MAGIC Trailing7d, Trailing15d, Trailing30d, Trailing90d averages by Year & Grid ID. 

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

days = lambda i: i * 86400 
if FEATURE_ENG_TRAILING: 
    # Trailing AOD 47
    
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing90d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(90), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing45d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(45), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing30d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(30), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing15d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(15), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing7d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(7), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing3d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(3), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing2d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(2), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing1d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(1), 0)))

#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing90d_Optical_Depth_055', sf.mean('median_Optical_Depth_055').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(90), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing45d_Optical_Depth_055', sf.mean('median_Optical_Depth_055').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(45), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing30d_Optical_Depth_055', sf.mean('median_Optical_Depth_055').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(30), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing15d_Optical_Depth_055', sf.mean('median_Optical_Depth_055').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(15), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing7d_Optical_Depth_055', sf.mean('median_Optical_Depth_055').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(7), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing3d_Optical_Depth_055', sf.mean('median_Optical_Depth_055').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(3), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing2d_Optical_Depth_055', sf.mean('median_Optical_Depth_055').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(2), 0)))
#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing1d_Optical_Depth_055', sf.mean('median_Optical_Depth_055').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(1), 0)))

# COMMAND ----------

# MAGIC %md Change in weather throughout day time18 - time00. 

# COMMAND ----------

if FEATURE_ENG_DELTA:  
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('change_in_pbl', aod_gfs_joined_with_labels['avg(pbl_surface18)'] - aod_gfs_joined_with_labels['avg(pbl_surface00)'])
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('change_in_t', aod_gfs_joined_with_labels['avg(t_surface18)'] - aod_gfs_joined_with_labels['avg(t_surface00)'])
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('change_in_v_pbl', aod_gfs_joined_with_labels['avg(v_pbl18)'] - aod_gfs_joined_with_labels['avg(v_pbl00)'])
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('change_vrate_pbl', aod_gfs_joined_with_labels['avg(vrate_pbl18)'] - aod_gfs_joined_with_labels['avg(vrate_pbl00)'])


# COMMAND ----------

aod_gfs_joined_with_labels.cache()

# COMMAND ----------

# MAGIC %md Particulate Transport Path Joins

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.) Impute Missing AOD Values

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
# MAGIC ### 5.) PM2.5 Prediction

# COMMAND ----------

# small_train_df = final_train_df.select('final_median_Optical_Depth_047', 'date', 'value')
# medium_train_df = final_train_df.select('final_median_Optical_Depth_047', 'avg(pbl_surface00)', 'avg(t_surface00)', 'avg(r_atmosphere00)', 'day_of_month', 'day_of_week', 'day_of_year', 'week_of_year', 'month', 'year', 'date', 'value')
# large_train_df = final_train_df.drop(*['grid_id', 'min_AOD_QA_Cloud_Mask', 'max_AOD_QA_Cloud_Mask', 'median_AOD_QA_Cloud_Mask', 'min_AOD_QA_LWS_Mask', 'max_AOD_QA_LWS_Mask', 'median_AOD_QA_LWS_Mask', 'min_AOD_QA_Adj_Mask', 'max_AOD_QA_Adj_Mask', 'median_AOD_QA_Adj_Mask', 
#                                       'min_AOD_Level', 'max_AOD_Level', 'median_AOD_Level', 'min_Algo_init', 'max_Algo_init', 'median_Algo_init', 'min_BRF_over_snow', 'max_BRF_over_snow', 'median_BRF_over_snow', 'min_BRF_climatology', 'max_BRF_climatology', 'median_BRF_climatology', 
#                                       'min_AOD_QA_SC_Mask', 'max_AOD_QA_SC_Mask', 'median_AOD_QA_SC_Mask', 'wkt', 'final_min_AOD_Uncertainty', 'final_max_AOD_Uncertainty', 'final_median_AOD_Uncertainty', 'final_min_Column_WV', 'final_max_Column_WV', 'final_median_Column_WV'])



# COMMAND ----------

# tp_train_df = final_train_df.filter(final_train_df.location == "Taipei")
# la_train_df = final_train_df.filter(final_train_df.location == "Los Angeles (SoCAB)")
# dl_train_df = final_train_df.filter(final_train_df.location == "Delhi")

# COMMAND ----------

# MAGIC %md Helper functions. 

# COMMAND ----------

# MAGIC %md Modified code for pyspark.ml.tuning to get a time-series valid, cross-validation class implementation. 

# COMMAND ----------

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import itertools
from multiprocessing.pool import ThreadPool

import numpy as np

from pyspark import keyword_only, since, SparkContext, inheritable_thread_target
from pyspark.ml import Estimator, Transformer, Model
from pyspark.ml.common import inherit_doc, _py2java, _java2py
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasCollectSubModels, HasParallelism, HasSeed
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MetaAlgorithmReadWrite,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
    JavaMLReader,
    JavaMLWriter,
)
from pyspark.ml.wrapper import JavaParams, JavaEstimator, JavaWrapper
from pyspark.sql.functions import col, lit, rand, UserDefinedFunction
from pyspark.sql.types import BooleanType

__all__ = [
    "ParamGridBuilder",
    "CrossValidator",
    "CrossValidatorModel",
    "TrainValidationSplit",
    "TrainValidationSplitModel",
]


def _parallelFitTasks(est, train, eva, validation, epm, collectSubModels):
    """
    Creates a list of callables which can be called from different threads to fit and evaluate
    an estimator in parallel. Each callable returns an `(index, metric)` pair.
    Parameters
    ----------
    est : :py:class:`pyspark.ml.baseEstimator`
        he estimator to be fit.
    train : :py:class:`pyspark.sql.DataFrame`
        DataFrame, training data set, used for fitting.
    eva : :py:class:`pyspark.ml.evaluation.Evaluator`
        used to compute `metric`
    validation : :py:class:`pyspark.sql.DataFrame`
        DataFrame, validation data set, used for evaluation.
    epm : :py:class:`collections.abc.Sequence`
        Sequence of ParamMap, params maps to be used during fitting & evaluation.
    collectSubModel : bool
        Whether to collect sub model.
    Returns
    -------
    tuple
        (int, float, subModel), an index into `epm` and the associated metric value.
    """
    modelIter = est.fitMultiple(train, epm)

    def singleTask():
        index, model = next(modelIter)
        # TODO: duplicate evaluator to take extra params from input
        #  Note: Supporting tuning params in evaluator need update method
        #  `MetaAlgorithmReadWrite.getAllNestedStages`, make it return
        #  all nested stages and evaluators
        metric = eva.evaluate(model.transform(validation, epm[index]))
        return index, metric, model if collectSubModels else None

    return [singleTask] * len(epm)


class ParamGridBuilder(object):
    r"""
    Builder for a param grid used in grid search-based model selection.
    .. versionadded:: 1.4.0
    Examples
    --------
    >>> from pyspark.ml.classification import LogisticRegression
    >>> lr = LogisticRegression()
    >>> output = ParamGridBuilder() \
    ...     .baseOn({lr.labelCol: 'l'}) \
    ...     .baseOn([lr.predictionCol, 'p']) \
    ...     .addGrid(lr.regParam, [1.0, 2.0]) \
    ...     .addGrid(lr.maxIter, [1, 5]) \
    ...     .build()
    >>> expected = [
    ...     {lr.regParam: 1.0, lr.maxIter: 1, lr.labelCol: 'l', lr.predictionCol: 'p'},
    ...     {lr.regParam: 2.0, lr.maxIter: 1, lr.labelCol: 'l', lr.predictionCol: 'p'},
    ...     {lr.regParam: 1.0, lr.maxIter: 5, lr.labelCol: 'l', lr.predictionCol: 'p'},
    ...     {lr.regParam: 2.0, lr.maxIter: 5, lr.labelCol: 'l', lr.predictionCol: 'p'}]
    >>> len(output) == len(expected)
    True
    >>> all([m in expected for m in output])
    True
    """

    def __init__(self):
        self._param_grid = {}

    @since("1.4.0")
    def addGrid(self, param, values):
        """
        Sets the given parameters in this grid to fixed values.
        param must be an instance of Param associated with an instance of Params
        (such as Estimator or Transformer).
        """
        if isinstance(param, Param):
            self._param_grid[param] = values
        else:
            raise TypeError("param must be an instance of Param")

        return self

    @since("1.4.0")
    def baseOn(self, *args):
        """
        Sets the given parameters in this grid to fixed values.
        Accepts either a parameter dictionary or a list of (parameter, value) pairs.
        """
        if isinstance(args[0], dict):
            self.baseOn(*args[0].items())
        else:
            for (param, value) in args:
                self.addGrid(param, [value])

        return self

    @since("1.4.0")
    def build(self):
        """
        Builds and returns all combinations of parameters specified
        by the param grid.
        """
        keys = self._param_grid.keys()
        grid_values = self._param_grid.values()

        def to_key_value_pairs(keys, values):
            return [(key, key.typeConverter(value)) for key, value in zip(keys, values)]

        return [dict(to_key_value_pairs(keys, prod)) for prod in itertools.product(*grid_values)]


class _ValidatorParams(HasSeed):
    """
    Common params for TrainValidationSplit and CrossValidator.
    """

    estimator = Param(Params._dummy(), "estimator", "estimator to be cross-validated")
    estimatorParamMaps = Param(Params._dummy(), "estimatorParamMaps", "estimator param maps")
    evaluator = Param(
        Params._dummy(),
        "evaluator",
        "evaluator used to select hyper-parameters that maximize the validator metric",
    )

    @since("2.0.0")
    def getEstimator(self):
        """
        Gets the value of estimator or its default value.
        """
        return self.getOrDefault(self.estimator)

    @since("2.0.0")
    def getEstimatorParamMaps(self):
        """
        Gets the value of estimatorParamMaps or its default value.
        """
        return self.getOrDefault(self.estimatorParamMaps)

    @since("2.0.0")
    def getEvaluator(self):
        """
        Gets the value of evaluator or its default value.
        """
        return self.getOrDefault(self.evaluator)

    @classmethod
    def _from_java_impl(cls, java_stage):
        """
        Return Python estimator, estimatorParamMaps, and evaluator from a Java ValidatorParams.
        """

        # Load information from java_stage to the instance.
        estimator = JavaParams._from_java(java_stage.getEstimator())
        evaluator = JavaParams._from_java(java_stage.getEvaluator())
        if isinstance(estimator, JavaEstimator):
            epms = [
                estimator._transfer_param_map_from_java(epm)
                for epm in java_stage.getEstimatorParamMaps()
            ]
        elif MetaAlgorithmReadWrite.isMetaEstimator(estimator):
            # Meta estimator such as Pipeline, OneVsRest
            epms = _ValidatorSharedReadWrite.meta_estimator_transfer_param_maps_from_java(
                estimator, java_stage.getEstimatorParamMaps()
            )
        else:
            raise ValueError("Unsupported estimator used in tuning: " + str(estimator))

        return estimator, epms, evaluator

    def _to_java_impl(self):
        """
        Return Java estimator, estimatorParamMaps, and evaluator from this Python instance.
        """

        gateway = SparkContext._gateway
        cls = SparkContext._jvm.org.apache.spark.ml.param.ParamMap

        estimator = self.getEstimator()
        if isinstance(estimator, JavaEstimator):
            java_epms = gateway.new_array(cls, len(self.getEstimatorParamMaps()))
            for idx, epm in enumerate(self.getEstimatorParamMaps()):
                java_epms[idx] = self.getEstimator()._transfer_param_map_to_java(epm)
        elif MetaAlgorithmReadWrite.isMetaEstimator(estimator):
            # Meta estimator such as Pipeline, OneVsRest
            java_epms = _ValidatorSharedReadWrite.meta_estimator_transfer_param_maps_to_java(
                estimator, self.getEstimatorParamMaps()
            )
        else:
            raise ValueError("Unsupported estimator used in tuning: " + str(estimator))

        java_estimator = self.getEstimator()._to_java()
        java_evaluator = self.getEvaluator()._to_java()
        return java_estimator, java_epms, java_evaluator


class _ValidatorSharedReadWrite:
    @staticmethod
    def meta_estimator_transfer_param_maps_to_java(pyEstimator, pyParamMaps):
        pyStages = MetaAlgorithmReadWrite.getAllNestedStages(pyEstimator)
        stagePairs = list(map(lambda stage: (stage, stage._to_java()), pyStages))
        sc = SparkContext._active_spark_context

        paramMapCls = SparkContext._jvm.org.apache.spark.ml.param.ParamMap
        javaParamMaps = SparkContext._gateway.new_array(paramMapCls, len(pyParamMaps))

        for idx, pyParamMap in enumerate(pyParamMaps):
            javaParamMap = JavaWrapper._new_java_obj("org.apache.spark.ml.param.ParamMap")
            for pyParam, pyValue in pyParamMap.items():
                javaParam = None
                for pyStage, javaStage in stagePairs:
                    if pyStage._testOwnParam(pyParam.parent, pyParam.name):
                        javaParam = javaStage.getParam(pyParam.name)
                        break
                if javaParam is None:
                    raise ValueError("Resolve param in estimatorParamMaps failed: " + str(pyParam))
                if isinstance(pyValue, Params) and hasattr(pyValue, "_to_java"):
                    javaValue = pyValue._to_java()
                else:
                    javaValue = _py2java(sc, pyValue)
                pair = javaParam.w(javaValue)
                javaParamMap.put([pair])
            javaParamMaps[idx] = javaParamMap
        return javaParamMaps

    @staticmethod
    def meta_estimator_transfer_param_maps_from_java(pyEstimator, javaParamMaps):
        pyStages = MetaAlgorithmReadWrite.getAllNestedStages(pyEstimator)
        stagePairs = list(map(lambda stage: (stage, stage._to_java()), pyStages))
        sc = SparkContext._active_spark_context
        pyParamMaps = []
        for javaParamMap in javaParamMaps:
            pyParamMap = dict()
            for javaPair in javaParamMap.toList():
                javaParam = javaPair.param()
                pyParam = None
                for pyStage, javaStage in stagePairs:
                    if pyStage._testOwnParam(javaParam.parent(), javaParam.name()):
                        pyParam = pyStage.getParam(javaParam.name())
                if pyParam is None:
                    raise ValueError(
                        "Resolve param in estimatorParamMaps failed: "
                        + javaParam.parent()
                        + "."
                        + javaParam.name()
                    )
                javaValue = javaPair.value()
                if sc._jvm.Class.forName(
                    "org.apache.spark.ml.util.DefaultParamsWritable"
                ).isInstance(javaValue):
                    pyValue = JavaParams._from_java(javaValue)
                else:
                    pyValue = _java2py(sc, javaValue)
                pyParamMap[pyParam] = pyValue
            pyParamMaps.append(pyParamMap)
        return pyParamMaps

    @staticmethod
    def is_java_convertible(instance):
        allNestedStages = MetaAlgorithmReadWrite.getAllNestedStages(instance.getEstimator())
        evaluator_convertible = isinstance(instance.getEvaluator(), JavaParams)
        estimator_convertible = all(map(lambda stage: hasattr(stage, "_to_java"), allNestedStages))
        return estimator_convertible and evaluator_convertible

    @staticmethod
    def saveImpl(path, instance, sc, extraMetadata=None):
        numParamsNotJson = 0
        jsonEstimatorParamMaps = []
        for paramMap in instance.getEstimatorParamMaps():
            jsonParamMap = []
            for p, v in paramMap.items():
                jsonParam = {"parent": p.parent, "name": p.name}
                if (
                    (isinstance(v, Estimator) and not MetaAlgorithmReadWrite.isMetaEstimator(v))
                    or isinstance(v, Transformer)
                    or isinstance(v, Evaluator)
                ):
                    relative_path = f"epm_{p.name}{numParamsNotJson}"
                    param_path = os.path.join(path, relative_path)
                    numParamsNotJson += 1
                    v.save(param_path)
                    jsonParam["value"] = relative_path
                    jsonParam["isJson"] = False
                elif isinstance(v, MLWritable):
                    raise RuntimeError(
                        "ValidatorSharedReadWrite.saveImpl does not handle parameters of type: "
                        "MLWritable that are not Estimaor/Evaluator/Transformer, and if parameter "
                        "is estimator, it cannot be meta estimator such as Validator or OneVsRest"
                    )
                else:
                    jsonParam["value"] = v
                    jsonParam["isJson"] = True
                jsonParamMap.append(jsonParam)
            jsonEstimatorParamMaps.append(jsonParamMap)

        skipParams = ["estimator", "evaluator", "estimatorParamMaps"]
        jsonParams = DefaultParamsWriter.extractJsonParams(instance, skipParams)
        jsonParams["estimatorParamMaps"] = jsonEstimatorParamMaps

        DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, jsonParams)
        evaluatorPath = os.path.join(path, "evaluator")
        instance.getEvaluator().save(evaluatorPath)
        estimatorPath = os.path.join(path, "estimator")
        instance.getEstimator().save(estimatorPath)

    @staticmethod
    def load(path, sc, metadata):
        evaluatorPath = os.path.join(path, "evaluator")
        evaluator = DefaultParamsReader.loadParamsInstance(evaluatorPath, sc)
        estimatorPath = os.path.join(path, "estimator")
        estimator = DefaultParamsReader.loadParamsInstance(estimatorPath, sc)

        uidToParams = MetaAlgorithmReadWrite.getUidMap(estimator)
        uidToParams[evaluator.uid] = evaluator

        jsonEstimatorParamMaps = metadata["paramMap"]["estimatorParamMaps"]

        estimatorParamMaps = []
        for jsonParamMap in jsonEstimatorParamMaps:
            paramMap = {}
            for jsonParam in jsonParamMap:
                est = uidToParams[jsonParam["parent"]]
                param = getattr(est, jsonParam["name"])
                if "isJson" not in jsonParam or ("isJson" in jsonParam and jsonParam["isJson"]):
                    value = jsonParam["value"]
                else:
                    relativePath = jsonParam["value"]
                    valueSavedPath = os.path.join(path, relativePath)
                    value = DefaultParamsReader.loadParamsInstance(valueSavedPath, sc)
                paramMap[param] = value
            estimatorParamMaps.append(paramMap)

        return metadata, estimator, evaluator, estimatorParamMaps

    @staticmethod
    def validateParams(instance):
        estiamtor = instance.getEstimator()
        evaluator = instance.getEvaluator()
        uidMap = MetaAlgorithmReadWrite.getUidMap(estiamtor)

        for elem in [evaluator] + list(uidMap.values()):
            if not isinstance(elem, MLWritable):
                raise ValueError(
                    f"Validator write will fail because it contains {elem.uid} "
                    f"which is not writable."
                )

        estimatorParamMaps = instance.getEstimatorParamMaps()
        paramErr = (
            "Validator save requires all Params in estimatorParamMaps to apply to "
            "its Estimator, An extraneous Param was found: "
        )
        for paramMap in estimatorParamMaps:
            for param in paramMap:
                if param.parent not in uidMap:
                    raise ValueError(paramErr + repr(param))

    @staticmethod
    def getValidatorModelWriterPersistSubModelsParam(writer):
        if "persistsubmodels" in writer.optionMap:
            persistSubModelsParam = writer.optionMap["persistsubmodels"].lower()
            if persistSubModelsParam == "true":
                return True
            elif persistSubModelsParam == "false":
                return False
            else:
                raise ValueError(
                    f"persistSubModels option value {persistSubModelsParam} is invalid, "
                    f"the possible values are True, 'True' or False, 'False'"
                )
        else:
            return writer.instance.subModels is not None


_save_with_persist_submodels_no_submodels_found_err = (
    "When persisting tuning models, you can only set persistSubModels to true if the tuning "
    "was done with collectSubModels set to true. To save the sub-models, try rerunning fitting "
    "with collectSubModels set to true."
)


@inherit_doc
class CrossValidatorReader(MLReader):
    def __init__(self, cls):
        super(CrossValidatorReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if not DefaultParamsReader.isPythonParamsInstance(metadata):
            return JavaMLReader(self.cls).load(path)
        else:
            metadata, estimator, evaluator, estimatorParamMaps = _ValidatorSharedReadWrite.load(
                path, self.sc, metadata
            )
            cv = CrossValidator(
                estimator=estimator, estimatorParamMaps=estimatorParamMaps, evaluator=evaluator
            )
            cv = cv._resetUid(metadata["uid"])
            DefaultParamsReader.getAndSetParams(cv, metadata, skipParams=["estimatorParamMaps"])
            return cv


@inherit_doc
class CrossValidatorWriter(MLWriter):
    def __init__(self, instance):
        super(CrossValidatorWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path):
        _ValidatorSharedReadWrite.validateParams(self.instance)
        _ValidatorSharedReadWrite.saveImpl(path, self.instance, self.sc)


@inherit_doc
class CrossValidatorModelReader(MLReader):
    def __init__(self, cls):
        super(CrossValidatorModelReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if not DefaultParamsReader.isPythonParamsInstance(metadata):
            return JavaMLReader(self.cls).load(path)
        else:
            metadata, estimator, evaluator, estimatorParamMaps = _ValidatorSharedReadWrite.load(
                path, self.sc, metadata
            )
            numFolds = metadata["paramMap"]["numFolds"]
            bestModelPath = os.path.join(path, "bestModel")
            bestModel = DefaultParamsReader.loadParamsInstance(bestModelPath, self.sc)
            avgMetrics = metadata["avgMetrics"]
            if "stdMetrics" in metadata:
                stdMetrics = metadata["stdMetrics"]
            else:
                stdMetrics = None
            persistSubModels = ("persistSubModels" in metadata) and metadata["persistSubModels"]

            if persistSubModels:
                subModels = [[None] * len(estimatorParamMaps)] * numFolds
                for splitIndex in range(numFolds):
                    for paramIndex in range(len(estimatorParamMaps)):
                        modelPath = os.path.join(
                            path, "subModels", f"fold{splitIndex}", f"{paramIndex}"
                        )
                        subModels[splitIndex][paramIndex] = DefaultParamsReader.loadParamsInstance(
                            modelPath, self.sc
                        )
            else:
                subModels = None

            cvModel = CrossValidatorModel(
                bestModel, avgMetrics=avgMetrics, subModels=subModels, stdMetrics=stdMetrics
            )
            cvModel = cvModel._resetUid(metadata["uid"])
            cvModel.set(cvModel.estimator, estimator)
            cvModel.set(cvModel.estimatorParamMaps, estimatorParamMaps)
            cvModel.set(cvModel.evaluator, evaluator)
            DefaultParamsReader.getAndSetParams(
                cvModel, metadata, skipParams=["estimatorParamMaps"]
            )
            return cvModel


@inherit_doc
class CrossValidatorModelWriter(MLWriter):
    def __init__(self, instance):
        super(CrossValidatorModelWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path):
        _ValidatorSharedReadWrite.validateParams(self.instance)
        instance = self.instance
        persistSubModels = _ValidatorSharedReadWrite.getValidatorModelWriterPersistSubModelsParam(
            self
        )
        extraMetadata = {"avgMetrics": instance.avgMetrics, "persistSubModels": persistSubModels}
        if instance.stdMetrics:
            extraMetadata["stdMetrics"] = instance.stdMetrics

        _ValidatorSharedReadWrite.saveImpl(path, instance, self.sc, extraMetadata=extraMetadata)
        bestModelPath = os.path.join(path, "bestModel")
        instance.bestModel.save(bestModelPath)
        if persistSubModels:
            if instance.subModels is None:
                raise ValueError(_save_with_persist_submodels_no_submodels_found_err)
            subModelsPath = os.path.join(path, "subModels")
            for splitIndex in range(instance.getNumFolds()):
                splitPath = os.path.join(subModelsPath, f"fold{splitIndex}")
                for paramIndex in range(len(instance.getEstimatorParamMaps())):
                    modelPath = os.path.join(splitPath, f"{paramIndex}")
                    instance.subModels[splitIndex][paramIndex].save(modelPath)


class _CrossValidatorParams(_ValidatorParams):
    """
    Params for :py:class:`CrossValidator` and :py:class:`CrossValidatorModel`.
    .. versionadded:: 3.0.0
    """

    numFolds = Param(
        Params._dummy(),
        "numFolds",
        "number of folds for cross validation",
        typeConverter=TypeConverters.toInt,
    )

    foldCol = Param(
        Params._dummy(),
        "foldCol",
        "Param for the column name of user "
        + "specified fold number. Once this is specified, :py:class:`CrossValidator` "
        + "won't do random k-fold split. Note that this column should be integer type "
        + "with range [0, numFolds) and Spark will throw exception on out-of-range "
        + "fold numbers.",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self, *args):
        super(_CrossValidatorParams, self).__init__(*args)
        self._setDefault(numFolds=2, foldCol="")

    @since("1.4.0")
    def getNumFolds(self):
        """
        Gets the value of numFolds or its default value.
        """
        return self.getOrDefault(self.numFolds)

    @since("3.1.0")
    def getFoldCol(self):
        """
        Gets the value of foldCol or its default value.
        """
        return self.getOrDefault(self.foldCol)


class CrossValidator(
    Estimator, _CrossValidatorParams, HasParallelism, HasCollectSubModels, MLReadable, MLWritable
):
    """
    K-fold cross validation performs model selection by splitting the dataset into a set of
    non-overlapping randomly partitioned folds which are used as separate training and test datasets
    e.g., with k=3 folds, K-fold cross validation will generate 3 (training, test) dataset pairs,
    each of which uses 2/3 of the data for training and 1/3 for testing. Each fold is used as the
    test set exactly once.
    .. versionadded:: 1.4.0
    Examples
    --------
    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.evaluation import BinaryClassificationEvaluator
    >>> from pyspark.ml.linalg import Vectors
    >>> from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
    >>> import tempfile
    >>> dataset = spark.createDataFrame(
    ...     [(Vectors.dense([0.0]), 0.0),
    ...      (Vectors.dense([0.4]), 1.0),
    ...      (Vectors.dense([0.5]), 0.0),
    ...      (Vectors.dense([0.6]), 1.0),
    ...      (Vectors.dense([1.0]), 1.0)] * 10,
    ...     ["features", "label"])
    >>> lr = LogisticRegression()
    >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    >>> evaluator = BinaryClassificationEvaluator()
    >>> cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator,
    ...     parallelism=2)
    >>> cvModel = cv.fit(dataset)
    >>> cvModel.getNumFolds()
    3
    >>> cvModel.avgMetrics[0]
    0.5
    >>> path = tempfile.mkdtemp()
    >>> model_path = path + "/model"
    >>> cvModel.write().save(model_path)
    >>> cvModelRead = CrossValidatorModel.read().load(model_path)
    >>> cvModelRead.avgMetrics
    [0.5, ...
    >>> evaluator.evaluate(cvModel.transform(dataset))
    0.8333...
    >>> evaluator.evaluate(cvModelRead.transform(dataset))
    0.8333...
    """

    @keyword_only
    def __init__(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        numFolds=2,
        seed=None,
        parallelism=1,
        collectSubModels=False,
        foldCol="",
    ):
        """
        __init__(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                 seed=None, parallelism=1, collectSubModels=False, foldCol="")
        """
        super(CrossValidator, self).__init__()
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    @since("1.4.0")
    def setParams(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        numFolds=2,
        seed=None,
        parallelism=1,
        collectSubModels=False,
        foldCol="",
    ):
        """
        setParams(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                  seed=None, parallelism=1, collectSubModels=False, foldCol=""):
        Sets params for cross validator.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @since("2.0.0")
    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`.
        """
        return self._set(estimator=value)

    @since("2.0.0")
    def setEstimatorParamMaps(self, value):
        """
        Sets the value of :py:attr:`estimatorParamMaps`.
        """
        return self._set(estimatorParamMaps=value)

    @since("2.0.0")
    def setEvaluator(self, value):
        """
        Sets the value of :py:attr:`evaluator`.
        """
        return self._set(evaluator=value)

    @since("1.4.0")
    def setNumFolds(self, value):
        """
        Sets the value of :py:attr:`numFolds`.
        """
        return self._set(numFolds=value)

    @since("3.1.0")
    def setFoldCol(self, value):
        """
        Sets the value of :py:attr:`foldCol`.
        """
        return self._set(foldCol=value)

    def setSeed(self, value):
        """
        Sets the value of :py:attr:`seed`.
        """
        return self._set(seed=value)

    def setParallelism(self, value):
        """
        Sets the value of :py:attr:`parallelism`.
        """
        return self._set(parallelism=value)

    def setCollectSubModels(self, value):
        """
        Sets the value of :py:attr:`collectSubModels`.
        """
        return self._set(collectSubModels=value)

    @staticmethod
    def _gen_avg_and_std_metrics(metrics_all):
        avg_metrics = np.mean(metrics_all, axis=0)
        std_metrics = np.std(metrics_all, axis=0)
        return list(avg_metrics), list(std_metrics)

    def _fit(self, dataset):
        print("Running Custom CrossValidator Class")
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        metrics_all = [[0.0] * numModels for i in range(nFolds)]

        pool = ThreadPool(processes=self.getParallelism())
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        datasets = self._kFold(dataset)
        for i in range(nFolds):
            validation = datasets[i][1].cache()
            train = datasets[i][0].cache()

            tasks = map(
                inheritable_thread_target,
                _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam),
            )
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics_all[i][j] = metric
                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        metrics, std_metrics = CrossValidator._gen_avg_and_std_metrics(metrics_all)

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics, subModels, std_metrics))

    def _kFold(self, dataset):
        nFolds = self.getOrDefault(self.numFolds)
        foldCol = self.getOrDefault(self.foldCol)

        datasets = []
        if not foldCol:
            # Do random k-fold split.
            seed = self.getOrDefault(self.seed)
            h = 1.0 / nFolds
            randCol = self.uid + "_rand"
            df = dataset.select("*", rand(seed).alias(randCol))
            for i in range(nFolds):
                validateLB = i * h
                validateUB = (i + 1) * h
                condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
                validation = df.filter(condition)
                train = df.filter(~condition)
                datasets.append((train, validation))
        else:
            # Use user-specified fold numbers.
            def checker(foldNum):
                if foldNum < 0 or foldNum >= (nFolds*2):
                    raise ValueError(
                        "Fold number must be in range [0, %s), but got %s." % (nFolds, foldNum)
                    )
                return True

            checker_udf = UserDefinedFunction(checker, BooleanType())
            for i in range(nFolds*2):
            #Custom logic to use i as training, and i+1 as validation for i / 2 folds (since each fold is a training/val pair).
                if i % 2 == 0:
                    training = dataset.filter(checker_udf(dataset[foldCol]) & (col(foldCol) == lit(i)))
                    validation = dataset.filter(
                        checker_udf(dataset[foldCol]) & (col(foldCol) == lit((i+1)))
                    )
                    if training.rdd.getNumPartitions() == 0 or len(training.take(1)) == 0:
                        raise ValueError("The training data at fold %s is empty." % i)
                    if validation.rdd.getNumPartitions() == 0 or len(validation.take(1)) == 0:
                        raise ValueError("The validation data at fold %s is empty." % i+1)
                    datasets.append((training, validation))

        return datasets

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies creates a deep copy of
        the embedded paramMap, and copies the embedded and extra parameters over.
        .. versionadded:: 1.4.0
        Parameters
        ----------
        extra : dict, optional
            Extra parameters to copy to the new instance
        Returns
        -------
        :py:class:`CrossValidator`
            Copy of this instance
        """
        if extra is None:
            extra = dict()
        newCV = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newCV.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMaps remain the same
        if self.isSet(self.evaluator):
            newCV.setEvaluator(self.getEvaluator().copy(extra))
        return newCV

    @since("2.3.0")
    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        if _ValidatorSharedReadWrite.is_java_convertible(self):
            return JavaMLWriter(self)
        return CrossValidatorWriter(self)

    @classmethod
    @since("2.3.0")
    def read(cls):
        """Returns an MLReader instance for this class."""
        return CrossValidatorReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java CrossValidator, create and return a Python wrapper of it.
        Used for ML persistence.
        """

        estimator, epms, evaluator = super(CrossValidator, cls)._from_java_impl(java_stage)
        numFolds = java_stage.getNumFolds()
        seed = java_stage.getSeed()
        parallelism = java_stage.getParallelism()
        collectSubModels = java_stage.getCollectSubModels()
        foldCol = java_stage.getFoldCol()
        # Create a new instance of this stage.
        py_stage = cls(
            estimator=estimator,
            estimatorParamMaps=epms,
            evaluator=evaluator,
            numFolds=numFolds,
            seed=seed,
            parallelism=parallelism,
            collectSubModels=collectSubModels,
            foldCol=foldCol,
        )
        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java CrossValidator. Used for ML persistence.
        Returns
        -------
        py4j.java_gateway.JavaObject
            Java object equivalent to this instance.
        """

        estimator, epms, evaluator = super(CrossValidator, self)._to_java_impl()

        _java_obj = JavaParams._new_java_obj("org.apache.spark.ml.tuning.CrossValidator", self.uid)
        _java_obj.setEstimatorParamMaps(epms)
        _java_obj.setEvaluator(evaluator)
        _java_obj.setEstimator(estimator)
        _java_obj.setSeed(self.getSeed())
        _java_obj.setNumFolds(self.getNumFolds())
        _java_obj.setParallelism(self.getParallelism())
        _java_obj.setCollectSubModels(self.getCollectSubModels())
        _java_obj.setFoldCol(self.getFoldCol())

        return _java_obj


class CrossValidatorModel(Model, _CrossValidatorParams, MLReadable, MLWritable):
    """
    CrossValidatorModel contains the model with the highest average cross-validation
    metric across folds and uses this model to transform input data. CrossValidatorModel
    also tracks the metrics for each param map evaluated.
    .. versionadded:: 1.4.0
    Notes
    -----
    Since version 3.3.0, CrossValidatorModel contains a new attribute "stdMetrics",
    which represent standard deviation of metrics for each paramMap in
    CrossValidator.estimatorParamMaps.
    """

    def __init__(self, bestModel, avgMetrics=None, subModels=None, stdMetrics=None):
        super(CrossValidatorModel, self).__init__()
        #: best model from cross validation
        self.bestModel = bestModel
        #: Average cross-validation metrics for each paramMap in
        #: CrossValidator.estimatorParamMaps, in the corresponding order.
        self.avgMetrics = avgMetrics or []
        #: sub model list from cross validation
        self.subModels = subModels
        #: standard deviation of metrics for each paramMap in
        #: CrossValidator.estimatorParamMaps, in the corresponding order.
        self.stdMetrics = stdMetrics or []

    def _transform(self, dataset):
        return self.bestModel.transform(dataset)

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying bestModel,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.
        It does not copy the extra Params into the subModels.
        .. versionadded:: 1.4.0
        Parameters
        ----------
        extra : dict, optional
            Extra parameters to copy to the new instance
        Returns
        -------
        :py:class:`CrossValidatorModel`
            Copy of this instance
        """
        if extra is None:
            extra = dict()
        bestModel = self.bestModel.copy(extra)
        avgMetrics = list(self.avgMetrics)
        subModels = [
            [sub_model.copy() for sub_model in fold_sub_models]
            for fold_sub_models in self.subModels
        ]
        stdMetrics = list(self.stdMetrics)
        return self._copyValues(
            CrossValidatorModel(bestModel, avgMetrics, subModels, stdMetrics), extra=extra
        )

    @since("2.3.0")
    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        if _ValidatorSharedReadWrite.is_java_convertible(self):
            return JavaMLWriter(self)
        return CrossValidatorModelWriter(self)

    @classmethod
    @since("2.3.0")
    def read(cls):
        """Returns an MLReader instance for this class."""
        return CrossValidatorModelReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java CrossValidatorModel, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        sc = SparkContext._active_spark_context
        bestModel = JavaParams._from_java(java_stage.bestModel())
        avgMetrics = _java2py(sc, java_stage.avgMetrics())
        estimator, epms, evaluator = super(CrossValidatorModel, cls)._from_java_impl(java_stage)

        py_stage = cls(bestModel=bestModel, avgMetrics=avgMetrics)
        params = {
            "evaluator": evaluator,
            "estimator": estimator,
            "estimatorParamMaps": epms,
            "numFolds": java_stage.getNumFolds(),
            "foldCol": java_stage.getFoldCol(),
            "seed": java_stage.getSeed(),
        }
        for param_name, param_val in params.items():
            py_stage = py_stage._set(**{param_name: param_val})

        if java_stage.hasSubModels():
            py_stage.subModels = [
                [JavaParams._from_java(sub_model) for sub_model in fold_sub_models]
                for fold_sub_models in java_stage.subModels()
            ]

        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java CrossValidatorModel. Used for ML persistence.
        Returns
        -------
        py4j.java_gateway.JavaObject
            Java object equivalent to this instance.
        """

        sc = SparkContext._active_spark_context
        _java_obj = JavaParams._new_java_obj(
            "org.apache.spark.ml.tuning.CrossValidatorModel",
            self.uid,
            self.bestModel._to_java(),
            _py2java(sc, self.avgMetrics),
        )
        estimator, epms, evaluator = super(CrossValidatorModel, self)._to_java_impl()

        params = {
            "evaluator": evaluator,
            "estimator": estimator,
            "estimatorParamMaps": epms,
            "numFolds": self.getNumFolds(),
            "foldCol": self.getFoldCol(),
            "seed": self.getSeed(),
        }
        for param_name, param_val in params.items():
            java_param = _java_obj.getParam(param_name)
            pair = java_param.w(param_val)
            _java_obj.set(pair)

        if self.subModels is not None:
            java_sub_models = [
                [sub_model._to_java() for sub_model in fold_sub_models]
                for fold_sub_models in self.subModels
            ]
            _java_obj.setSubModels(java_sub_models)
        return _java_obj


@inherit_doc
class TrainValidationSplitReader(MLReader):
    def __init__(self, cls):
        super(TrainValidationSplitReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if not DefaultParamsReader.isPythonParamsInstance(metadata):
            return JavaMLReader(self.cls).load(path)
        else:
            metadata, estimator, evaluator, estimatorParamMaps = _ValidatorSharedReadWrite.load(
                path, self.sc, metadata
            )
            tvs = TrainValidationSplit(
                estimator=estimator, estimatorParamMaps=estimatorParamMaps, evaluator=evaluator
            )
            tvs = tvs._resetUid(metadata["uid"])
            DefaultParamsReader.getAndSetParams(tvs, metadata, skipParams=["estimatorParamMaps"])
            return tvs


@inherit_doc
class TrainValidationSplitWriter(MLWriter):
    def __init__(self, instance):
        super(TrainValidationSplitWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path):
        _ValidatorSharedReadWrite.validateParams(self.instance)
        _ValidatorSharedReadWrite.saveImpl(path, self.instance, self.sc)


@inherit_doc
class TrainValidationSplitModelReader(MLReader):
    def __init__(self, cls):
        super(TrainValidationSplitModelReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if not DefaultParamsReader.isPythonParamsInstance(metadata):
            return JavaMLReader(self.cls).load(path)
        else:
            metadata, estimator, evaluator, estimatorParamMaps = _ValidatorSharedReadWrite.load(
                path, self.sc, metadata
            )
            bestModelPath = os.path.join(path, "bestModel")
            bestModel = DefaultParamsReader.loadParamsInstance(bestModelPath, self.sc)
            validationMetrics = metadata["validationMetrics"]
            persistSubModels = ("persistSubModels" in metadata) and metadata["persistSubModels"]

            if persistSubModels:
                subModels = [None] * len(estimatorParamMaps)
                for paramIndex in range(len(estimatorParamMaps)):
                    modelPath = os.path.join(path, "subModels", f"{paramIndex}")
                    subModels[paramIndex] = DefaultParamsReader.loadParamsInstance(
                        modelPath, self.sc
                    )
            else:
                subModels = None

            tvsModel = TrainValidationSplitModel(
                bestModel, validationMetrics=validationMetrics, subModels=subModels
            )
            tvsModel = tvsModel._resetUid(metadata["uid"])
            tvsModel.set(tvsModel.estimator, estimator)
            tvsModel.set(tvsModel.estimatorParamMaps, estimatorParamMaps)
            tvsModel.set(tvsModel.evaluator, evaluator)
            DefaultParamsReader.getAndSetParams(
                tvsModel, metadata, skipParams=["estimatorParamMaps"]
            )
            return tvsModel


@inherit_doc
class TrainValidationSplitModelWriter(MLWriter):
    def __init__(self, instance):
        super(TrainValidationSplitModelWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path):
        _ValidatorSharedReadWrite.validateParams(self.instance)
        instance = self.instance
        persistSubModels = _ValidatorSharedReadWrite.getValidatorModelWriterPersistSubModelsParam(
            self
        )

        extraMetadata = {
            "validationMetrics": instance.validationMetrics,
            "persistSubModels": persistSubModels,
        }
        _ValidatorSharedReadWrite.saveImpl(path, instance, self.sc, extraMetadata=extraMetadata)
        bestModelPath = os.path.join(path, "bestModel")
        instance.bestModel.save(bestModelPath)
        if persistSubModels:
            if instance.subModels is None:
                raise ValueError(_save_with_persist_submodels_no_submodels_found_err)
            subModelsPath = os.path.join(path, "subModels")
            for paramIndex in range(len(instance.getEstimatorParamMaps())):
                modelPath = os.path.join(subModelsPath, f"{paramIndex}")
                instance.subModels[paramIndex].save(modelPath)


class _TrainValidationSplitParams(_ValidatorParams):
    """
    Params for :py:class:`TrainValidationSplit` and :py:class:`TrainValidationSplitModel`.
    .. versionadded:: 3.0.0
    """

    trainRatio = Param(
        Params._dummy(),
        "trainRatio",
        "Param for ratio between train and\
     validation data. Must be between 0 and 1.",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self, *args):
        super(_TrainValidationSplitParams, self).__init__(*args)
        self._setDefault(trainRatio=0.75)

    @since("2.0.0")
    def getTrainRatio(self):
        """
        Gets the value of trainRatio or its default value.
        """
        return self.getOrDefault(self.trainRatio)


class TrainValidationSplit(
    Estimator,
    _TrainValidationSplitParams,
    HasParallelism,
    HasCollectSubModels,
    MLReadable,
    MLWritable,
):
    """
    Validation for hyper-parameter tuning. Randomly splits the input dataset into train and
    validation sets, and uses evaluation metric on the validation set to select the best model.
    Similar to :class:`CrossValidator`, but only splits the set once.
    .. versionadded:: 2.0.0
    Examples
    --------
    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.evaluation import BinaryClassificationEvaluator
    >>> from pyspark.ml.linalg import Vectors
    >>> from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
    >>> from pyspark.ml.tuning import TrainValidationSplitModel
    >>> import tempfile
    >>> dataset = spark.createDataFrame(
    ...     [(Vectors.dense([0.0]), 0.0),
    ...      (Vectors.dense([0.4]), 1.0),
    ...      (Vectors.dense([0.5]), 0.0),
    ...      (Vectors.dense([0.6]), 1.0),
    ...      (Vectors.dense([1.0]), 1.0)] * 10,
    ...     ["features", "label"]).repartition(1)
    >>> lr = LogisticRegression()
    >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    >>> evaluator = BinaryClassificationEvaluator()
    >>> tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator,
    ...     parallelism=1, seed=42)
    >>> tvsModel = tvs.fit(dataset)
    >>> tvsModel.getTrainRatio()
    0.75
    >>> tvsModel.validationMetrics
    [0.5, ...
    >>> path = tempfile.mkdtemp()
    >>> model_path = path + "/model"
    >>> tvsModel.write().save(model_path)
    >>> tvsModelRead = TrainValidationSplitModel.read().load(model_path)
    >>> tvsModelRead.validationMetrics
    [0.5, ...
    >>> evaluator.evaluate(tvsModel.transform(dataset))
    0.833...
    >>> evaluator.evaluate(tvsModelRead.transform(dataset))
    0.833...
    """

    @keyword_only
    def __init__(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        trainRatio=0.75,
        parallelism=1,
        collectSubModels=False,
        seed=None,
    ):
        """
        __init__(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, \
                 trainRatio=0.75, parallelism=1, collectSubModels=False, seed=None)
        """
        super(TrainValidationSplit, self).__init__()
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @since("2.0.0")
    @keyword_only
    def setParams(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        trainRatio=0.75,
        parallelism=1,
        collectSubModels=False,
        seed=None,
    ):
        """
        setParams(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, \
                  trainRatio=0.75, parallelism=1, collectSubModels=False, seed=None):
        Sets params for the train validation split.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @since("2.0.0")
    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`.
        """
        return self._set(estimator=value)

    @since("2.0.0")
    def setEstimatorParamMaps(self, value):
        """
        Sets the value of :py:attr:`estimatorParamMaps`.
        """
        return self._set(estimatorParamMaps=value)

    @since("2.0.0")
    def setEvaluator(self, value):
        """
        Sets the value of :py:attr:`evaluator`.
        """
        return self._set(evaluator=value)

    @since("2.0.0")
    def setTrainRatio(self, value):
        """
        Sets the value of :py:attr:`trainRatio`.
        """
        return self._set(trainRatio=value)

    def setSeed(self, value):
        """
        Sets the value of :py:attr:`seed`.
        """
        return self._set(seed=value)

    def setParallelism(self, value):
        """
        Sets the value of :py:attr:`parallelism`.
        """
        return self._set(parallelism=value)

    def setCollectSubModels(self, value):
        """
        Sets the value of :py:attr:`collectSubModels`.
        """
        return self._set(collectSubModels=value)

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(self.epm)
        eva = self.getOrDefault(self.evaluator)
        tRatio = self.getOrDefault(self.trainRatio)
        seed = self.getOrDefault(self.seed)
        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        condition = df[randCol] >= tRatio
        validation = df.filter(condition).cache()
        train = df.filter(~condition).cache()

        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [None for i in range(numModels)]

        tasks = map(
            inheritable_thread_target,
            _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam),
        )
        pool = ThreadPool(processes=self.getParallelism())
        metrics = [None] * numModels
        for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
            metrics[j] = metric
            if collectSubModelsParam:
                subModels[j] = subModel

        train.unpersist()
        validation.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(TrainValidationSplitModel(bestModel, metrics, subModels))

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies creates a deep copy of
        the embedded paramMap, and copies the embedded and extra parameters over.
        .. versionadded:: 2.0.0
        Parameters
        ----------
        extra : dict, optional
            Extra parameters to copy to the new instance
        Returns
        -------
        :py:class:`TrainValidationSplit`
            Copy of this instance
        """
        if extra is None:
            extra = dict()
        newTVS = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newTVS.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMaps remain the same
        if self.isSet(self.evaluator):
            newTVS.setEvaluator(self.getEvaluator().copy(extra))
        return newTVS

    @since("2.3.0")
    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        if _ValidatorSharedReadWrite.is_java_convertible(self):
            return JavaMLWriter(self)
        return TrainValidationSplitWriter(self)

    @classmethod
    @since("2.3.0")
    def read(cls):
        """Returns an MLReader instance for this class."""
        return TrainValidationSplitReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java TrainValidationSplit, create and return a Python wrapper of it.
        Used for ML persistence.
        """

        estimator, epms, evaluator = super(TrainValidationSplit, cls)._from_java_impl(java_stage)
        trainRatio = java_stage.getTrainRatio()
        seed = java_stage.getSeed()
        parallelism = java_stage.getParallelism()
        collectSubModels = java_stage.getCollectSubModels()
        # Create a new instance of this stage.
        py_stage = cls(
            estimator=estimator,
            estimatorParamMaps=epms,
            evaluator=evaluator,
            trainRatio=trainRatio,
            seed=seed,
            parallelism=parallelism,
            collectSubModels=collectSubModels,
        )
        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java TrainValidationSplit. Used for ML persistence.
        Returns
        -------
        py4j.java_gateway.JavaObject
            Java object equivalent to this instance.
        """

        estimator, epms, evaluator = super(TrainValidationSplit, self)._to_java_impl()

        _java_obj = JavaParams._new_java_obj(
            "org.apache.spark.ml.tuning.TrainValidationSplit", self.uid
        )
        _java_obj.setEstimatorParamMaps(epms)
        _java_obj.setEvaluator(evaluator)
        _java_obj.setEstimator(estimator)
        _java_obj.setTrainRatio(self.getTrainRatio())
        _java_obj.setSeed(self.getSeed())
        _java_obj.setParallelism(self.getParallelism())
        _java_obj.setCollectSubModels(self.getCollectSubModels())
        return _java_obj


class TrainValidationSplitModel(Model, _TrainValidationSplitParams, MLReadable, MLWritable):
    """
    Model from train validation split.
    .. versionadded:: 2.0.0
    """

    def __init__(self, bestModel, validationMetrics=None, subModels=None):
        super(TrainValidationSplitModel, self).__init__()
        #: best model from train validation split
        self.bestModel = bestModel
        #: evaluated validation metrics
        self.validationMetrics = validationMetrics or []
        #: sub models from train validation split
        self.subModels = subModels

    def _transform(self, dataset):
        return self.bestModel.transform(dataset)

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying bestModel,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.
        And, this creates a shallow copy of the validationMetrics.
        It does not copy the extra Params into the subModels.
        .. versionadded:: 2.0.0
        Parameters
        ----------
        extra : dict, optional
            Extra parameters to copy to the new instance
        Returns
        -------
        :py:class:`TrainValidationSplitModel`
            Copy of this instance
        """
        if extra is None:
            extra = dict()
        bestModel = self.bestModel.copy(extra)
        validationMetrics = list(self.validationMetrics)
        subModels = [model.copy() for model in self.subModels]
        return self._copyValues(
            TrainValidationSplitModel(bestModel, validationMetrics, subModels), extra=extra
        )

    @since("2.3.0")
    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        if _ValidatorSharedReadWrite.is_java_convertible(self):
            return JavaMLWriter(self)
        return TrainValidationSplitModelWriter(self)

    @classmethod
    @since("2.3.0")
    def read(cls):
        """Returns an MLReader instance for this class."""
        return TrainValidationSplitModelReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java TrainValidationSplitModel, create and return a Python wrapper of it.
        Used for ML persistence.
        """

        # Load information from java_stage to the instance.
        sc = SparkContext._active_spark_context
        bestModel = JavaParams._from_java(java_stage.bestModel())
        validationMetrics = _java2py(sc, java_stage.validationMetrics())
        estimator, epms, evaluator = super(TrainValidationSplitModel, cls)._from_java_impl(
            java_stage
        )
        # Create a new instance of this stage.
        py_stage = cls(bestModel=bestModel, validationMetrics=validationMetrics)
        params = {
            "evaluator": evaluator,
            "estimator": estimator,
            "estimatorParamMaps": epms,
            "trainRatio": java_stage.getTrainRatio(),
            "seed": java_stage.getSeed(),
        }
        for param_name, param_val in params.items():
            py_stage = py_stage._set(**{param_name: param_val})

        if java_stage.hasSubModels():
            py_stage.subModels = [
                JavaParams._from_java(sub_model) for sub_model in java_stage.subModels()
            ]

        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java TrainValidationSplitModel. Used for ML persistence.
        Returns
        -------
        py4j.java_gateway.JavaObject
            Java object equivalent to this instance.
        """

        sc = SparkContext._active_spark_context
        _java_obj = JavaParams._new_java_obj(
            "org.apache.spark.ml.tuning.TrainValidationSplitModel",
            self.uid,
            self.bestModel._to_java(),
            _py2java(sc, self.validationMetrics),
        )
        estimator, epms, evaluator = super(TrainValidationSplitModel, self)._to_java_impl()

        params = {
            "evaluator": evaluator,
            "estimator": estimator,
            "estimatorParamMaps": epms,
            "trainRatio": self.getTrainRatio(),
            "seed": self.getSeed(),
        }
        for param_name, param_val in params.items():
            java_param = _java_obj.getParam(param_name)
            pair = java_param.w(param_val)
            _java_obj.set(pair)

        if self.subModels is not None:
            java_sub_models = [sub_model._to_java() for sub_model in self.subModels]
            _java_obj.setSubModels(java_sub_models)

        return _java_obj


# COMMAND ----------

def compareBaselines(model = None, model_name = None, features = None, paramGrid = None, cv_train_data = None, validation_data = None, test_data = False):  
    '''Baseline model comparison: Similiar to the custom tuning function, this function will take a model, a feature list, training data, validation data. 
    It will train and test the model, appending the modelName, modelObject, featuresList, precision score (i.e. the model precision score) to a list of lists to use for comparison. 
    If model is None, predict never delayed as our 'null hypothesis' comparison.'''
    #If no model is passed, predict the majority class for our validation data (the odd numbered fold numbers in our foldCol). 
    if model is None: 
        #Append 0.0 literal to evaluation data as "prediction". 
        predictions = validation_data.withColumn('prediction_majority', f.lit(0.0))
        r2 = RegressionEvaluator(labelCol = label, predictionCol = 'prediction', metricName = 'r2')
        r2_score = r2.evaluate(predictions)
        stdDev = 0.0
        bestParams = None
        #Note we pass the paramGrid object with the baseline model so that we can easily extract the paramGrid to use for best model. 
        return [model_name, model, features, f_beta_score, stdDev, paramGrid, bestParams]
    else:
        pipeline = Pipeline(stages=[model])
        r2 = RegressionEvaluator(labelCol = label, predictionCol = 'prediction', metricName = 'r2')
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=r2, numFolds = 6, parallelism = 12, foldCol = 'foldNumber', collectSubModels = False)
        cvModel = cv.fit(cv_train_data)
        bestModel = cvModel.bestModel
        # Get average of performance metric (F-beta) for best model 
        r2_score = cvModel.avgMetrics[0]
        #Get standard deviation of performance metric (F-beta) for best model
        stdDev = cvModel.stdMetrics[0]
        #Get the best params
        bestParams = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]
        return [model_name, cvModel, features, r2_score, stdDev, paramGrid, bestParams]

# COMMAND ----------

def statisticalTestModels(df = None): 
    '''Takes a dateframe, sorted by increasing f_beta scores. Conducts two sample welch t-test for unequal variances
    between two rows to determine if the mean population f_beta score is significantly higher than previous model 'population', or due to chance.'''
    prev_r2 = None
    prev_std = None 
    p_value_to_prev = []

    for index, row in df.iterrows(): 
        if index > 0: 
            #Update current row values
            current_r2 = row['r2_score']
            current_std = row['r2_std_dev']
            #Two sample welch t-test for unequal variances
            p_value = stats.ttest_ind_from_stats(mean1 = current_r2, std1 = current_std, nobs1 = 6, mean2 = prev_r2, std2 = prev_std, nobs2 = 6, equal_var = False, alternative = 'greater').pvalue
            p_value_to_prev.append(p_value)
        else: 
            # Append null if on first row
            p_value_to_prev.append(None)

        #Update the previous row values
        prev_r2 = row['r2_score']
        prev_std = row['r2_std_dev']


    df['p_value_to_prev'] = p_value_to_prev
    return df

# COMMAND ----------

from matplotlib.pyplot import figure

# COMMAND ----------

def plotPerformance(df = None, col = 'r2_score', title = None): 
    '''Given Dataframe of model performance, 
    display formatted bar chart comparison of
    model performance.'''

    x = df['model_name']
    y = df[col]
    
    figure(figsize=(10, 6), dpi=100)
    
    plt.bar(x, y)
    for x,y in zip(x,y): 
        label = "{:.2f}".format(y)
        plt.annotate(label, # this is the text
                       (x,y), # these are the coordinates to position the label
                       textcoords="offset points", # how to position the text
                       xytext=(0,2), # distance from text to points (x,y)
                       ha='center')
    plt.xticks(rotation = 90)

    if title: 
        plt.title(title)

    plt.ylabel('Most Performant R2 Score')

    plt.show()

# COMMAND ----------

models = ['DT', 'GBT', 'TabNet', 'Ensemble', 'XGB', 'XGB on Test']
r2_scores = [0.51, 0.60, 0.76, 0.81, 0.83, 0.71]

d = {'model_name': models, 'r2_score': r2_scores}

df = pd.DataFrame(d)

plotPerformance(df, title = 'Model Performance by R2 Score')

# COMMAND ----------

# MAGIC %md Train / Val Split

# COMMAND ----------

if RUN_BASELINES or RUN_CROSSVAL:  
# if RUN_SPLIT: 
    #Start by only using our clean data for dates that we've pipelined both sources for.
    clean_df = aod_gfs_joined_with_labels

    # Add percent rank to aid in cross validation/splitting
    clean_df = clean_df.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('date')))
    
    if RUN_CROSSVAL == False or RUN_SPLIT == True:  
#         train_data, validation_data = clean_df.randomSplit([0.7, 0.3])
        # For non-Cross Validation (1 fold): 
        clean_df = clean_df.withColumn("foldNumber", when((clean_df.rank < .67), lit(0)) \
                                                   .otherwise(lit(1))).cache()
#                                                     .when((clean_df.rank > 0.66), lit(0))                                                  
#     else: 
#         #For 6 Folds: 
#         clean_df = clean_df.withColumn("foldNumber", when((clean_df.rank < .12), lit(0)) \
#                                                      .when((clean_df.rank < .17), lit(1)) \
#                                                      .when((clean_df.rank < .28), lit(2)) \
#                                                      .when((clean_df.rank < .33), lit(3)) \
#                                                    .when((clean_df.rank < .45), lit(4)) \
#                                                    .when((clean_df.rank < .50), lit(5)) \
#                                                    .when((clean_df.rank < .62), lit(6)) \
#                                                    .when((clean_df.rank < .67), lit(7)) \
#                                                    .when((clean_df.rank < .78), lit(8)) \
#                                                    .when((clean_df.rank < .83), lit(9)) \
#                                                    .when((clean_df.rank < .95), lit(10)) \
#                                                    .otherwise(lit(11))).cache()


    #Create a validation dataframe for our majority class prediction model. This will be all rows with odd numbers in "foldNumber" column (i.e i % 2 != 0). 
    train_data = clean_df.where("foldNumber % 2 == 0")
    validation_data = clean_df.where("foldNumber % 2 != 0")

# COMMAND ----------

# train_data = train_data.drop('aod_lon_lat_list')
# validation_data = validation_data.drop('aod_lon_lat_list')

# COMMAND ----------

# x_cols = train_data.columns
# x_cols.pop(x_cols.index("value"))

# COMMAND ----------

# train_data_reformatted = train_data.select(["value"]+x_cols)#.show(2)
# validation_data_reformatted = validation_data.select(["value"]+x_cols)

# COMMAND ----------

#Export CVS File for Tabnet

# import boto3
# from botocore import UNSIGNED
# from botocore.config import Config
# import io
# import pickle


# train_data_reformatted.write.csv('/databricks/driver/train_data_reformatted.csv')
# validation_data_reformatted.write.csv('/databricks/driver/validation_data_reformatted.csv')

# COMMAND ----------

#train_data_reformatted.write.csv("/mnt/capstone/train/train_data_reformatted.csv")

# COMMAND ----------

#validation_data_reformatted.write.csv("/mnt/capstone/train/validation_data_reformatted.csv")

# COMMAND ----------

#!pwd

# COMMAND ----------

# import os
# os.join('dbfs', databricks/driver/train_data.csv', 4)

# COMMAND ----------

# with open(os.path.join('/dbfs/databricks/driver', 'train_data.csv'), 'rb') as data:
#         s3.upload_fileobj(data, 'capstone-particulate-storage', 'train_data.csv')
# # with open('/dbfs/databricks/driver/val_data.csv', 'rb') as data:
# #         s3.upload_fileobj(data, 'capstone-particulate-storage', 'val_data.csv')

# COMMAND ----------

# !pwd

# COMMAND ----------

# MAGIC %md
# MAGIC Classify Features

# COMMAND ----------

if RUN_BASELINES or RUN_CROSSVAL:  
    # Categorize features into str or int feature lists. Drop desired features and drop the label column. Remove features that are of type array. 
    
    # Fill NA values with 0.0 (as recommended for Xgboost)
    data = train_data
    

    #Remove troublesome features
    drop_cols = ['aod_lon_lat_list', 'datetime', 'date', 'year']
    #Remove our train label (PM2.5 'value'). 
    label = 'value'

    features, int_features, str_features = get_cols(data, drop_cols, label)


# COMMAND ----------

# MAGIC %md Transform Features for PySpark Models. 

# COMMAND ----------

if RUN_BASELINES or RUN_CROSSVAL:     
    #For pipeline
    stages = []

    # Process strings into categorical vars first since we can't pass strings to VectorAssembler. 
    si = StringIndexer(
        inputCols=[col for col in str_features], 
        outputCols=[col + "_classVec" for col in str_features], 
        handleInvalid='keep'
    )
    #Add to pipeline. 
    stages += [si]

    #Indexed string features + non-processed int features. 
    assemblerInputs = [c + '_classVec' for c in str_features] + int_features

    #VectorAssembler to vectorize all features. 
    va = VectorAssembler(inputCols=assemblerInputs, outputCol="features_assembled", handleInvalid='keep')
    stages += [va]

    # Now use Vector Indexer to bin our categorical features into N bins. 
    vi = VectorIndexer(inputCol= 'features_assembled', outputCol= 'features_indexed', handleInvalid='keep', maxCategories = 10)
    #Add to pipeline. 
    stages += [vi]

    # Finally standardize features to limit size of feature space (i.e. Categorical Feature 4 has 1049 values which would require max bins = 1050). 
    # TODO: Investigate why VectorIndexer isn't binning into 4 categories already. 
    scaler = StandardScaler(inputCol="features_indexed", outputCol="features",
                            withStd=True, withMean=False)
#     scaler = StandardScaler(inputCol="features_assembled", outputCol="features", p=2.0)
    stages += [scaler]

    #Define model pipeline. 
    pipeline = Pipeline(stages = stages)

    #Fit transform on train data (excl. validation data to prevent leakage). 
    #Transform both train_data and validation_data. 
    pipelineModel = pipeline.fit(train_data)
    train_df = pipelineModel.transform(train_data)
    validation_df = pipelineModel.transform(validation_data)

# COMMAND ----------

# if RUN_BASELINES or RUN_CROSSVAL:     
#     #For pipeline
#     stages = []

#     # Process strings into categorical vars first since we can't pass strings to VectorAssembler. 
#     si = StringIndexer(
#         inputCols=[col for col in str_features], 
#         outputCols=[col + "_classVec" for col in str_features], 
#         handleInvalid='keep'
#     )
#     #Add to pipeline. 
#     stages += [si]
    
#     str_scaler = QuantileDiscretizer(numBuckets = 10, 
#                                  inputCols = [col + '_classVec' for col in str_features], 
#                                  outputCols = [col + '_quantDisc' for col in str_features],
#                                 )
#     int_scaler = QuantileDiscretizer(numBuckets = 10, 
#                                  inputCols = [col for col in int_features], 
#                                  outputCols = [col + '_quantDisc' for col in int_features],
#                                 )
#     stages += [str_scaler, int_scaler]
    
    
#     #Indexed string features + non-processed int features. 
#     assemblerInputs = [c + '_quantDisc' for c in str_features] + [c + '_quantDisc' for c in int_features]
    
    
#     #VectorAssembler to vectorize all features. 
#     va = VectorAssembler(inputCols=assemblerInputs, outputCol="features", handleInvalid='keep')
#     stages += [va]

#     #Define model pipeline. 
#     pipeline = Pipeline(stages = stages)

#     #Fit transform on train data (excl. validation data to prevent leakage). 
#     #Transform both train_data and validation_data. 
#     pipelineModel = pipeline.fit(train_data)
#     train_df = pipelineModel.transform(train_data)
#     validation_df = pipelineModel.transform(validation_data)

# COMMAND ----------

# MAGIC %md Create a unioned dataframe for CrossValidation, which is simply the train_df + validation_df. This follows valid fit on train and transform train & validation methodology to prevent leakage. 

# COMMAND ----------

if RUN_CROSSVAL:    
    crossVal_df = validation_df.union(train_df)
    modelComparisons = []

# COMMAND ----------

# MAGIC %md 1.) Decision Tree Regressor

# COMMAND ----------

baselineComparisons = []

# COMMAND ----------

if RUN_BASELINES:  
    #Define Decision Tree Regressor
    dtr = DecisionTreeRegressor(featuresCol = "features", labelCol= label)

    #Fit on train and predict (transform) on test. 
    dtr = dtr.fit(train_df)
    dtr_predictions = dtr.transform(validation_df)

    #Show a couple predictions
    dtr_predictions.select("prediction", label, "features").show(5)

    # Select (prediction, true label) and compute r2 and RMSE. 
    evaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
    r2 = evaluator.setMetricName('r2').evaluate(dtr_predictions)
    rmse = evaluator.setMetricName('rmse').evaluate(dtr_predictions)

    print(f'R-Squared on test data = {r2}')
    print(f'RMSE on test data = {rmse}')
    
    baselineComparisons.append(['Decision Tree', r2, rmse])
    
    featureImportance = pd.DataFrame(list(zip(va.getInputCols(), dtr.featureImportances)), columns = ['Feature', 'Importance']).sort_values(by = 'Importance', ascending = False).head(10)

# COMMAND ----------

featureImportance

# COMMAND ----------

#View tree. 
if RUN_BASELINES:
    display(dtr)

# COMMAND ----------

# MAGIC %md 2.) Gradient Boosted Tree

# COMMAND ----------

if RUN_BASELINES:      
    #Define Decision Tree Regressor
    gbt = GBTRegressor(featuresCol = "features", labelCol= label, maxIter = 10)

    #Fit on train and predict (transform) on test. 
    gbt = gbt.fit(train_df)
    gbt_predictions = gbt.transform(validation_df)

    #Show a couple predictions
    gbt_predictions.select("prediction", label, "features").show(5)

    # Select (prediction, true label) and compute r2. 
    evaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
    r2 = evaluator.setMetricName('r2').evaluate(gbt_predictions)
    rmse = evaluator.setMetricName('rmse').evaluate(gbt_predictions)

    print(f'R-Squared on test data = {r2}')
    print(f'RMSE on test data = {rmse}')
    
    baselineComparisons.append(['Gradient Boosted Tree', r2, rmse])
    
    featureImportance = pd.DataFrame(list(zip(va.getInputCols(), gbt.featureImportances)), columns = ['Feature', 'Importance']).sort_values(by = 'Importance', ascending = False).head(10)
    featureImportance

# COMMAND ----------

if RUN_BASELINES:
    featureImportance

# COMMAND ----------

featureImportance

# COMMAND ----------

# MAGIC %md 3.) XGBoost Regressor

# COMMAND ----------

if RUN_BASELINES:  
    # The next step is to define the model training stage of the pipeline. 
    # The following command defines a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.
    # If you are running Databricks Runtime for Machine Learning 9.0 ML or above, you can set the `num_workers` parameter to leverage the cluster for distributed training.
    xgb = XgboostRegressor(num_workers=8, featuresCol = "features", labelCol=label)
    #Fit on train and predict (transform) on test. 
    xgb = xgb.fit(train_df)
    xgb_predictions = xgb.transform(validation_df)

    #Show a couple predictions
    xgb_predictions.select("prediction", label, "features").show(5)

    # Select (prediction, true label) and compute r2. 
    evaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
    r2 = evaluator.setMetricName('r2').evaluate(xgb_predictions)
    rmse = evaluator.setMetricName('rmse').evaluate(xgb_predictions)
    
    baselineComparisons.append(['XGBoost', r2, rmse])

    print(f'R-Squared on test data = {r2}')
    print(f'RMSE on test data = {rmse}') 

# COMMAND ----------

 modelComparisonsDF = pd.DataFrame(baselineComparisons, columns = ['model_name', 'r2_score','rmse_score']).sort_values(by = 'r2_score').reset_index(drop=True)

# COMMAND ----------

plotPerformance(modelComparisonsDF, col = 'r2_score', title = 'Experimental R2 Scores on 80/20 Time-Series Split')

# COMMAND ----------

plotPerformance(modelComparisonsDF, col = 'rmse_score', title = 'Experimental RMSE Scores on 80/20 Time-Series Split')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Ensemble Model to Evaluation Level-0 Predictions of Baseline Models

# COMMAND ----------

display(dtr_predictions)

# COMMAND ----------

level_0_predictions = [dtr_predictions.prediction, gbt_predictions.prediction, xgb_predictions.prediction]

# COMMAND ----------

level_0_predictions

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Time-Series K-Fold CrossValidation on Models

# COMMAND ----------

if RUN_CROSSVAL:     
    # The next step is to define the model training stage of the pipeline. 
    # The following command defines a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.
    # If you are running Databricks Runtime for Machine Learning 9.0 ML or above, you can set the `num_workers` parameter to leverage the cluster for distributed training.
    
    #Define model objects. 
    dtr = DecisionTreeRegressor(featuresCol = "features", labelCol= label)
    gbt = GBTRegressor(featuresCol = "features", labelCol= label)
    xgb = XgboostRegressor(featuresCol = "features", labelCol=label)
    
    
    
    # Manually define non-random search space. This ensures data and params are controlled for in our experimental comparisons, yielding valid results. 
    MIN_INFO_GAIN_SEARCH_LIST = [0.0]
    MAX_DEPTH_SEARCH_LIST = [2, 3]

    # Define ParamGrids
    dt_paramGrid = ParamGridBuilder() \
        .addGrid(dtr.minInfoGain, MIN_INFO_GAIN_SEARCH_LIST) \
        .addGrid(dtr.maxDepth, MAX_DEPTH_SEARCH_LIST) \
        .build()
    
    gbt_paramGrid = ParamGridBuilder() \
        .addGrid(gbt.minInfoGain, MIN_INFO_GAIN_SEARCH_LIST) \
        .addGrid(gbt.maxDepth, MAX_DEPTH_SEARCH_LIST) \
        .build()
    
    
    xgb_paramGrid = ParamGridBuilder() \
        .addGrid(xgb.max_depth, MAX_DEPTH_SEARCH_LIST) \
        .build()

    
    
    
    modelList = [(dtr, "Decision Tree", dt_paramGrid),
             (gbt, "Gradient Boosted Tree", gbt_paramGrid),
             (xgb, "XGBoost", xgb_paramGrid)]
    
    #Create an empty list of lists that we will append models & performance metrics to.
    # Data order will be: model_name[str], model[obj], features[list], f_beta_score[float], f_beta_std_dev[float], paramGrid [obj] 
    modelComparisons = []
    
    #Build comparison table. 
    for model, model_name, paramGrid in modelList: 
        modelComparisons.append(compareBaselines(model = model, model_name = model_name, paramGrid = paramGrid, cv_train_data = crossVal_df, validation_data = None))

    #model_name[str], model[obj], features[list], precision[float]
    modelComparisonsDF = pd.DataFrame(modelComparisons, columns = ['model_name', 'model_obj','feature_names','r2_score', 'r2_std_dev', 'paramGrid_obj', 'bestParams']).sort_values(by = 'r2_score').reset_index(drop=True)
    
    modelComparisonsDF
    #Cross Validator for Time Series
#     pipeline = Pipeline(stages=[xgb])
#     r2 = RegressionEvaluator(labelCol = label, predictionCol = 'prediction', metricName = 'r2')
#     cv = CrossValidator(estimator=pipeline, estimatorParamMaps = xgb_paramGrid, evaluator = r2, numFolds = 6, parallelism = 4, foldCol = 'foldNumber', collectSubModels = False)
#     cvModel = cv.fit(crossVal_df)
#     bestModel = cvModel.bestModel
#     # Get average of performance metric (F-beta) for best model 
#     avg_r2 = cvModel.avgMetrics[0]
#     #Get standard deviation of performance metric (F-beta) for best model
#     stdDev = cvModel.stdMetrics[0]
#     #Get the best params
#     bestParams = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]

#     # return [model_name, cvModel, features, r2_score, stdDev, paramGrid, bestParams, runtime]

#     print(f'Average R2 on Cross Validation: {avg_r2}')
#     print(bestParams)

# COMMAND ----------

modelComparisonsDF

# COMMAND ----------

if RUN_CROSSVAL:  
    modelComparisonsDF = statisticalTestModels(modelComparisonsDF)
    modelComparisonsDF

# COMMAND ----------

if RUN_CROSSVAL:     
    plotPerformance(modelComparisonsDF, col = 'r2_score', title = 'Experimental R2 Scores on 6-Fold Cross-Validation')

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


import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np
np.random.seed(0)


import os
#import wget
from pathlib import Path

# COMMAND ----------

display(dbutils.fs.ls("/mnt/capstone/model"))

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.where("geo_grid_id IS NOT NULL")

# COMMAND ----------

display(aod_gfs_joined_with_labels)

# COMMAND ----------

train_split, val_split, test_split = aod_gfs_joined_with_labels.select("pm25_date_d").distinct().randomSplit(weights=[0.8, 0.1, 0.1], seed = 43)

# COMMAND ----------

train_split = train_split.withColumn("Set", lit("train"))
val_split = val_split.withColumn("Set", lit("valid"))
test_split = test_split.withColumn("Set", lit("test"))

sets = train_split.union(val_split)
sets = sets.union(test_split)

# COMMAND ----------

tabnet_df = aod_gfs_joined_with_labels.join(sets, on = "pm25_date_d", how = "left")

# COMMAND ----------

display(tabnet_df)

# COMMAND ----------

# tabnet_df = aod_gfs_joined_with_labels.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('datetime')))

# #5-5-80-5-5 split
# tabnet_df = tabnet_df.withColumn("Set", when((tabnet_df.rank < .08), "test") \
#                                               .when((tabnet_df.rank < .16), "valid") \
#                                                .when((tabnet_df.rank < .90), "train") \
#                                                .when((tabnet_df.rank < .95), "valid") \
#                                                .otherwise("test")).cache()
                                           
# #For 8-1-1 split: 
# tabnet_df = tabnet_df.withColumn("Set", when((tabnet_df.rank < .08), "train") \
#                                               .when((tabnet_df.rank < .09), "valid") \
#                                              .when((tabnet_df.rank < .10), "test") \
#                                              .when((tabnet_df.rank < .18), "train") \
#                                              .when((tabnet_df.rank < .19), "valid") \
#                                            .when((tabnet_df.rank < .20), "test") \
#                                            .when((tabnet_df.rank < .28), "train") \
#                                            .when((tabnet_df.rank < .29), "valid") \
#                                            .when((tabnet_df.rank < .30), "test") \
#                                            .when((tabnet_df.rank < .38), "train") \
#                                            .when((tabnet_df.rank < .39), "valid") \
#                                            .when((tabnet_df.rank < .40), "test") \
#                                            .when((tabnet_df.rank < .48), "train") \
#                                            .when((tabnet_df.rank < .49), "valid") \
#                                            .when((tabnet_df.rank < .50), "test") \
#                                            .when((tabnet_df.rank < .58), "train") \
#                                            .when((tabnet_df.rank < .59), "valid") \
#                                            .when((tabnet_df.rank < .60), "test") \
#                                            .when((tabnet_df.rank < .68), "train") \
#                                            .when((tabnet_df.rank < .69), "valid") \
#                                            .when((tabnet_df.rank < .70), "test") \
#                                            .when((tabnet_df.rank < .78), "train") \
#                                            .when((tabnet_df.rank < .79), "valid") \
#                                            .when((tabnet_df.rank < .80), "test") \
#                                            .when((tabnet_df.rank < .88), "train") \
#                                            .when((tabnet_df.rank < .89), "valid") \
#                                            .when((tabnet_df.rank < .90), "test") \
#                                            .when((tabnet_df.rank < .98), "train") \
#                                            .when((tabnet_df.rank < .99), "valid") \
#                                            .otherwise("test")).cache()
                                           

# COMMAND ----------

tabnet_df = tabnet_df.drop(*['aod_lon_lat_list', 'pm25_date_d', 'pm25_datetime_dt', 'rank', 'aod_reading_end', 'wkt', 'pm25_reading_date', 'tz', 'datetime']) #'avg(pm25_rh35_gcc)', 'avg(co)', 'avg(no2)', 'avg(o3)', 'avg(so2)'
train = tabnet_df.toPandas()
target = 'value'

# if "Set" not in train.columns:
#     train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

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

for col in train.columns[train.dtypes == 'float64']:
    train.fillna(train.loc[train_indices, col].mean(), inplace=True)



# categorical_columns = []
# categorical_dims =  {}
# for col in train.columns[train.dtypes == object]:
#     print(col, train[col].nunique())
#     l_enc = LabelEncoder()
#     train[col] = train[col].fillna("VV_likely")
#     train[col] = l_enc.fit_transform(train[col].values)
#     categorical_columns.append(col)
#     categorical_dims[col] = len(l_enc.classes_)

# for col in train.columns[train.dtypes != object]:
#     train[col].fillna(train[col].mean(), inplace=True)

# COMMAND ----------

display(train)

# COMMAND ----------

unused_feat = ['Set', 'rank']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# define your embedding sizes : here just a random choice
cat_emb_dim = cat_dims

# COMMAND ----------

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices].reshape(-1, 1)

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices].reshape(-1, 1)

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices].reshape(-1, 1)

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



clf_xgb = XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=0.2992802183791389, colsample_bynode=0.682013536915446, colsample_bytree=0.5516566925070775, enable_categorical=False, gamma=6.5509851489035205, gpu_id=-1, importance_type=None, interaction_constraints='', learning_rate=0.26625867177253454, max_delta_step=8, max_depth=3, min_child_weight=0.41408362535166765, monotone_constraints='()', n_estimators=1000, n_jobs=8, num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=118, reg_lambda=1, scale_pos_weight=1, subsample=0.8548910717873056, tree_method='exact', validate_parameters=1, verbosity=None)

clf_xgb.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=40,
        verbose=10)


# COMMAND ----------

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


# COMMAND ----------

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
       'objective': hp.choice('objective', ['reg:squarederror', 'reg:squaredlogerror']), 
       'booster': hp.choice('booster', ['gbtree', 'gblinear','dart']), 
       'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),
       'subsample': hp.uniform('subsample', 0, 1), 
        'gamma': hp.uniform('gamma', 0,9),
        'reg_alpha' : hp.quniform('reg_alpha', 0,120,1),
        'reg_lambda' : hp.quniform('reg_lambda', 0,120, 1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.2,1),
        'colsample_bylevel' : hp.uniform('colsample_bylevel', 0.2,1), 
        'colsample_bynode' : hp.uniform('colsample_bynode', 0.2,1), 
        'min_child_weight' : hp.uniform('min_child_weight', 0, 1),
        'n_estimators': hp.quniform("n_estimators", 100, 1200, 100),
        'learning_rate': hp.uniform("learning_rate", 0.01, 1), 
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

# best_model_new.save_model('/mnt/capstone/model/aod_filtered_gfs_elev_wlabels_joined_prepostsplit.json')

# COMMAND ----------

# saved_model = XGBRegressor()
# saved_model.load_model('dbfs:/mnt/capstone/model/aod_filtered_gfs_elev_wlabels_joined_prepostsplit.json')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Submission

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Retrain Best Model on all Train Data

# COMMAND ----------

test_aod_gfs_joined_with_labels = spark.read.parquet("dbfs:/mnt/capstone/test/aod_gfs_elev_joined.parquet")
submission = test_aod_gfs_joined_with_labels.select('datetime', 'grid_id')
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*['aod_lon_lat_list', 'pm25_date_d', 'pm25_datetime_dt', 'aod_reading_end', 'rank'])

# COMMAND ----------

display(test_aod_gfs_joined_with_labels )

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_month', dayofmonth(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_week', dayofweek(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_year', dayofyear(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('week_of_year', weekofyear(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('month', month(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('year', year(test_aod_gfs_joined_with_labels.datetime))

# COMMAND ----------

test_full = test_aod_gfs_joined_with_labels.toPandas()
features_clean = [ col for col in test_full.columns ] 
test_clean = test_full[features_clean].values
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*['datetime'])

# COMMAND ----------

train = tabnet_df.toPandas()
test = test_aod_gfs_joined_with_labels.toPandas()
target = 'value'

# COMMAND ----------

display(train)

# COMMAND ----------

for col in train.columns: 
    print(col, ' ', train[col].dtype)

# COMMAND ----------

categorical_columns = []
categorical_dims =  {}
for col in train.columns[train.dtypes == object]:
    print(col, train[col].nunique())
    l_enc = LabelEncoder()
    train[col] = train[col].fillna("VV_likely")
    if col != 'Set' and col != 'pm25_reading_date' and col != 'datetime':
        test[col] = test[col].fillna("VV_likely")
    train[col] = l_enc.fit_transform(train[col].values)
    if col != 'Set' and col != 'pm25_reading_date' and col != 'datetime': 
        test[col] = l_enc.fit_transform(test[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

for col in train.columns[train.dtypes == 'float64']:
    train.fillna(train[col].mean(), inplace=True)
    if col != 'avg(latitude)' and col != 'avg(longitude)' and col != 'Set': 
        test.fillna(train[col].mean(), inplace = True)


# COMMAND ----------

unused_feat = ['Set', 'avg(latitude)', 'avg(longitude)', 'datetime']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# define your embedding sizes : here just a random choice
cat_emb_dim = cat_dims

# COMMAND ----------

X_train = train[features].values
y_train = train[target].values.reshape(-1, 1)

X_test = test[features].values
y_test = test[target].values.reshape(-1, 1)

# COMMAND ----------

best_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.4376675359128531, colsample_bynode=0.2841379441009236, colsample_bytree=0.3893070330625502, enable_categorical=False, gamma=5.2975038848774885, gpu_id=-1, importance_type=None, interaction_constraints='', learning_rate=0.08702869895192422, max_delta_step=0, max_depth=6, min_child_weight=0.7826055590298712, monotone_constraints='()', n_estimators=900, n_jobs=8, num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=25, reg_lambda=38, scale_pos_weight=1, subsample=0.41990551649414654, tree_method='exact', validate_parameters=1, verbosity=None)

# COMMAND ----------

# Fit tuned best model on full train data. 
best_model.fit(X_train, y_train)

# COMMAND ----------

df_test = pd.DataFrame(data = X_test)

# COMMAND ----------

display(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make predictions on test data. 

# COMMAND ----------

preds = np.array(best_model.predict(X_test))

# COMMAND ----------

preds

# COMMAND ----------

df = pd.DataFrame(data = test_clean, columns = features_clean)

# COMMAND ----------

df['value'] = preds
df

# COMMAND ----------

display(df)

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

submission_df_sp.coalesce(1).write.option("header",True).csv("/mnt/capstone/test/submission_03_18_aod_gfs_elev_joined_random_date.csv") 

# COMMAND ----------

