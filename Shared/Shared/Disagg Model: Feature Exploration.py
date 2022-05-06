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
from pyspark.sql.functions import *
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

#Take single closest observation. 

# aod_gfs_joined_with_labels_1 = aod_gfs_joined_with_labels.where('AOD_distance_rank = 1')
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.where('AOD_distance_rank <= 20')

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA', aod_gfs_joined_with_labels.AOD_QA.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_qa_str', aod_gfs_joined_with_labels.AOD_qa_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA_Cloud_Mask_str', aod_gfs_joined_with_labels.AOD_QA_Cloud_Mask_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA_LWS_Mask_str', aod_gfs_joined_with_labels.AOD_QA_LWS_Mask_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA_Adj_Mask_str', aod_gfs_joined_with_labels.AOD_QA_Adj_Mask_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_Level_str', aod_gfs_joined_with_labels.AOD_Level_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('BRF_over_snow_str', aod_gfs_joined_with_labels.BRF_over_snow_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('BRF_climatology_str', aod_gfs_joined_with_labels.BRF_climatology_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_QA_SC_Mask_str', aod_gfs_joined_with_labels.AOD_QA_SC_Mask_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('Algo_init_str', aod_gfs_joined_with_labels.Algo_init_str.cast('int'))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_distance_to_grid_center', aod_gfs_joined_with_labels.AOD_distance_to_grid_center.cast('float'))

# str_cols = [item[0] for item in aod_gfs_joined_with_labels.dtypes if item[1].startswith('string')]
# str_cols.append('value')

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.groupBy(*str_cols).mean()
# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.drop(*['avg(value)'])

# COMMAND ----------

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels[aod_gfs_joined_with_labels['distance_rank']==1]

# COMMAND ----------

display(aod_gfs_joined_with_labels)

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
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_month', dayofmonth(aod_gfs_joined_with_labels.datetime))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_week', dayofweek(aod_gfs_joined_with_labels.datetime))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_year', dayofyear(aod_gfs_joined_with_labels.datetime))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('week_of_year', weekofyear(aod_gfs_joined_with_labels.datetime))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('month', month(aod_gfs_joined_with_labels.datetime))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('year', year(aod_gfs_joined_with_labels.datetime))

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

cols_aod = [col for col in aod_gfs_joined_with_labels.columns if '_047' in col or '_055' in col]
cols_aod

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

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed00', 
                   (((aod_gfs_joined_with_labels['max(u_pbl00_new)']**2)+
                    (aod_gfs_joined_with_labels['max(v_pbl00_new)'])**2)**(1/2)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed06', 
                   (((aod_gfs_joined_with_labels['max(u_pbl06_new)']**2)+
                    (aod_gfs_joined_with_labels['max(v_pbl06_new)'])**2)**(1/2)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed12', 
                   (((aod_gfs_joined_with_labels['max(u_pbl12)']**2)+
                    (aod_gfs_joined_with_labels['max(v_pbl12)'])**2)**(1/2)))

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed18', 
                   (((aod_gfs_joined_with_labels['max(u_pbl18_new)']**2)+
                    (aod_gfs_joined_with_labels['max(v_pbl18_new)'])**2)**(1/2)))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed00', 
#                    (((aod_gfs_joined_with_labels['avg(max(u_pbl00_new))']**2)+
#                     (aod_gfs_joined_with_labels['avg(max(v_pbl00_new))'])**2)**(1/2)))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed06', 
#                    (((aod_gfs_joined_with_labels['avg(max(u_pbl06_new))']**2)+
#                     (aod_gfs_joined_with_labels['avg(max(v_pbl06_new))'])**2)**(1/2)))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed12', 
#                    (((aod_gfs_joined_with_labels['avg(max(u_pbl12))']**2)+
#                     (aod_gfs_joined_with_labels['avg(max(v_pbl12))'])**2)**(1/2)))

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('wind_speed18', 
#                    (((aod_gfs_joined_with_labels['avg(max(u_pbl18_new))']**2)+
#                     (aod_gfs_joined_with_labels['avg(max(v_pbl18_new))'])**2)**(1/2)))

# COMMAND ----------

import math

# Fill nan and -inf with Null
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(-math.inf, np.nan) 
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(math.inf, np.nan) 
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(-np.inf, np.nan) 
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(np.nan, np.nan)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(math.nan, np.nan)

# COMMAND ----------

# MAGIC %md
# MAGIC Trailing7d, Trailing15d, Trailing30d, Trailing90d averages by Year & Grid ID. 

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

days = lambda i: i * 86400 
if FEATURE_ENG_TRAILING: 
    # Trailing AOD 47
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing90d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(90), 0)))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing45d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(45), 0)))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing30d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(30), 0)))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing15d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(15), 0)))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing7d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(7), 0)))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing3d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(3), 0)))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing2d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(2), 0)))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('trailing1d_Optical_Depth_047', sf.mean('median_Optical_Depth_047').over(Window.partitionBy('grid_id').orderBy(col('datetime').cast("timestamp").cast("long")).rangeBetween(-days(1), 0)))

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

# MAGIC %md Train/Val Split

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

# tp_train_df = final_train_df.filter(final_train_df.location == "Taipei")
# la_train_df = final_train_df.filter(final_train_df.location == "Los Angeles (SoCAB)")
# dl_train_df = final_train_df.filter(final_train_df.location == "Delhi")

# COMMAND ----------

# MAGIC %md Helper functions. 

# COMMAND ----------

# MAGIC %md Modified code for pyspark.ml.tuning to get a time-series valid, cross-validation class implementation. 

# COMMAND ----------

# MAGIC %md Train / Val Split

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

import math

# COMMAND ----------

# Fill nan and -inf with Null
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(-math.inf, None) 
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(np.nan, None)
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.replace(math.nan, None)

# COMMAND ----------

split_strategy='random_day'

#Need to create a standard date column (not timestamp) so we can split the data on date. 
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('date', to_date('datetime'))
# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('AOD_distance_to_grid_center', aod_gfs_joined_with_labels.AOD_distance_to_grid_center.cast('int'))

# COMMAND ----------

if split_strategy == 'time':
    tabnet_df = aod_gfs_joined_with_labels.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('date')))

    #5-5-80-5-5 split
    tabnet_df = tabnet_df.withColumn("Set", when((tabnet_df.rank < .08), "test") \
                                                  .when((tabnet_df.rank < .16), "valid") \
                                                   .when((tabnet_df.rank < .90), "train") \
                                                   .when((tabnet_df.rank < .95), "valid") \
                                                   .otherwise("test")).cache()
    tabnet_df = tabnet_df.drop(*['aod_lon_lat_list', 'pm25_date_d', 'pm25_datetime_dt', 'rank', 'aod_reading_end',
                             'datetime','pm25_reading_date', 'tz'])
    train = tabnet_df.toPandas()
    target = 'value'
    train_indices = train[train.Set=="train"].index
    valid_indices = train[train.Set=="valid"].index
    test_indices = train[train.Set=="test"].index
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


# COMMAND ----------

if split_strategy == 'random_day':
    train_split, val_split, test_split = aod_gfs_joined_with_labels.select("date").distinct().randomSplit(weights=[0.8, 0.1, 0.1], seed = 0)
    
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
    
     #Create Second split for pseudo-cross validation
    train_split, val_split, test_split = aod_gfs_joined_with_labels.select("date").distinct().randomSplit(weights=[0.8, 0.1, 0.1], seed = 1)

    train_split = train_split.withColumn("Set", lit("train"))
    val_split = val_split.withColumn("Set", lit("valid"))
    test_split = test_split.withColumn("Set", lit("test"))

    sets = train_split.union(val_split)
    sets = sets.union(test_split)
    tabnet_df_2 = aod_gfs_joined_with_labels.join(sets, on = "date", how = "left")
    tabnet_df_2 = tabnet_df_2.drop(*['aod_lon_lat_list', 'pm25_date_d', 'pm25_datetime_dt', 'rank', 'aod_reading_end',
                             'datetime','pm25_reading_date', 'tz'])
    train_2 = tabnet_df_2.toPandas()

    train_indices_2 = train_2[train_2.Set=="train"].index
    valid_indices_2 = train_2[train_2.Set=="valid"].index
    test_indices_2 = train_2[train_2.Set=="test"].index

# COMMAND ----------

display(aod_gfs_joined_with_labels)

# COMMAND ----------

display(val_split)

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

display(train)

# COMMAND ----------

def imputeFeatures(train, train_indices): 
    target = 'value'
    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns[train.dtypes == object]:
    #     print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

    for col in train.columns[train.dtypes != object]:
        
        if col != target:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)
            
    return train, categorical_columns, categorical_dims

# COMMAND ----------

train, categorical_columns, categorical_dims = imputeFeatures(train, train_indices)
train_2, categorical_columns_2, categorical_dims_2 = imputeFeatures(train_2, train_indices_2)

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


X_train_2 = train_2[features].values[train_indices_2]
y_train_2 = train_2[target].values[train_indices_2].reshape(-1, 1)

X_valid_2 = train_2[features].values[valid_indices_2]
y_valid_2 = train_2[target].values[valid_indices_2].reshape(-1, 1)

X_test_2 = train_2[features].values[test_indices_2]
y_test_2 = train_2[target].values[test_indices_2].reshape(-1, 1)

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
# MAGIC ## XGBoost

# COMMAND ----------

from xgboost import XGBRegressor

# COMMAND ----------

clf_xgb = XGBRegressor(base_score=0.5, booster='dart',
             colsample_bylevel=0.6673878201117321,
             colsample_bynode=0.6551077584289375,
             colsample_bytree=0.6089558940954254, enable_categorical=False,
             gamma=67.87020347097445, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.5199240626776392,
             max_delta_step=7, max_depth=10,
             min_child_weight=0.37554830143796114,
             monotone_constraints='()', n_estimators=800, n_jobs=8,
             num_parallel_tree=1, objective='reg:linear', predictor='auto',
             random_state=0, reg_alpha=10, reg_lambda=198, scale_pos_weight=1,
             subsample=0.5937470788067594, tree_method='exact',
             validate_parameters=1, verbosity=None)

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


# COMMAND ----------

from sklearn import base

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
    return {'loss': -avg_expected, 'model1': clf, 'model2':clf_2, 'status': STATUS_OK }

# COMMAND ----------

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 40,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

best_model_new = trials.results[np.argmin([r['loss'] for r in  trials.results])]['model1']

best_model_new_2 = trials.results[np.argmin([r['loss'] for r in  trials.results])]['model2']

# COMMAND ----------

# best_model_new = trials.results[np.argmin([r['loss'] for r in  trials.results])]['model']
best_model_new

# COMMAND ----------

best_model_new_2

# COMMAND ----------

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

# COMMAND ----------

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

# MAGIC %md
# MAGIC ## Collapse model

# COMMAND ----------

train_df = pd.DataFrame(data = X_train, columns = features)
train_df['value'] = y_train
train_df['pred'] = train_preds

val_df = pd.DataFrame(data = X_valid, columns = features)
val_df['value'] = y_valid
val_df['pred'] = val_preds

test_df = pd.DataFrame(data = X_test, columns = features)
test_df['value'] = y_test
test_df['pred'] = test_preds

# COMMAND ----------

display(train_df)

# COMMAND ----------

train_c = spark.createDataFrame(train_df).select('date', 'grid_id', 'location', 'pred', 'value').groupBy('date', 'grid_id', 'location', 'value').agg(concat((collect_list('pred'))).alias("preds"))
val_c = spark.createDataFrame(val_df).select('date', 'grid_id', 'location', 'pred', 'value').groupBy('date', 'grid_id', 'location', 'value').agg(concat((collect_list('pred'))).alias("preds"))
test_c = spark.createDataFrame(test_df).select('date', 'grid_id', 'location', 'pred', 'value').groupBy('date', 'grid_id', 'location', 'value').agg(concat(collect_list('pred')).alias("preds"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Calculate the min, max, mean, median, and percentile based summary statistics to see how these impact R2 score. 

# COMMAND ----------

def calcSummaryR2(df): 
    r2 = r2_score(y_pred=df['preds'].values, y_true=df['value'].values)
    return r2

# COMMAND ----------

#Average
train_mean = spark.createDataFrame(train_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(mean('pred').alias("preds")).toPandas()
val_mean = spark.createDataFrame(val_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(mean('pred').alias("preds")).toPandas()
test_mean = spark.createDataFrame(test_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(mean('pred').alias("preds")).toPandas()

train_r2 = calcSummaryR2(train_mean)
val_r2 = calcSummaryR2(val_mean)
test_r2 = calcSummaryR2(test_mean)


print(train_r2)
print(val_r2)
print(test_r2)

# COMMAND ----------

#Min
train_min = spark.createDataFrame(train_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(min('pred').alias("preds")).toPandas()
val_min = spark.createDataFrame(val_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(min('pred').alias("preds")).toPandas()
test_min = spark.createDataFrame(test_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(min('pred').alias("preds")).toPandas()

train_r2 = calcSummaryR2(train_min)
val_r2 = calcSummaryR2(val_min)
test_r2 = calcSummaryR2(test_min)


print(train_r2)
print(val_r2)
print(test_r2)

# COMMAND ----------

#Max
train_max = spark.createDataFrame(train_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(max('pred').alias("preds")).toPandas()
val_max = spark.createDataFrame(val_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(max('pred').alias("preds")).toPandas()
test_max = spark.createDataFrame(test_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(max('pred').alias("preds")).toPandas()

train_r2 = calcSummaryR2(train_max)
val_r2 = calcSummaryR2(val_max)
test_r2 = calcSummaryR2(test_max)


print(train_r2)
print(val_r2)
print(test_r2)

# COMMAND ----------

#Median
train_med = spark.createDataFrame(train_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(percentile_approx('pred', 0.5).alias("preds")).toPandas()
val_med = spark.createDataFrame(val_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(percentile_approx('pred', 0.5).alias("preds")).toPandas()
test_med = spark.createDataFrame(test_df).select('date', 'grid_id', 'pred', 'value').groupBy('date', 'grid_id', 'value').agg(percentile_approx('pred', 0.5).alias("preds")).toPandas()

train_r2 = calcSummaryR2(train_med)
val_r2 = calcSummaryR2(val_med)
test_r2 = calcSummaryR2(test_med)


print(train_r2)
print(val_r2)
print(test_r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expand Predictions for Modeling

# COMMAND ----------

def expandPreds(df): 
    for i in range(1, 21): 
        df = df.withColumn((f'pred{i}'), element_at(df.preds, i))
    df = df.drop(*['preds'])
    return df

# COMMAND ----------

train_c = expandPreds(train_c)
train_c = train_c.fillna(0)
train_c = train_c.toPandas()
val_c = expandPreds(val_c)
val_c = val_c.fillna(0)
val_c = val_c.toPandas()
test_c = expandPreds(test_c)
test_c = test_c.fillna(0)
test_c = test_c.toPandas()

# COMMAND ----------

test_c

# COMMAND ----------

target = 'value'
features = [ col for col in train_c.columns if col not in [target]] 

# COMMAND ----------

X_train_c = train_c[features].values
y_train_c = train_c[target].values.reshape(-1, 1)

X_valid_c = val_c[features].values
y_valid_c = val_c[target].values.reshape(-1, 1)

X_test_c = test_c[features].values
y_test_c = test_c[target].values.reshape(-1, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

clf = LinearRegression(fit_intercept = True, normalize = True)

clf.fit(X_train_c, y_train_c)
#         eval_set=[(X_valid, y_valid)],
#         early_stopping_rounds=40,
#         verbose=10)

# COMMAND ----------

train_preds_c = np.array(clf.predict(X_train_c))
train_r2_c = r2_score(y_pred=train_preds_c, y_true=y_train_c)
print(train_r2_c)

val_preds_c = np.array(clf.predict(X_valid_c))
valid_r2_c = r2_score(y_pred=val_preds_c, y_true=y_valid_c)
print(valid_r2_c)

test_preds_c = np.array(clf.predict(X_test_c))
test_r2_c = r2_score(y_pred=test_preds_c, y_true=y_test_c)
print(test_r2)

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

# COMMAND ----------

#Rename GFS variables to match train 
def rename_GFS(df, rename_list):
    for time in ['12']:
        for name in rename_list:
            old_name = name.replace('XX', time)
            new_name = old_name.replace('_new)', ')')
            df = df.withColumn(new_name, df[old_name]).drop(*[old_name])
    return df

# COMMAND ----------

rename_list = ['max(t_surfaceXX_new)',
               'max(pbl_surfaceXX_new)', 
              'max(hindex_surfaceXX_new)', 
               'max(gust_surfaceXX_new)', 
               'max(r_atmosphereXX_new)', 
               'max(pwat_atmosphereXX_new)', 
               'max(u_pblXX_new)', 
               'max(v_pblXX_new)', 
               'max(vrate_pblXX_new)']
          
test_aod_gfs_joined_with_labels = rename_GFS(test_aod_gfs_joined_with_labels, rename_list)

# COMMAND ----------

display(test_aod_gfs_joined_with_labels.select('grid_id', 'datetime').distinct())

# COMMAND ----------

display(test_aod_gfs_joined_with_labels.dtypes)

# COMMAND ----------

#Take single closest observation. 

# test_aod_gfs_joined_with_labels_1 = test_aod_gfs_joined_with_labels.where('AOD_distance_rank = 1')
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
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('date', to_date('datetime'))

# COMMAND ----------

#Rename GFS variables to match train 
def drop_GFS(df, rename_list):
    for time in ['12']:
        for name in rename_list:
            old_name = name.replace('XX', time)
            df = df.drop(*[old_name])
    return df

# COMMAND ----------

display(test_aod_gfs_joined_with_labels)

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

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_month', dayofmonth(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_week', dayofweek(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('day_of_year', dayofyear(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('week_of_year', weekofyear(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('month', month(test_aod_gfs_joined_with_labels.datetime))
test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.withColumn('year', year(test_aod_gfs_joined_with_labels.datetime))

# COMMAND ----------

cols_aod = [col for col in test_aod_gfs_joined_with_labels.columns if '_047' in col or '_055' in col]
cols_aod

# COMMAND ----------

def aod_scale(x):
    if x:
        return (46.759*x)+7.1333
    else: 
        return None
aod_scale_udf = sf.udf(lambda x:aod_scale(x) ,DoubleType())

# COMMAND ----------

for col_aod in cols_aod:
    test_aod_gfs_joined_with_labels= test_aod_gfs_joined_with_labels.withColumn(col_aod+'_scaled',aod_scale_udf(test_aod_gfs_joined_with_labels[col_aod]))

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*cols_aod)

# COMMAND ----------

display(aod_gfs_joined_with_labels)

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.replace(-math.inf, None) 

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.replace(np.nan, None)

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.replace(math.nan, None)

# COMMAND ----------

test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop('value')

# COMMAND ----------

test_full = test_aod_gfs_joined_with_labels.toPandas()
features_clean = [ col for col in test_full.columns ] 
test_clean = test_full[features_clean].values
#test_aod_gfs_joined_with_labels = test_aod_gfs_joined_with_labels.drop(*['datetime'])

# COMMAND ----------

train = tabnet_df.drop(*['wind_speed12', 'wind_speed18', 'wind_speed00', 'wind_speed06']).toPandas()
test = test_aod_gfs_joined_with_labels.toPandas()
target = 'value'

# COMMAND ----------

test

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

X_train, X_valid, X_test = pca_features(X_train, None, X_test)

# COMMAND ----------

best_model_new = XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=0.2484906463769062, colsample_bynode=0.9465727754932802, colsample_bytree=0.9249323530026272, enable_categorical=False, gamma=96.89886840840305, gpu_id=-1, importance_type=None, interaction_constraints='', learning_rate=0.5180448254093463, max_delta_step=10, max_depth=8, min_child_weight=0.07919184774601618, monotone_constraints='()', n_estimators=1200, n_jobs=8, num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=22, reg_lambda=167, scale_pos_weight=1, subsample=0.7134807755048374, tree_method='exact', validate_parameters=1, verbosity=None)

# COMMAND ----------

# Fit tuned best model on full train data. 
best_model_new.fit(X_train, y_train,
            verbose=10)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make predictions on test data. 

# COMMAND ----------

X_test.shape

# COMMAND ----------

preds = np.array(best_model_new.predict(X_test))

# COMMAND ----------

preds

# COMMAND ----------

df = pd.DataFrame(data = test_clean, columns = features_clean)

# COMMAND ----------

df['value'] = preds

# COMMAND ----------

submission_df = df[['datetime', 'grid_id', 'value']]
submission_df = submission_df.sort_values(by = ['datetime', 'grid_id'])

# COMMAND ----------

display(submission_df_sp.select('value'))

# COMMAND ----------

display(aod_gfs_joined_with_labels.select('value'))

# COMMAND ----------

submission_df_sp = spark.createDataFrame(submission_df)

# COMMAND ----------

display(submission_df_sp)

# COMMAND ----------

submission_df_sp.coalesce(1).write.option("header",True).csv("/mnt/capstone/test/submission_03_20_distance20_aod_filtered_gfs_elev_joined_randomday.csv") 

# COMMAND ----------

