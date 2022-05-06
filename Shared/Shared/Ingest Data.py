# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Data Notebook

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
from pyspark.ml import Pipeline
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sparkdl.xgboost import XgboostRegressor
from pyspark.ml.feature import PCA


warnings.simplefilter('ignore')

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Install AWS CLI and Pull Data from Public s3 Bucket

# COMMAND ----------

gcs_bucket_name = "capstone-gcs"
gcs_mount_name = "capstone-gcs"
dbutils.fs.mount("gs://%s" % gcs_bucket_name, "/mnt/%s" % gcs_mount_name)

# COMMAND ----------

display(dbutils.fs.ls("/mnt/capstone/model"))

# COMMAND ----------

# access_key = "AKIA33CENIT6JFRXRNOE"
# secret_key = dbutils.secrets.get(scope = "capstone-s3", key = access_key)
# encoded_secret_key = secret_key.replace("/", "%2F")
# aws_bucket_name = "particulate-articulate-capstone"
# mount_name = "capstone"

# dbutils.fs.mount("s3a://%s:%s@%s" % (access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)
display(dbutils.fs.ls("/mnt/capstone/train"))

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined = spark.read.parquet("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined.parquet")
aod_gfs_joined_with_labels = spark.read.parquet("/mnt/capstone/train/aod_gfs_joined_with_labels.parquet")

# COMMAND ----------

!curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
!unzip awscliv2.zip
!sudo ./aws/install
!aws --version

# COMMAND ----------

#Download GFS data from s3 bucket to Databricks workspace. We will then load these files into a Spark DF. 
!aws s3 cp s3://capstone-particulate-storage/ train/GFS/ --no-sign-request --recursive

# COMMAND ----------

#Download AOD data from s3 bucket to Databricsk workspace. 
!aws s3 cp s3://particulate-articulate-capstone/train/aod/ train/AOD/ --no-sign-request --recursive

# COMMAND ----------

# MAGIC %md Now open downloaded GFS files and union into a full Spark Dataframe. 

# COMMAND ----------

def directory_to_sparkDF(directory, schema=None): 
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
            first = False
        else: 
            # If files have mismatched column counts (a handful of files may be missing certain forecast times), 
            # continue without union (i.e. drop "corrupt" file from training). 
            try: 
                df_new = spark.createDataFrame(df, schema=schema)
                df_out = df_new.union(df_out)
            except: 
                continue

    # Output shape so that we can quickly check that function works as intended. 
#     print(f'Rows: {df_out.count()}, Columns: {len(df_out.columns)}')
    return df_out

# COMMAND ----------

# create GFS Spark DataFrame
df_GFS = directory_to_sparkDF(directory = 'train/GFS/')

# COMMAND ----------

df_GFS_agg = df_GFS.groupBy("grid_id","date").mean()

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
        StructField("utc_date", StringType(), True)])

# COMMAND ----------

df_AOD = directory_to_sparkDF(directory = 'train/AOD', schema=AODCustomSchema)

# COMMAND ----------

# # Examine GFS DataFrame
# df_GFS.toPandas()

# COMMAND ----------

# # Examine AOD DataFrame
# df_AOD.toPandas()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

# Check schema
df_GFS.schema

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

# col_names, num_vars, str_vars = get_cols(df_GFS, [], None)
# summary = df_GFS.select(col_names).describe().toPandas()
# summary

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

# MAGIC %md 
# MAGIC 
# MAGIC In correlation matrix, notice that: 
# MAGIC 
# MAGIC - 1.) Correlations between previous forecast and future forecast (i.e. 00 and 06) show decreasing strength of correlations than correlations that exist within same time window. 
# MAGIC - 2.) Strongest correlations within a time-window appear to be pwat (precipitable water) & r (relative humidity): 0.96	AND vrate (ventilation rate) & gust (max wind gust): 0.91. 
# MAGIC - 3.) Weakest correlations within a time-window appear to be r (relative humidity) & pbl (planetary boundary layer height): 0.08, v component of wind & r (relative humidity): -0.05, and pbl (planetary boundary layer height) & pwat (precipitable water): -0.09. 

# COMMAND ----------

# # Get Correlation Matrix. 
# corr_mat_df = compute_correlation_matrix(df_GFS.select(num_vars), method = 'pearson')
# corr_mat_df

# COMMAND ----------

# #Plot Correlation Matrix. 
# sns.set(rc = {'figure.figsize':(20,20)})
# plot_corr_matrix(corr_mat_df, (num_vars[:9]), 234)

# COMMAND ----------

# #Create Tableau like dashboard to quickly explore all variables
# display(df_GFS.select(col_names))

# COMMAND ----------

# MAGIC %md Note that Haines Index (hindex) variable is only feature with nulls in our GFS data. We want to preserve this column given it's been shown to be correlated with fire growth. To ensure we can use we'll replace NaNs with -1. For more info on Haines Index (https://www.wfas.net/index.php/haines-index-fire-potential--danger-34). 

# COMMAND ----------

# df_AOD.printSchema()

# COMMAND ----------

# MAGIC %md Group GFS Data by date, grid_id

# COMMAND ----------

# df_GFS_agg = df_GFS.groupBy("grid_id","date").mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ## AOD QA

# COMMAND ----------

def qa_format(val):
    if val:
        return '{0:016b}'.format(val)

# COMMAND ----------

udf_qa_format = sf.udf(lambda x:qa_format(x),StringType() )

# COMMAND ----------

df_AOD=df_AOD.withColumn("AOD_qa_str",udf_qa_format(col("AOD_QA")))

# COMMAND ----------

# df_AOD.select("AOD_qa_str").distinct().show()

# COMMAND ----------

df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask_str', substring('AOD_qa_str', 0,3))\
    .withColumn('AOD_QA_LWS_Mask_str', substring('AOD_qa_str', 3,2))\
    .withColumn('AOD_QA_Adj_Mask_str', substring('AOD_qa_str', 5,3))\
    .withColumn('AOD_Level_str', substring('AOD_qa_str', 8,1))\
    .withColumn('Algo_init_str', substring('AOD_qa_str', 9,1))\
    .withColumn('BRF_over_snow_str', substring('AOD_qa_str', 10,1))\
    .withColumn('BRF_climatology_str', substring('AOD_qa_str', 11,1))\
    .withColumn('AOD_QA_SC_Mask_str', substring('AOD_qa_str', 12,3))

# COMMAND ----------

def masks_to_int(s):
    if s:
        return int(s, 2)

# COMMAND ----------

udf_mask_int = sf.udf(lambda x:masks_to_int(x),StringType() )

# COMMAND ----------

df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('AOD_QA_LWS_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('AOD_QA_Adj_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('AOD_Level', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('Algo_init', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('BRF_over_snow', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('BRF_climatology', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('AOD_QA_SC_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))

# COMMAND ----------

qa_masks_str_cols = ('AOD_QA_Cloud_Mask_str',
                    'AOD_QA_LWS_Mask_str',
                    'AOD_QA_Adj_Mask_str',
                    'AOD_Level_str',
                    'Algo_init_str',
                    'BRF_over_snow_str',
                    'BRF_climatology_str',
                    'AOD_QA_SC_Mask_str')
df_AOD.drop(*qa_masks_str_cols)

# COMMAND ----------

df_AOD.registerTempTable("aod")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AOD Lat-Lon pairs as list

# COMMAND ----------

df_AOD = df_AOD.withColumn("lon-lat-pair", sf.concat_ws('_',df_AOD.lon,df_AOD.lat))

# COMMAND ----------

lat_lon_list_df = df_AOD.groupBy("grid_id","utc_date")\
.agg(sf.collect_list("lon-lat-pair").alias("aod_lon_lat_list"))

# COMMAND ----------

# MAGIC %md
# MAGIC # AOD Grid Level

# COMMAND ----------

df_AOD = df_AOD.withColumn('AOD_QA_Cloud_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('AOD_QA_LWS_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('AOD_QA_Adj_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('AOD_Level', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('Algo_init', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('BRF_over_snow', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('BRF_climatology', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))\
    .withColumn('AOD_QA_SC_Mask', udf_mask_int(col("AOD_QA_Cloud_Mask_str")))

# COMMAND ----------

df_aod_grid = spark.sql("SELECT grid_id, utc_date,\
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
            FROM aod group by grid_id, utc_date")

# COMMAND ----------

test_aod_grid = df_aod_grid.groupBy('grid_id','utc_date').count()

# COMMAND ----------

test_aod_grid.where(test_aod_grid['count']>1).show()

# COMMAND ----------

df_aod_grid = df_aod_grid.join(lat_lon_list_df, on=[df_aod_grid.grid_id == lat_lon_list_df.grid_id,  
                                                   df_aod_grid.utc_date == lat_lon_list_df.utc_date],
                                                how="left").drop(lat_lon_list_df.grid_id).drop(lat_lon_list_df.utc_date)

# COMMAND ----------

# MAGIC %md
# MAGIC # Labels

# COMMAND ----------

file='meta_data/train_labels_grid.csv'
bucket='capstone-particulate-storage'

#buffer = io.BytesIO()
s3_read_client = boto3.client('s3')
s3_tl_obj = s3_read_client.get_object(Bucket= bucket, Key= file)
#s3_tl_obj.download_fileobj(buffer)
train_labels = pd.read_csv(s3_tl_obj['Body'],delimiter='|',header=0)

# COMMAND ----------

train_labels_df = spark.createDataFrame(train_labels)

# COMMAND ----------

# train_labels_df.head(4)

# COMMAND ----------

train_labels_df = train_labels_df.withColumn("date", date_format(train_labels_df['datetime'],"yyyy-MM-dd"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Joins

# COMMAND ----------

aod_gfs_joined = df_aod_grid.join(df_GFS_agg, on=[df_aod_grid.grid_id == df_GFS_agg.grid_id,  
                                              df_aod_grid.utc_date == df_GFS_agg.date],how="inner").drop(df_GFS_agg.grid_id).drop(df_GFS_agg.date)

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined.join(train_labels_df, on=[aod_gfs_joined.grid_id == train_labels_df.grid_id,  
                                              aod_gfs_joined.utc_date == train_labels_df.date],how="inner").drop(aod_gfs_joined.grid_id).drop(aod_gfs_joined.utc_date)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Date Components

# COMMAND ----------

from pyspark.sql.functions import dayofyear
from pyspark.sql.functions import dayofmonth
from pyspark.sql.functions import dayofweek
from pyspark.sql.functions import weekofyear
from pyspark.sql.functions import month
from pyspark.sql.functions import year
#TODO: Import holidays package (https://towardsdatascience.com/5-minute-guide-to-detecting-holidays-in-python-c270f8479387) package to get country's holidays. 

# COMMAND ----------

if FEATURE_ENG_TIME:   
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_month', dayofmonth(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_week', dayofweek(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_year', dayofyear(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('week_of_year', weekofyear(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('month', month(aod_gfs_joined_with_labels.pm25_reading_date))
    aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('year', year(aod_gfs_joined_with_labels.pm25_reading_date))

# COMMAND ----------

aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_month', dayofmonth(aod_gfs_joined_with_labels.date))
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_week', dayofweek(aod_gfs_joined_with_labels.date))
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('day_of_year', dayofyear(aod_gfs_joined_with_labels.date))
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('week_of_year', weekofyear(aod_gfs_joined_with_labels.date))
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('month', month(aod_gfs_joined_with_labels.date))
aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.withColumn('year', year(aod_gfs_joined_with_labels.date))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Daily, Weekly, Monthly, Yearly Averages by Grid ID. Trailing7d, Trailing15d, Trailing30d, Trailing90d averages by Year & Grid ID. 

# COMMAND ----------

df_day = aod_filtered_gfs_elev_wlabels_joined.groupBy("grid_id","day_of_week").mean()
df_weekly = aod_filtered_gfs_elev_wlabels_joined.groupBy("grid_id","week_of_year").mean()
df_monthly = aod_filtered_gfs_elev_wlabels_joined.groupBy("grid_id","month").mean()
df_yearly = aod_filtered_gfs_elev_wlabels_joined.groupBy("grid_id","year").mean()

# COMMAND ----------

df_day = aod_gfs_joined_with_labels.groupBy("grid_id","day_of_week").mean()
df_weekly = aod_gfs_joined_with_labels.groupBy("grid_id","week_of_year").mean()
df_monthly = aod_gfs_joined_with_labels.groupBy("grid_id","month").mean()
df_yearly = aod_gfs_joined_with_labels.groupBy("grid_id","year").mean()

# COMMAND ----------

# aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.join(df_day, ['grid_id', 'day_of_week'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Particulate Transport Path Joins

# COMMAND ----------

# MAGIC %md
# MAGIC # Imputation

# COMMAND ----------

# #df_aod_grid.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_aod_grid.columns[:-1]]).toPandas()
# df_aod_grid.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in df_aod_grid.dtypes if c not in ('aod_lon_lat_list')]).toPandas()
# #columns[:-1] to avoid checking latlon pairs list

# COMMAND ----------

# df_aod_grid.count()

# COMMAND ----------

def impute(to_impute, additional_drop_cols, df, RUN_CROSSVAL):
    
    imp_test = df.filter(df[to_impute].isNull())
    imp_train = df.filter(df[to_impute].isNotNull())
    print("train size:", imp_train.count())
    print("test size:", imp_test.count())
    
    #Remove troublesome features
    imp_drop_cols = ['aod_lon_lat_list', 'datetime','pm25_date_d','pm25_datetime_dt','value']+additional_drop_cols
    #Remove our train label (to impute). 
    imp_label = to_impute
    imp_features, imp_int_features, imp_str_features = get_cols(imp_train, imp_drop_cols, imp_label)
    imp_train=imp_train.drop(*imp_drop_cols)
    
    #convert to pandas for mean imputation for predictors for imputation columns
    imp_train_pandas = imp_train.toPandas()
    
    #get all Xs with nulls
    cols_with_nulls = imp_train_pandas.columns[imp_train_pandas.isna().any()].tolist()
    
    for var in cols_with_nulls:
        imp_train_pandas[var].fillna(imp_train_pandas[var].mean(), inplace=True)
    
    # Add percent rank to aid in cross validation/splitting
    imp_train = imp_train.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('pm25_reading_date')))

    if RUN_CROSSVAL == False:  
        # For non-Cross Validation (1 fold): 
        imp_train = imp_train.withColumn("foldNumber", when((imp_train.rank < .8), lit(0)) \
                                                   .otherwise(lit(1))).cache()
    else: 
        #For 6 Folds: 
        imp_train = imp_train.withColumn("foldNumber", when((imp_train.rank < .12), lit(0)) \
                                                     .when((imp_train.rank < .17), lit(1)) \
                                                     .when((imp_train.rank < .28), lit(2)) \
                                                     .when((imp_train.rank < .33), lit(3)) \
                                                   .when((imp_train.rank < .45), lit(4)) \
                                                   .when((imp_train.rank < .50), lit(5)) \
                                                   .when((imp_train.rank < .62), lit(6)) \
                                                   .when((imp_train.rank < .67), lit(7)) \
                                                   .when((imp_train.rank < .78), lit(8)) \
                                                   .when((imp_train.rank < .83), lit(9)) \
                                                   .when((imp_train.rank < .95), lit(10)) \
                                                   .otherwise(lit(11))).cache()

    imp_train_train = imp_train.where("foldNumber % 2 == 0")
    imp_train_val = imp_train.where("foldNumber % 2 != 0")
    #create imputation input pipeline
    imp_pipelineModel, imp_train_df, imp_validation_df, va = create_imputation_input_pipeline(imp_train_train, imp_train_val,
                                                                                         imp_features, imp_int_features, imp_str_features)
    
    imp_crossVal_df = imp_validation_df.union(imp_train_df)    

    if RUN_CROSSVAL:    
        
        modelComparisons = []

        #Define model objects. 
        gbt = gbt_for_imputation(imp_train_df, imp_validation_df, imp_label)
        xgb = xgboost_for_imputation(imp_train_df, imp_validation_df, imp_label)

        # Manually define non-random search space. 
        # This ensures data and params are controlled for in our experimental comparisons, yielding valid results. 
        MIN_INFO_GAIN_SEARCH_LIST = [0.0]
        MAX_DEPTH_SEARCH_LIST = [2, 3]

        # Define ParamGrids

        gbt_paramGrid = ParamGridBuilder() \
            .addGrid(gbt.minInfoGain, MIN_INFO_GAIN_SEARCH_LIST) \
            .addGrid(gbt.maxDepth, MAX_DEPTH_SEARCH_LIST) \
            .build()


        xgb_paramGrid = ParamGridBuilder() \
            .addGrid(xgb.max_depth, MAX_DEPTH_SEARCH_LIST) \
            .build()



        modelList = [(gbt, "Gradient Boosted Tree", gbt_paramGrid),
                 (xgb, "XGBoost", xgb_paramGrid)]

        #Create an empty list of lists that we will append models & performance metrics to.
        # Data order will be: model_name[str], model[obj], features[list], f_beta_score[float], f_beta_std_dev[float], paramGrid [obj] 
        modelComparisons = []

        #Build comparison table. 
        for model, model_name, paramGrid in modelList:
            modelComparisons.append(compareBaselines(model = model, model_name = model_name, paramGrid = paramGrid, 
                                                     cv_train_data = imp_crossVal_df, label = imp_label))

        #model_name[str], model[obj], features[list], precision[float]
        modelComparisonsDF = pd.DataFrame(modelComparisons, columns = ['model_name', 'model_obj','feature_names','r2_score', 
                                                                       'r2_std_dev', 'paramGrid_obj', 'bestParams'])\
                               .sort_values(by = 'r2_score').reset_index(drop=True)

        imp_model = modelComparisonsDF['model_obj'].iloc[0]
    
    else:
        # get xgboost model
        # imp_xgb = xgboost_for_imputation(imp_train_df, imp_validation_df, imp_label)
        imp_model= gbt_for_imputation(imp_train_df, imp_validation_df, imp_label)
    
    imp_test_df = imp_pipelineModel.transform(imp_test)
    predictions = imp_model.transform(imp_test_df).select("grid_id","pm25_reading_date","prediction")\
                                                .withColumnRenamed("prediction", to_impute+"_imputed")
    
    #featureImportance = pd.DataFrame(list(zip(va.getInputCols(), imp_model.featureImportances)), columns = ['Feature', 'Importance']).sort_values(by = 'Importance', ascending = False).head(10)
    #print([imp_train_df.columns[i] for i in important_features[:5]])
    #print(featureImportance)
    return predictions#,featureImportance)

# COMMAND ----------

def create_imputation_input_pipeline(imp_train_train, imp_train_val, imp_features, imp_int_features, imp_str_features):
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
    return (imp_pipelineModel, imp_train_df, imp_validation_df, va)

# COMMAND ----------

def xgboost_for_imputation(imp_train_df, imp_validation_df, imp_label):
    # The next step is to define the model training stage of the pipeline. 
    # The following command defines a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.
    # If you are running Databricks Runtime for Machine Learning 9.0 ML or above, you can set the `num_workers` parameter to leverage the cluster for distributed training.
    imp_xgb = XgboostRegressor(num_workers=3, featuresCol = "features", labelCol=imp_label)
    #Fit on train and predict (transform) on test. 
    imp_xgb = imp_xgb.fit(imp_train_df)
    imputations = imp_xgb.transform(imp_validation_df)

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
    
    return imp_xgb

# COMMAND ----------

def gbt_for_imputation(imp_train_df, imp_validation_df, imp_label):
    imp_gbt = GBTRegressor(featuresCol = "features", labelCol= imp_label, maxIter = 10)

    #Fit on train and predict (transform) on test. 
    imp_gbt = imp_gbt.fit(imp_train_df)
    imputations = imp_gbt.transform(imp_validation_df)

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
    return imp_gbt

# COMMAND ----------

#df_aod_grid_pd = df_aod_grid.toPandas()
cols_to_impute = ['min_Optical_Depth_047','max_Optical_Depth_047','median_Optical_Depth_047',
                 'min_Optical_Depth_055','max_Optical_Depth_055','median_Optical_Depth_055']
additional_drop_dict = {
    'min_Optical_Depth_047':['max_Optical_Depth_047','median_Optical_Depth_047'],
    'max_Optical_Depth_047':['min_Optical_Depth_047','median_Optical_Depth_047'],
    'median_Optical_Depth_047':['max_Optical_Depth_047','min_Optical_Depth_047'],
    'min_Optical_Depth_055':['max_Optical_Depth_055','median_Optical_Depth_055'],
    'max_Optical_Depth_055':['min_Optical_Depth_055','median_Optical_Depth_055'],
    'median_Optical_Depth_055':['max_Optical_Depth_055','min_Optical_Depth_055'],
    'min_AOD_Uncertainty':['max_AOD_Uncertainty','median_AOD_Uncertainty'],
    'max_AOD_Uncertainty':['min_AOD_Uncertainty','median_AOD_Uncertainty'],
    'median_AOD_Uncertainty':['max_AOD_Uncertainty','min_AOD_Uncertainty'],
    'min_Column_WV':['max_Column_WV','median_Column_WV'],
    'max_Column_WV':['min_Column_WV','median_Column_WV'],
    'median_Column_WV':['max_Column_WV','min_Column_WV']   
}
#predictions_combined = aod_gfs_joined_with_labels.select(["grid_id","date"])
# for to_impute in cols_to_impute[0:3]:
#     df_orig = aod_gfs_joined_with_labels
    
#     additional_drop_cols = additional_drop_dict[to_impute]
#     predictions = impute(to_impute, additional_drop_cols, df_orig)

#     aod_gfs_joined_with_labels = aod_gfs_joined_with_labels.join(predictions, on=["grid_id","date"], how="left")
#     aod_gfs_joined_with_labels.cache()

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined = spark.read.parquet("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined.parquet")

# COMMAND ----------

test_aod_gfs_elev_joined = spark.read.parquet("/mnt/capstone/test/aod_filtered_gfs_elev_joined.parquet")

# COMMAND ----------

for to_impute in cols_to_impute:
    df_orig = aod_filtered_gfs_elev_wlabels_joined
    
    additional_drop_cols = additional_drop_dict[to_impute]
    predictions = impute(to_impute, additional_drop_cols, df_orig, True)

    aod_filtered_gfs_elev_wlabels_joined = aod_filtered_gfs_elev_wlabels_joined.join(predictions, on=["grid_id","pm25_reading_date"], how="left")
    aod_filtered_gfs_elev_wlabels_joined = aod_filtered_gfs_elev_wlabels_joined.withColumn("final_"+to_impute,
                                                    sf.coalesce(aod_filtered_gfs_elev_wlabels_joined[to_impute], 
             aod_filtered_gfs_elev_wlabels_joined[to_impute+"_imputed"]))\
                                                        .drop(to_impute).drop(to_impute+"_imputed")\
                                                        .withColumnRenamed("final_"+to_impute, to_impute)
    print(aod_filtered_gfs_elev_wlabels_joined.columns)
    aod_filtered_gfs_elev_wlabels_joined.cache()

# COMMAND ----------

test_aod_gfs_elev_joined.write.mode('overwrite').parquet("/mnt/capstone/test/test_aod_gfs_elev_joined_wimputations.parquet")

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined.write.mode('overwrite').parquet("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_wimputations.parquet")

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined_wimputations = spark.read.parquet("/aod_filtered_gfs_elev_wlabels_joined_wimputations.parquet")

# COMMAND ----------

cols_without_value = aod_filtered_gfs_elev_wlabels_joined_wimputations.columns
cols_without_value.pop(cols_without_value.index("value"))

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined_wimputations\
.write.mode("overwrite").parquet("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_wimputations.parquet")

# COMMAND ----------

trial_df = spark.read.parquet("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_wimputations.parquet")

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined_wimputations\
.select(['value']+cols_without_value)\
.coalesce(1)\
.write.mode("overwrite").csv("/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_wimputations.csv")

# COMMAND ----------

aod_filtered_gfs_elev_wlabels_joined_wimputations.toPandas()\
.to_csv("aod_filtered_gfs_elev_wlabels_joined_wimputations_cons.csv")

# COMMAND ----------

dbutils.fs.cp("aod_filtered_gfs_elev_wlabels_joined_wimputations_cons.csv", "/mnt/capstone/train/aod_filtered_gfs_elev_wlabels_joined_wimputations_cons.csv")

# COMMAND ----------

aod_gfs_labels_w_imputation = aod_gfs_joined_with_labels.withColumn("final_min_Optical_Depth_047",
    sf.coalesce(aod_gfs_joined_with_labels["min_Optical_Depth_047"], 
             aod_gfs_joined_with_labels["min_Optical_Depth_047_imputed"]))\
    .withColumn("final_max_Optical_Depth_047",
        sf.coalesce(aod_gfs_joined_with_labels["max_Optical_Depth_047"], 
                 aod_gfs_joined_with_labels["max_Optical_Depth_047_imputed"]))\
    .withColumn("final_median_Optical_Depth_047",
        sf.coalesce(aod_gfs_joined_with_labels["median_Optical_Depth_047"], 
                 aod_gfs_joined_with_labels["median_Optical_Depth_047_imputed"]))\
    .withColumn("final_min_Optical_Depth_055",
        sf.coalesce(aod_gfs_joined_with_labels["min_Optical_Depth_055"], 
                 aod_gfs_joined_with_labels["min_Optical_Depth_055_imputed"]))\
    .withColumn("final_max_Optical_Depth_055",
        sf.coalesce(aod_gfs_joined_with_labels["max_Optical_Depth_055"], 
                 aod_gfs_joined_with_labels["max_Optical_Depth_055_imputed"]))\
    .withColumn("final_median_Optical_Depth_055",
        sf.coalesce(aod_gfs_joined_with_labels["median_Optical_Depth_055"], 
                 aod_gfs_joined_with_labels["median_Optical_Depth_055_imputed"]))\
    .withColumn("final_min_AOD_Uncertainty",
        sf.coalesce(aod_gfs_joined_with_labels["min_AOD_Uncertainty"], 
                 aod_gfs_joined_with_labels["min_AOD_Uncertainty_imputed"]))\
    .withColumn("final_max_AOD_Uncertainty",
        sf.coalesce(aod_gfs_joined_with_labels["max_AOD_Uncertainty"], 
                 aod_gfs_joined_with_labels["max_AOD_Uncertainty_imputed"]))\
    .withColumn("final_median_AOD_Uncertainty",
        sf.coalesce(aod_gfs_joined_with_labels["median_AOD_Uncertainty"], 
                 aod_gfs_joined_with_labels["median_AOD_Uncertainty_imputed"]))\
    .withColumn("final_min_Column_WV",
        sf.coalesce(aod_gfs_joined_with_labels["min_Column_WV"], 
                 aod_gfs_joined_with_labels["min_Column_WV_imputed"]))\
    .withColumn("final_max_Column_WV",
        sf.coalesce(aod_gfs_joined_with_labels["max_Column_WV"], 
                 aod_gfs_joined_with_labels["max_Column_WV_imputed"]))\
    .withColumn("final_median_Column_WV",
        sf.coalesce(aod_gfs_joined_with_labels["median_Column_WV"], 
                 aod_gfs_joined_with_labels["median_Column_WV_imputed"]))

# COMMAND ----------

all_cols = aod_gfs_labels_w_imputation.columns

# COMMAND ----------

for col in cols_to_impute:
    all_cols.pop(all_cols.index(col))
    all_cols.pop(all_cols.index(col+"_imputed"))

# COMMAND ----------

all_cols

# COMMAND ----------

final_train_df = aod_gfs_labels_w_imputation.select(all_cols)

# COMMAND ----------

final_train_df.write.parquet("final_train_df.parquet")

# COMMAND ----------

final_train_df = spark.read.parquet("/final_train_df.parquet")

# COMMAND ----------

final_train_df.select([count(when(isnan(c) | sf.isnull(c), c)).alias(c) for (c,c_type) in final_train_df.dtypes if c not in ('aod_lon_lat_list')]).toPandas()
#columns[:-1] to avoid checking latlon pairs list

# COMMAND ----------

imp_label="min_Optical_Depth_047"
#Define Decision Tree Regressor
imp_gbt = GBTRegressor(featuresCol = "features", labelCol= imp_label, maxIter = 10)

#Fit on train and predict (transform) on test. 
imp_gbt = imp_gbt.fit(imp_train_df)
imputations = imp_gbt.transform(imp_validation_df)

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

# COMMAND ----------

#Linear Regressor
#For pipeline
lr_imp_train_train = imp_train_train.na.drop("any")
lr_imp_train_val = imp_train_val.na.drop("any")

lr_imp_pipelineModel = imp_pipeline.fit(lr_imp_train_train)
lr_imp_train_df = lr_imp_pipelineModel.transform(lr_imp_train_train)
lr_imp_validation_df = lr_imp_pipelineModel.transform(lr_imp_train_val)

#Fit on train and predict (transform) on test. 
imp_lr = LinearRegression(featuresCol = "features", labelCol= "", maxIter = 10)
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

# COMMAND ----------

imp_test_df = imp_pipelineModel.transform(imp_test)
predictions = imp_xgb.transform(imp_test_df).select("grid_id","date","prediction").withColumnRenamed("prediction", to_impute+"_imputed")

# COMMAND ----------

predictions.show(4)

# COMMAND ----------

# MAGIC %md 
# MAGIC # PM2.5 Prediction

# COMMAND ----------

# MAGIC %md TODOs: 
# MAGIC - Create smaller working dataframe for testing (i.e random 6 months across training sample). 
# MAGIC - Implement W261 methods 
# MAGIC   - Probabilistic experiments for model selection
# MAGIC   - Time series valid cross-validation strategy

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Helper functions. 

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

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
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
        numModels = len(epm)
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
        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
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

def compareBaselines(model = None, model_name = None, features = None, paramGrid = None, cv_train_data = None, validation_data = None, test_data = False, label=None):  
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

def plotPerformance(df = None, col = 'r2_score', title = None): 
    '''Given Dataframe of model performance, 
    display formatted bar chart comparison of
    model performance.'''

    x = df['model_name']
    y = df[col]

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

    plt.ylabel(col)

    plt.show()

# COMMAND ----------

def pcaFeatureChoice(df = None):   
    ''' Method to run test on PCA features and normal features. Will return whichever method is preferable
    given F-beta performance and runtime on 10-fold CV'''

    # Manually define non-random search space. This ensures data and params are controlled for in our experimental comparisons, yielding valid results. 
    MIN_INFO_GAIN_SEARCH_LIST = [0.0, 0.2]
    MAX_DEPTH_SEARCH_LIST = [2, 4]
    ### 1.) Decision Tree Classifier ###
    dt_model = DecisionTreeClassifier(featuresCol = 'features', labelCol='dep_del15')

    # DT Param Grid
    dt_paramGrid = ParamGridBuilder() \
      .addGrid(dt_model.minInfoGain, MIN_INFO_GAIN_SEARCH_LIST) \
      .addGrid(dt_model.maxDepth, MAX_DEPTH_SEARCH_LIST) \
      .build()

    ### 2.) Decision Tree Classifier with PCA ###
    dt_model_PCA = DecisionTreeClassifier(featuresCol = 'pca_features', labelCol='dep_del15')

    # DT Param Grid
    dt_PCA_paramGrid = ParamGridBuilder() \
      .addGrid(dt_model.minInfoGain, MIN_INFO_GAIN_SEARCH_LIST) \
      .addGrid(dt_model.maxDepth, MAX_DEPTH_SEARCH_LIST) \
      .build()

    modelListPCA = [(dt_model, "Decision Tree", dt_paramGrid),
               (dt_model_PCA, "PCA", dt_PCA_paramGrid)]


    #Create an empty list of lists that we will append models & performance metrics to.
    # Data order will be: model_name[str], model[obj], features[list], f_beta_score[float], f_beta_std_dev[float], paramGrid [obj] 
    modelComparisonsPCA = []

    #Build comparison table. 
    for model, model_name, paramGrid in modelListPCA: 
        modelComparisonsPCA.append(compareBaselines(train_data, validation_data, model = model, model_name = model_name, paramGrid = paramGrid))

        #model_name[str], model[obj], features[list], precision[float]
        modelComparisonsDF_PCA = pd.DataFrame(modelComparisonsPCA, columns = ['model_name', 'model_obj','feature_names','f_beta_score', 'f_beta_std_dev', 'paramGrid_obj', 'bestParams', 'runtime']).sort_values(by = 'f_beta_score').reset_index(drop=True)

        modelComparisonsDF_PCA = statisticalTestModels(modelComparisonsDF_PCA)

        modelComparisonsDF_PCA

    # If most performanant model is with PCA features, use PCA
    if modelComparisonsDF_PCA['model_name'][1] == 'PCA': 
        features = 'pca_features'
    # If most performant model is not PCA, but DT model is not statistically different, use PCA
    elif modelComparisonsDF_PCA['p_value_to_prev'][1] >= ALPHA_SIG_VALUE: 
        features = 'pca_features'
    # Otherwise use 'common-sense' features. 
    else: 
        features = 'features'

    return features

# COMMAND ----------

def selectBestModel(df): 
    '''Given a sorted (by F-beta score) Dataframe of models, runtimes, and p-values, 
    select the best model that 1.) Is significantly better than previous model (p < alpha)
    OR 2.) Is not statistically better than previous model, but has improved runtime.'''

    # Logic to select the most performant model along both F-beta and runtime. 
    prev_fbeta = None
    alpha_sig_value = 0.05
    prev_runtime = None
    best_model_index = None

    for index, row in df.iterrows(): 
        if index > 0: 
            #Update current row values
            current_fbeta = row['f_beta_score']
            #Current p_value
            current_p_value = row['p_value_to_prev']
            # Current runtime
            current_runtime = row['runtime']
            #If performance is better, with constant or better runtime, select model regardless of significance. 
            if current_fbeta > prev_fbeta and current_runtime <= prev_runtime: 
                best_model_index = index
            #If performance is significantly better, select model so long as runtime is less than 2x the previous model. 
            elif current_fbeta > prev_fbeta and current_p_value < alpha_sig_value and current_runtime < (prev_runtime * 2): 
                best_model_index = index
        else: 
            # Append null if on first row
            best_model_index = index

        #Update the previous row values
        prev_fbeta = row['f_beta_score']
        prev_runtime = row['runtime']


    best_model = df["model_obj"][best_model_index]
    best_model_name = df["model_name"][best_model_index]

    return best_model, best_model_name

# COMMAND ----------

# MAGIC %md Start by only using our clean data for dates that we've pipelined both sources for. 

# COMMAND ----------

clean_df = aod_gfs_joined_with_labels

# COMMAND ----------

# MAGIC %md Now partition dataframe by using date percentiles. 

# COMMAND ----------

# Add percent rank to aid in cross validation/splitting
clean_df = clean_df.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('date')))

# Get 10 folds (20 train + val splits) to generate distribution for statistical testing. 
# Cache for faster access times. 
# For non-Cross Validation (1 fold): 
clean_df = clean_df.withColumn("foldNumber", when((clean_df.rank < .8), lit(0)) \
                                           .otherwise(lit(1))).cache()
# For 6 Folds: 
# clean_df = clean_df.withColumn("foldNumber", when((clean_df.rank < .12), lit(0)) \
#                                              .when((clean_df.rank < .17), lit(1)) \
#                                              .when((clean_df.rank < .28), lit(2)) \
#                                               .when((clean_df.rank < .33), lit(3)) \
#                                            .when((clean_df.rank < .45), lit(4)) \
#                                            .when((clean_df.rank < .50), lit(5)) \
#                                            .when((clean_df.rank < .62), lit(6)) \
#                                            .when((clean_df.rank < .67), lit(7)) \
#                                            .when((clean_df.rank < .78), lit(8)) \
#                                            .when((clean_df.rank < .83), lit(9)) \
#                                            .when((clean_df.rank < .95), lit(10)) \
# #                                            .when((clean_df.rank < .6), lit(11)) \
# #                                            .when((clean_df.rank < .67), lit(12)) \
# #                                            .when((clean_df.rank < .7), lit(13)) \
# #                                            .when((clean_df.rank < .77), lit(14)) \
# #                                            .when((clean_df.rank < .8), lit(15)) \
# #                                            .when((clean_df.rank < .87), lit(16)) \
# #                                            .when((clean_df.rank < .9), lit(17)) \
# #                                            .when((clean_df.rank < .97), lit(18)) \
#                                            .otherwise(lit(11))).cache()

# COMMAND ----------

# clean_df.toPandas()

# COMMAND ----------

# MAGIC %md Train/Test Split for Time Series Cross Validation

# COMMAND ----------

#Create a validation dataframe for our majority class prediction model. This will be all rows with odd numbers in "foldNumber" column (i.e i % 2 != 0). 
train_data = clean_df.where("foldNumber % 2 == 0")
validation_data = clean_df.where("foldNumber % 2 != 0")

# COMMAND ----------

# MAGIC %md Categorize features into str or int feature lists. Drop desired features and drop the label column. Remove features that are of type array. 

# COMMAND ----------

data = train_data

#Remove troublesome features
drop_cols = ['aod_lon_lat_list', 'datetime']
#Remove our train label (PM2.5 'value'). 
label = 'value'

features, int_features, str_features = get_cols(data, drop_cols, label)


# COMMAND ----------

# MAGIC %md
# MAGIC Transform Features for PySpark Models. 

# COMMAND ----------

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
vi = VectorIndexer(inputCol= 'features_assembled', outputCol= 'features_indexed', handleInvalid='keep', maxCategories = 4)
#Add to pipeline. 
stages += [vi]

# Finally standardize features to limit size of feature space (i.e. Categorical Feature 4 has 1049 values which would require max bins = 1050). 
# TODO: Investigate why VectorIndexer isn't binning into 4 categories already. 
scaler = StandardScaler(inputCol="features_indexed", outputCol="features",
                        withStd=True, withMean=False)

stages += [scaler]

#Define model pipeline. 
pipeline = Pipeline(stages = stages)

#Fit transform on train data (excl. validation data to prevent leakage). 
#Transform both train_data and validation_data. 
pipelineModel = pipeline.fit(train_data)
train_df = pipelineModel.transform(train_data)
validation_df = pipelineModel.transform(validation_data)

# COMMAND ----------

# MAGIC %md 
# MAGIC Create a unioned dataframe for CrossValidation, which is simply the train_df + validation_df. This follows valid fit on train and transform train & validation methodology to prevent leakage. 

# COMMAND ----------

# MAGIC %md 1.) Decision Tree Regressor

# COMMAND ----------

# #Define Decision Tree Regressor
# dtr = DecisionTreeRegressor(featuresCol = "features", labelCol= label)

# #Fit on train and predict (transform) on test. 
# dtr = dtr.fit(train_df)
# predictions = dtr.transform(validation_df)

# #Show a couple predictions
# predictions.select("prediction", label, "features").show(5)

# # Select (prediction, true label) and compute r2 and RMSE. 
# evaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
# r2 = evaluator.setMetricName('r2').evaluate(predictions)
# rmse = evaluator.setMetricName('rmse').evaluate(predictions)

# print(f'R-Squared on test data = {r2}')
# print(f'RMSE on test data = {rmse}')

# COMMAND ----------

# #Cross Validator for Time Series
# pipeline = Pipeline(stages=[dtr])
# r2 = RegressionEvaluator(labelCol = label, predictionCol = 'prediction', metricName = 'r2')
# cv = CrossValidator(estimator=pipeline, estimatorParamMaps = dt_paramGrid, evaluator = r2, numFolds = 10, parallelism = 4, foldCol = 'foldNumber', collectSubModels = False)
# cvModel = cv.fit(crossVal_df)
# bestModel = cvModel.bestModel
# # Get average of performance metric (F-beta) for best model 
# avg_r2 = cvModel.avgMetrics[0]
# #Get standard deviation of performance metric (F-beta) for best model
# stdDev = cvModel.stdMetrics[0]
# #Get the best params
# bestParams = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]

# # return [model_name, cvModel, features, r2_score, stdDev, paramGrid, bestParams, runtime]

# print(f'Average R2 on Cross Validation: {avg_r2}')

# COMMAND ----------

# MAGIC %md 2.) Gradient Boosted Tree

# COMMAND ----------

# #Define Decision Tree Regressor
# gbt = GBTRegressor(featuresCol = "features", labelCol= label, maxIter = 10)

# #Fit on train and predict (transform) on test. 
# gbt = gbt.fit(train_df)
# predictions = gbt.transform(validation_df)

# #Show a couple predictions
# predictions.select("prediction", label, "features").show(5)

# # Select (prediction, true label) and compute r2. 
# evaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
# r2 = evaluator.setMetricName('r2').evaluate(predictions)
# rmse = evaluator.setMetricName('rmse').evaluate(predictions)

# print(f'R-Squared on test data = {r2}')
# print(f'RMSE on test data = {rmse}')

# COMMAND ----------

# MAGIC %md 3.) XGBoost Regressor

# COMMAND ----------

# The next step is to define the model training stage of the pipeline. 
# The following command defines a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.
# If you are running Databricks Runtime for Machine Learning 9.0 ML or above, you can set the `num_workers` parameter to leverage the cluster for distributed training.
xgb = XgboostRegressor(num_workers=4, featuresCol = "features", labelCol=label)
#Fit on train and predict (transform) on test. 
xgb = xgb.fit(train_df)
predictions = xgb.transform(validation_df)

#Show a couple predictions
predictions.select("prediction", label, "features").show(5)

# Select (prediction, true label) and compute r2. 
evaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
r2 = evaluator.setMetricName('r2').evaluate(predictions)
rmse = evaluator.setMetricName('rmse').evaluate(predictions)

print(f'R-Squared on test data = {r2}')
print(f'RMSE on test data = {rmse}')

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Time-Series K-Fold CrossValidation on Best Baseline Model (XGBoost)

# COMMAND ----------

# The next step is to define the model training stage of the pipeline. 
# The following command defines a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.
# If you are running Databricks Runtime for Machine Learning 9.0 ML or above, you can set the `num_workers` parameter to leverage the cluster for distributed training.
xgb = XgboostRegressor(num_workers=4, featuresCol = "features", labelCol=label, force_repartition = True)

# COMMAND ----------

# Manually define non-random search space. This ensures data and params are controlled for in our experimental comparisons, yielding valid results. 
MIN_INFO_GAIN_SEARCH_LIST = [0.0, 0.2, 0.6, 1.0]
MAX_DEPTH_SEARCH_LIST = [2, 6]
N_ESTIMATOR_LIST = [10]

# COMMAND ----------

# # DT Param Grid
xgb_paramGrid = ParamGridBuilder() \
    .addGrid(xgb.max_depth, MAX_DEPTH_SEARCH_LIST) \
    .addGrid(xgb.n_estimators, N_ESTIMATOR_LIST) \
    .build()

# COMMAND ----------

#Cross Validator for Time Series
pipeline = Pipeline(stages=[xgb])
r2 = RegressionEvaluator(labelCol = label, predictionCol = 'prediction', metricName = 'r2')
cv = CrossValidator(estimator=pipeline, estimatorParamMaps = xgb_paramGrid, evaluator = r2, numFolds = 6, parallelism = 4, foldCol = 'foldNumber', collectSubModels = False)
cvModel = cv.fit(crossVal_df)
bestModel = cvModel.bestModel
# Get average of performance metric (F-beta) for best model 
avg_r2 = cvModel.avgMetrics[0]
#Get standard deviation of performance metric (F-beta) for best model
stdDev = cvModel.stdMetrics[0]
#Get the best params
bestParams = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]

# return [model_name, cvModel, features, r2_score, stdDev, paramGrid, bestParams, runtime]

print(f'Average R2 on Cross Validation: {avg_r2}')

# COMMAND ----------

print(bestParams)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Neural Network

# COMMAND ----------

!pip install pytorch-tabnet

# COMMAND ----------

import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetRegressor

# COMMAND ----------

display(dbutils.fs.ls("/mnt/capstone/model"))

# COMMAND ----------

from pytorch_tabnet.tab_model import TabNetRegressor

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

train = aod_filtered_gfs_elev_wlabels_joined_wimputations.toPandas()
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

for col in train.columns[train.dtypes == 'float64']:
    train.fillna(train.loc[train_indices, col].mean(), inplace=True)

# COMMAND ----------

unused_feat = ['Set']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# define your embedding sizes : here just a random choice
cat_emb_dim = cat_dims

# COMMAND ----------

clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)

# COMMAND ----------

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices].reshape(-1, 1)

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices].reshape(-1, 1)

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices].reshape(-1, 1)

# COMMAND ----------

max_epochs = 1000 if not os.getenv("CI", False) else 2

# COMMAND ----------

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmse', 'mae'],
    max_epochs=max_epochs,
    patience=50,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False
) 

# COMMAND ----------

preds = clf.predict(X_test)

y_true = y_test

test_score = r2_score(y_pred=preds, y_true=y_true)
valid_score = r2_score(y_pred=clf.predict(X_valid), y_true=y_valid)

print(f"BEST VALID SCORE FOR {valid_score}")
print(f"FINAL TEST SCORE FOR {test_score}")

# COMMAND ----------

