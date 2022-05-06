# Databricks notebook source
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
import datetime

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

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.simplefilter('ignore')

# COMMAND ----------

# MAGIC %md 
# MAGIC #INGEST

# COMMAND ----------

# MAGIC %md
# MAGIC ## Elevation

# COMMAND ----------

df_elev = spark.read.parquet("dbfs:/mnt/capstone/train/df_elev_lat_lon_level.parquet")

# COMMAND ----------

display(df_elev)

# COMMAND ----------

# MAGIC %md
# MAGIC ##GFS

# COMMAND ----------

GFS_UNION = False

# COMMAND ----------

if GFS_UNION:
    df_GFS = spark.read.parquet("dbfs:/mnt/capstone/train/df_GFS.parquet")
    df_GFS = df_GFS.withColumn("latitude_double",df_GFS.latitude.cast(DoubleType()))\
                   .withColumn("longitude_double",df_GFS.longitude.cast(DoubleType()))
    df_GFS = df_GFS.drop("latitude").drop("longitude")
    df_GFS = df_GFS.withColumnRenamed("latitude_double","latitude")\
                   .withColumnRenamed("longitude_double","longitude")

# COMMAND ----------

if GFS_UNION:
    df_20190901 = spark.read.parquet("/mnt/capstone/train/gfs.0p25.20210901.f006.parquet")
    df_201801 = spark.read.parquet("s3://capstone-particulate-storage/GFS/201801/")
    df_gfs_2022 = spark.read.parquet("s3://capstone-particulate-storage/GFS_2022/")

# COMMAND ----------

if GFS_UNION:
    df_GFS_all = df_GFS.unionAll(df_20190901).unionAll(df_201801).unionAll(df_gfs_2022) 

# COMMAND ----------

if GFS_UNION:
    df_GFS_all.write.mode("overwrite").parquet("/mnt/capstone/train/df_GFS_all.parquet")

# COMMAND ----------

df_GFS_all = spark.read.parquet("/mnt/capstone/train/df_GFS_all.parquet")

# COMMAND ----------

display(df_GFS_all)

# COMMAND ----------

df_GFS_all.agg({'longitude':'max','latitude':'max'}).show()

# COMMAND ----------

df_GFS_all.agg({'longitude':'min','latitude':'min'}).show()

# COMMAND ----------

df_GFS_all = df_GFS_all.withColumn("lon_minus180_plus180", df_GFS_all.longitude-180) 

# COMMAND ----------

df_GFS_all.agg({'lon_minus180_plus180':'max'}).show()

# COMMAND ----------

df_GFS_all.agg({'lon_minus180_plus180':'min'}).show()

# COMMAND ----------

df_GFS_all = df_GFS_all.drop("longitude")

# COMMAND ----------

df_GFS_all = df_GFS_all.withColumnRenamed("lon_minus180_plus180","longitude")

# COMMAND ----------

# MAGIC %md
# MAGIC ##AOD - EE

# COMMAND ----------

df_aod_ee = spark.read.parquet("/mnt/capstone/train/ee_aod/")

# COMMAND ----------

df_aod_ee = df_aod_ee.withColumn("aod_reading_time",sf.to_timestamp(sf.from_unixtime(df_aod_ee.time/1000)))

# COMMAND ----------

display(df_aod_ee)

# COMMAND ----------

df_aod_ee.where((sf.round(df_aod_ee.latitude,2)==28.53) & (sf.round(df_aod_ee.longitude,2)==77.27) &
                (df_aod_ee.aod_reading_time <= datetime.strptime("2019-09-22 23:45:00", "%Y-%m-%d %H:%M:%S")) &
                (df_aod_ee.aod_reading_time >= datetime.strptime("2019-09-21 23:45:00", "%Y-%m-%d %H:%M:%S"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Labels - OpenAQ

# COMMAND ----------

final_labels_spark = spark.read.parquet("/mnt/capstone/train/final_labels.parquet")

# COMMAND ----------

final_labels_spark.where(final_labels_spark['value'] < 0).count()

# COMMAND ----------

final_labels_spark.where(final_labels_spark['value'] > 600).count()

# COMMAND ----------

display(final_labels_spark)

# COMMAND ----------

labels_column = final_labels_spark.columns
labels_column

# COMMAND ----------

final_labels_spark.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #JOIN

# COMMAND ----------

# MAGIC %md
# MAGIC ##AOD-Labels

# COMMAND ----------

aod_labels = final_labels_spark.join(df_aod_ee, 
                                     on=[(sf.round(df_aod_ee.latitude,2)==sf.round(final_labels_spark.latitude,2)) & 
                                         (sf.round(df_aod_ee.longitude,2)==sf.round(final_labels_spark.longitude,2)) & 
                                         (df_aod_ee.aod_reading_time<=final_labels_spark.datetime_utc) & 
                                         (df_aod_ee.aod_reading_time>=sf.date_sub(final_labels_spark.datetime_utc, 1))], 
                                     how="left").drop(df_aod_ee.latitude).drop(df_aod_ee.longitude)

# COMMAND ----------

aod_labels = aod_labels.groupBy(labels_column+['grid_id']).mean("Optical_Depth_047")

# COMMAND ----------

display(aod_labels)

# COMMAND ----------

aod_labels.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Elev-AOD-Lables

# COMMAND ----------

elev_aod_labels = aod_labels.join(df_elev, 
                                  on=[(aod_labels.longitude==df_elev.longitude) & 
                                      (aod_labels.latitude==df_elev.latitude)],
                                 how='left').drop(df_elev.longitude).drop(df_elev.latitude)

# COMMAND ----------

display(elev_aod_labels)

# COMMAND ----------

elev_aod_labels.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##GFS-Elev-AOD-Lables

# COMMAND ----------

gfs_elev_aod_labels_tmp = elev_aod_labels.join(df_GFS_all,
                                           on=[((sf.round(df_GFS_all.latitude,2)+12.5>=sf.round(elev_aod_labels.latitude,2))&
                                                (sf.round(df_GFS_all.latitude,2)-12.5<=sf.round(elev_aod_labels.latitude,2))) & 
                                         ((sf.round(df_GFS_all.longitude,2)+12.5>=sf.round(elev_aod_labels.longitude,2)) &
                                          (sf.round(df_GFS_all.longitude,2)-12.5<=sf.round(elev_aod_labels.longitude,2))) & 
                                         ((df_GFS_all.date==elev_aod_labels.date_utc) |
                                         (df_GFS_all.date==sf.date_add(elev_aod_labels.date_utc, 1)))], 
                                           how="left").drop(df_GFS_all.latitude).drop(df_GFS_all.longitude).drop(df_GFS_all.grid_id)

# COMMAND ----------

display(gfs_elev_aod_labels_tmp)

# COMMAND ----------

gfs_elev_aod_labels_tmp.select([count(when(gfs_elev_aod_labels_tmp[c]==None, c)).alias(c) for (c,c_type) in gfs_elev_aod_labels_tmp.dtypes if c not in ('date_utc','datetime_utc')]).toPandas()

# COMMAND ----------

cols_00 = [col for col in df_GFS_all.columns if '00' in col]
cols_06 = [col for col in df_GFS_all.columns if '06' in col]
cols_12 = [col for col in df_GFS_all.columns if '12' in col]
cols_18 = [col for col in df_GFS_all.columns if '18' in col]

# COMMAND ----------

cols_00

# COMMAND ----------

for col in cols_00:
    gfs_elev_aod_labels_tmp = gfs_elev_aod_labels_tmp\
    .withColumn(col,when((((gfs_elev_aod_labels_tmp.date_utc == gfs_elev_aod_labels_tmp.date) 
                                          & (sf.hour(gfs_elev_aod_labels_tmp.datetime_utc)<0)) | 
                                ((gfs_elev_aod_labels_tmp.date_utc < gfs_elev_aod_labels_tmp.date) 
                                         & (sf.hour(gfs_elev_aod_labels_tmp.datetime_utc)>=0)))
        ,gfs_elev_aod_labels_tmp[col]).otherwise(-math.inf))\
    .drop(gfs_elev_aod_labels_tmp[col])

# COMMAND ----------

for col in cols_06:
    gfs_elev_aod_labels_tmp = gfs_elev_aod_labels_tmp\
    .withColumn(col,when((((gfs_elev_aod_labels_tmp.date_utc == gfs_elev_aod_labels_tmp.date) 
                                          & (sf.hour(gfs_elev_aod_labels_tmp.datetime_utc)<6)) | 
                                ((gfs_elev_aod_labels_tmp.date_utc < gfs_elev_aod_labels_tmp.date) 
                                         & (sf.hour(gfs_elev_aod_labels_tmp.datetime_utc)>=6)))
        ,gfs_elev_aod_labels_tmp[col]).otherwise(-math.inf))\
    .drop(gfs_elev_aod_labels_tmp[col])

# COMMAND ----------

for col in cols_12:
    gfs_elev_aod_labels_tmp = gfs_elev_aod_labels_tmp\
    .withColumn(col,when((((gfs_elev_aod_labels_tmp.date_utc == gfs_elev_aod_labels_tmp.date) 
                                          & (sf.hour(gfs_elev_aod_labels_tmp.datetime_utc)<12)) | 
                                ((gfs_elev_aod_labels_tmp.date_utc < gfs_elev_aod_labels_tmp.date) 
                                         & (sf.hour(gfs_elev_aod_labels_tmp.datetime_utc)>=12)))
        ,gfs_elev_aod_labels_tmp[col]).otherwise(-math.inf))\
    .drop(gfs_elev_aod_labels_tmp[col])

# COMMAND ----------

for col in cols_18:
    gfs_elev_aod_labels_tmp = gfs_elev_aod_labels_tmp\
    .withColumn(col,when((((gfs_elev_aod_labels_tmp.date_utc == gfs_elev_aod_labels_tmp.date) 
                                          & (sf.hour(gfs_elev_aod_labels_tmp.datetime_utc)<18)) | 
                                ((gfs_elev_aod_labels_tmp.date_utc < gfs_elev_aod_labels_tmp.date) 
                                         & (sf.hour(gfs_elev_aod_labels_tmp.datetime_utc)>=18)))
        ,gfs_elev_aod_labels_tmp[col]).otherwise(-math.inf))\
    .drop(gfs_elev_aod_labels_tmp[col])

# COMMAND ----------

gfs_elev_aod_labels_tmp.count()

# COMMAND ----------

gfs_elev_aod_labels_tmp.select([count(when(gfs_elev_aod_labels_tmp[c]==-math.inf, c)).alias(c) for (c,c_type) in gfs_elev_aod_labels_tmp.dtypes if c not in ('date_utc','datetime_utc')]).toPandas()

# COMMAND ----------

gfs_elev_aod_labels = gfs_elev_aod_labels_tmp.groupby(labels_column+['grid_id','avg(Optical_Depth_047)','min_elevation','max_elevation','avg_elevation']).max()

# COMMAND ----------

gfs_elev_aod_labels.select([count(when(gfs_elev_aod_labels[c]==-math.inf, c)).alias(c) for (c,c_type) in gfs_elev_aod_labels.dtypes if c not in ('date_utc','datetime_utc')]).toPandas()

# COMMAND ----------

gfs_elev_aod_labels.where(gfs_elev_aod_labels['t_surface00']>0).count()

# COMMAND ----------

gfs_elev_aod_labels.count()

# COMMAND ----------

display(gfs_elev_aod_labels)

# COMMAND ----------

dropcols = ['max(latitude)',
            'max(longitude)',
            'max(value)',
            'max(value_lag1day)',
            'max(difflag)',
            'max(value_lag7day)',
            'max(value_lag30day)',
            'max(difflag_lag1day)',
            'max(difflag_lag7day)',
            'max(difflag_lag30day)',
            'max(avg(Optical_Depth_047))',
            'max(min_elevation)',
            'max(max_elevation)',
            'max(avg_elevation)']

# COMMAND ----------

gfs_elev_aod_labels = gfs_elev_aod_labels.drop(*dropcols)

# COMMAND ----------

gfs_elev_aod_labels = gfs_elev_aod_labels.withColumnRenamed("avg(Optical_Depth_047)","Optical_Depth_047")

# COMMAND ----------

cols_max = [col for col in gfs_elev_aod_labels.columns if 'max(' in col]
cols_max

# COMMAND ----------

for col in cols_max:
    gfs_elev_aod_labels=gfs_elev_aod_labels.withColumnRenamed(col,col.replace("max(","").replace(")",""))

# COMMAND ----------

gfs_elev_aod_labels.write.mode("overwrite").parquet("/mnt/capstone/lat_lon_level_gfs_elev_aod_labels.parquet")

# COMMAND ----------

