from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from data_preparation import dataParallelization
from data.data_type_by_column import type_by_column


def columns_type_casting(df):
    for column in type_by_column:
        df = df.withColumn(column, df[column].cast(type_by_column[column]))
    return df
