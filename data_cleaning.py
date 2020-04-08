from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer
from data_preparation import dataParallelization
from data.data_type_by_column import type_by_column


def columns_type_casting(df):
    for column in type_by_column:
        col_type = type_by_column[column]
        if (col_type == "string" or col_type == "date"):
            df = encode_column(df, column)
        elif (col_type == "boolean"):
            df = df.withColumn(column, df[column].cast("integer"))
        else:
            df = df.withColumn(column, df[column].cast(col_type))
    df = df.withColumn("label", df.adr)
    return df

def encode_column(df, col_name):
    indexed_col_name = "indexed_" + col_name
    df = StringIndexer(inputCol = col_name, outputCol = indexed_col_name, handleInvalid="keep").fit(df).transform(df)
    df = df.drop(col_name)
    df = df.withColumnRenamed(indexed_col_name, col_name)
    return df