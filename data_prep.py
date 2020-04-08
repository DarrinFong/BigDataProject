from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from data_init import dataParallelization
from data.data_type_by_column import type_by_column
from data.data_columns import data_columns
from data.data_categorical_columns import categorical_columns

def prepare_data():
    df = dataParallelization()
    df = columns_type_casting(df)
    df = filter_data(df)
    df = categorical_to_bin_vector(df)
    df = Add_Features_Column(df)
    return df


def columns_type_casting(df):
    for column in type_by_column:
        df = df.withColumn(column, df[column].cast(type_by_column[column]))
    return df


def filter_data(df):
    return df


def categorical_to_bin_vector(df):
    indexers, encoders = [], []
    for catColumn in categorical_columns:
        indexers.append(StringIndexer(inputCol=catColumn, outputCol=catColumn + "_index", handleInvalid="keep"))
        encoders.append(OneHotEncoder(inputCol=catColumn + "_index", outputCol=catColumn + "_vector"))

    indexer_pipeline = Pipeline(stages=indexers)
    encoder_pipeline = Pipeline(stages=encoders)
    df_indexed = indexer_pipeline.fit(df).transform(df)
    df_encoded = encoder_pipeline.fit(df_indexed).transform(df_indexed)

    for catColumn in categorical_columns:
        df_encoded = df_encoded.drop(catColumn).drop(catColumn + "_index")

    return df_encoded


def Add_Features_Column(df):
    # Use VectorAssembler to combine all the feature columns into a single vector column
    new_cat_columns = [col + "_vector" for col in categorical_columns]
    columns = [col for col in data_columns if col not in categorical_columns]
    columns.extend(new_cat_columns)
    columns.remove("reservation_status_date")
    df = df.drop("reservation_status_date")
    assembler = VectorAssembler(inputCols=columns, outputCol="features", handleInvalid="keep")
    pipeline = Pipeline(stages=[assembler])
    df = pipeline.fit(df).transform(df)

    return df