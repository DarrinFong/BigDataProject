from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from data_init import dataParallelization
from data.data_type_by_column import type_by_column
from data.data_columns import data_columns
from data.data_categorical_columns import categorical_columns

def prepare_data(column):
    df = dataParallelization()
    df = columns_type_casting(df)
    df = df.withColumn("label", df[column])
    df = filter_data(df, column)
    df = categorical_to_bin_vector(df)
    df = manipulate_features_column(df)
    return df


def columns_type_casting(df):
    for column in type_by_column:
        df = df.withColumn(column, df[column].cast(type_by_column[column]))
    return df


def filter_data(df, column):
    #toDrop = ['reservation_status_date', 'previous_bookings_not_canceled']
    toDrop = ['reservation_status_date']
    toDrop.append(column)
    for x in toDrop:
        if x in data_columns:
            data_columns.remove(x)
        if x in categorical_columns:
            categorical_columns.remove(x)
        df.drop(x)

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


def manipulate_features_column(df):
    # Use VectorAssembler to combine all the feature columns into a single vector column
    new_cat_columns = [col + "_vector" for col in categorical_columns]
    columns = [col for col in data_columns if col not in categorical_columns]
    columns.extend(new_cat_columns)
    assembler = VectorAssembler(inputCols=columns, outputCol="features", handleInvalid="keep")
    pipeline = Pipeline(stages=[assembler])
    df = pipeline.fit(df).transform(df)

    return df


def sampleBooleanColumn_data(df, column, seed, split=[0.7, 0.3]):
    p = df.where(df[column] == 0)
    n = df.where(df[column] == 1)

    pTrain, pTest = p.randomSplit(split, seed=seed)
    nTrain, nTest = n.randomSplit(split, seed=seed)

    train, test = pTrain.union(nTrain), pTest.union(nTest)
    return train, test


def sample_data(df, column, desiredRatio=0.9):
    count = df.count()
    class0 = df.where(df[column] == 0)
    class0Count = class0.count()
    class1 = df.where(df[column] == 1)
    class1Count = class1.count()
    
    majority, majorityCount = None, 0
    minority, minorityCount = None, 0
    if class0Count >= class1Count:
        majority = class0
        majorityCount = class0Count
        minority = class1
        minorityCount = class1Count
    else:
        majority = class1
        majorityCount = class1Count
        minority = class0
        minorityCount = class0Count

    numToDownsampleTo = desiredRatio * minorityCount / (1-desiredRatio)
    ratioToSample = numToDownsampleTo / majorityCount
    sampledMajority = majority.sample(ratioToSample)
    
    return sampledMajority.union(minority)