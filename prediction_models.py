from data_cleaning import dataParallelization, columns_type_casting, categorical_to_bin_vector, Add_Features_Column
from pyspark.ml.classification import RandomForestClassifier


def Apply_Random_Forest(df, column):
    df = df.withColumn("label", column)

    # Randomly split data into training and test dataset
    (train_data, test_data) = df.randomSplit([0.7, 0.3], seed=111)

    # Free up some memory
    # Train RandomForest model
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    rf_model = rf.fit(train_data)

    # Make predictions on test data
    predictions = rf_model.transform(test_data)
    predictions.show()


df = dataParallelization()
df = columns_type_casting(df)
df = categorical_to_bin_vector(df)
df = Add_Features_Column(df)
Apply_Random_Forest(df, df.babies)