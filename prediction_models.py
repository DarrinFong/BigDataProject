from data_cleaning import prepare_data
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier

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
    
    print ("-------------------\nRandom Forest Evaluation:")
    evaluate_predictions(predictions)
    print("-------------------")

def Apply_Decision_Tree(df, column):
    df = df.withColumn("label", column)

    # Randomly split data into training and test dataset
    (train_data, test_data) = df.randomSplit([0.8, 0.2], seed=420)
    
    # Free up some memory
    # Train DecisionTree model
    rf = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    rf_model = rf.fit(train_data)

    # Make predictions on test data
    predictions = rf_model.transform(test_data)

    print ("-------------------\nDecision Tree Evaluation:")
    evaluate_predictions(predictions)
    print("-------------------")

def extrapolatePositivesNegatives(predictions):
    possibleOutputs = ['TN', 'FP', 'FN', 'TP']

    def mapRow(row):
        prediction = row.prediction
        actual = row.label

        # TN - 0 + 0 = 0
        # FP - 1 + 0 = 1
        # FN - 0 + 2 = 2
        # TP - 1 + 2 = 3
        output = possibleOutputs[int(prediction + 2*actual)]

        return (output, 1)

    positivesNegatives = predictions.rdd.map(lambda x: mapRow(x)).reduceByKey(lambda x, y: x+y).collect()

    output = {x[0]: x[1] for x in positivesNegatives}
    for x in possibleOutputs:
        if (x not in output):
            output[x] = 0

    return output

def evaluate_predictions(predictions):
    evaluation = extrapolatePositivesNegatives(predictions)
    TP = evaluation['TP']
    TN = evaluation['TN']
    FP = evaluation['FP']
    FN = evaluation['FN']

    accuracy = 0 if (TP + TN == 0) else (TP+TN)/(TP+FP+TN+FN)
    precision = 0 if TP==0 else (TP)/(TP+FP)
    recall = 0 if TP==0 else (TP)/(TP+FN)
    f1_score = 0 if (precision + recall == 0) else (2.0*precision*recall)/(precision+recall)

    evaluation['accuracy'] = accuracy
    evaluation['precision'] = precision
    evaluation['recall'] = recall
    evaluation['f1_score'] = f1_score

    print(evaluation)

df = prepare_data()
Apply_Random_Forest(df, df.is_repeated_guest)
Apply_Decision_Tree(df, df.is_repeated_guest)