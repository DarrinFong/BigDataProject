# NOTE - ExtractFeaturesImp was acquired from https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
from data_prep import prepare_data, sample_data
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
import pandas
from datetime import datetime

split = [0.7, 0.3]
seed = datetime.now().microsecond

def Apply_Random_Forest(df, column, columnIsBinary=True, numTrees=10, maxDepth=5):
    # Randomly split data into training and test dataset
    (train_data, test_data) = df.randomSplit(split, seed=seed)
    train_data = sample_data(train_data, column)

    # Free up some memory
    # Train RandomForest model
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=numTrees, maxDepth=maxDepth)
    rf_model = rf.fit(train_data)

    # Make predictions on test data
    predictions = rf_model.transform(test_data)
    
    print("-------------------\nRandom Forest Evaluation, T=" + str(numTrees) + ", mD=" + str(maxDepth))
    evaluate_predictions(predictions, columnIsBinary)
    print("Feature Importance:")
    test = ExtractFeatureImp(rf_model.featureImportances, df, 'features')
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(test)
    print("-------------------")

def Apply_Decision_Tree(df, column, columnIsBinary = True, maxDepth=5):
    # Randomly split data into training and test dataset
    (train_data, test_data) = df.randomSplit(split, seed=seed)
    train_data = sample_data(train_data, column)

    # Train DecisionTree model
    rf = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=maxDepth)
    rf_model = rf.fit(train_data)

    # Make predictions on test data
    predictions = rf_model.transform(test_data)

    print("-------------------\nDecision Tree Evaluation, mD=" + str(maxDepth))
    evaluate_predictions(predictions, columnIsBinary)
    print("Feature Importance:")
    test = ExtractFeatureImp(rf_model.featureImportances, df, 'features')
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(test)
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

def evaluate_predictions(predictions, columnIsBinary):
    if not columnIsBinary:
        print ("Sorry, haven't handled non-binary prediction evaluations yet :/")
        return
      
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

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pandas.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending=False))

column = 'is_repeated_guest'
df = prepare_data(column)


Apply_Random_Forest(df, column, columnIsBinary=True, numTrees=10, maxDepth=10)
#Apply_Decision_Tree(df, column, columnIsBinary=True, maxDepth=20)

# TODO 
# - Add evaluation for non-binary predictions.
#   See: Cohen's Kappa Coefficient, avg accuracy, median etc. 
# - More data prep. Shouldn't be getting an f1 score of 1/near 1...
# - Convert featureImportances back to original table names so as to be able to understand what's what