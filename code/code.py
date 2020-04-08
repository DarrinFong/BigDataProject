import os
import sys
sys.path.append(".")
import copy
import time
import random
import pyspark
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import desc, size, max, abs, explode, collect_list

from data_cleaning import columns_type_casting
from data_preparation import dataParallelization
from data.data_columns import data_columns

def decision_tree():
    data = dataParallelization()
    data = columns_type_casting(data)

    (training_data, test_data) = data.randomSplit([0.8, 0.2])

    inputs = ["country","stays_in_week_nights", "adr", "customer_type"]

    assembler = VectorAssembler(inputCols = inputs, outputCol = "features")

    test = assembler.transform(training_data)
    test.show(1)
    
    classifier = DecisionTreeClassifier(labelCol="label", featuresCol = "features")


    pipe = Pipeline(stages=[assembler, classifier])

    model = pipe.fit(training_data)
    
    # predictions = model.transform(test_data)
    # predictions.show()
    

decision_tree()