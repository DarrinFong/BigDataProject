from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


def initSpark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config('spark.executor.memory', '32g') \
        .config('spark.driver.memory', '32g') \
        .getOrCreate()
    return spark


def dataParallelization():
    spark = initSpark()
    df = spark.read.format('csv')\
        .option('encoding', "UTF-8")\
        .option('header', 'true')\
        .load("./data/hotel_bookings.csv")
    return df
