# coding: utf-8
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics, MultilabelMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType



# try to evaluate results using mean average precision and other multi-label classification evaluations.
# similar to the evaluation used for the game recommendation system which doesn't have a explicit rating
def otherEvaluations(userRecs, test):
    # prepare recommendation results and modify test dataset
    newRecs = userRecs.rdd.map(lambda row: (row.userId, [float(x[0]) for x in row.recommendations]))
    # only keep movies with a higher rating
    filteredTest = test.rdd.filter(lambda x: x.rating > 2.5)
    groupedTest = filteredTest.sortBy(lambda x: x.rating, ascending=False).groupBy(lambda x: x.userId).map(
        lambda rating: (rating[0], [x[0] for x in rating[1]]))
    joinedTest = newRecs.join(groupedTest).repartition(16).map(lambda x: x[1])
    metrics = RankingMetrics(joinedTest)
    print("MAP = %s" % metrics.meanAveragePrecision)
    print("Precesion at 5 = %s" % metrics.precisionAt(5))
    print("Precesion at 10 = %s" % metrics.precisionAt(10))

# however, for movie review dataset, the multi-label classification's result is not good. Because when we split dataset into two sets, we put movies users reviewed
# into two sets, and when regard the situation as multi-label classification, we are trying to use training set to predict what else movies are reviewed by users in test set.
# However, for the gaming recommendation, it's better to use this multi-label classification. Because we are trying to use players' past one month play-record to predict what
# they are going to play in the future. When doing evaluation, we just compare the games list predicted with the real games list as a multi-label classification problem. It can
# help tell us how we predict what users are going to play in the future.

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("als_testing") \
        .config("spark.master", "local[8]") \
        .config("spark.default.parallelism", 16) \
        .getOrCreate()

    schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField('timestamp', IntegerType(), True)
    ])
    df = pd.read_csv('ratings.csv', sep=',')
    df = df.iloc[1:]
    ratings = spark.createDataFrame(df, schema=schema)
    print(ratings.show(10))

    (training, test) = ratings.randomSplit([0.7, 0.3])

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(rank=5, maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    # Generate top 20 movie recommendations for each user
    userRecs = model.recommendForAllUsers(20)

    otherEvaluations(userRecs, test)