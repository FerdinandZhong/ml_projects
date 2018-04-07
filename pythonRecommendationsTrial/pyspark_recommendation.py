# coding: utf-8

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics, MultilabelMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import os.path

sc = SparkContext('local[8]').getOrCreate()
spark = SparkSession(sc)

lines = spark.read.csv( "ratings.csv").rdd

header = lines.first()

# remove header
new_lines = lines.filter(lambda line: line != header)

# generate new RDD with column names
ratingsRDD = new_lines.map(lambda p: Row(userId=int(p[0]), movieId=float(p[1]),rating=float(p[2]), timestamp=int(p[3])))
# create dataframe from RDD
# generate training and test dataframe
ratings = spark.createDataFrame(ratingsRDD)
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
userRecs = model.recommendForAllUsers(10)

# try to evaluate results using mean average precision and other multi-label classification evaluations.
# similar to the evaluation used for the game recommendation system which doesn't have a explicit rating
def otherEvaluations(userRecs, test):
    # prepare recommendation results and modify test dataset
    newRecs = userRecs.rdd.map(lambda row: (row.userId, [float(x[0]) for x in row.recommendations]))
    # only keep movies with a higher rating
    filteredTest = test.rdd.filter(lambda x: x.rating > 2.5)
    groupedTest = filteredTest.sortBy(lambda x: x[2], ascending=False).groupBy(lambda x: x.userId).map(
        lambda rating: (rating[0], [x[0] for x in rating[1]]))
    joinedTest = groupedTest.repartition(4).join(newRecs).map(lambda x: x[1])
    metrics = RankingMetrics(joinedTest)
    print("MAE = %s" % metrics.meanAveragePrecision)
    metrics = MultilabelMetrics(joinedTest)
    print("Recall = %s" % metrics.recall())
    print("Precision = %s" % metrics.precision())
    print("F1 measure = %s" % metrics.f1Measure())
    print("Accuracy = %s" % metrics.accuracy)

otherEvaluations(userRecs, test)
# however, for movie review dataset, the multi-label classification's result is not good. Because when wen split dataset into two sets, we put movies users reviewed
# into two sets, and when regard the situation as multi-label classification, we are trying to use training set to predict what else movies are reviewed by users in test set.
# However, for the gaming recommendation, it's better to use this multi-label classification. Because we are trying to use players' past one month play-record to predict what
# they are going to play in the future. When doing evaluation, we just compare the games list predicted with the real games list as a multi-label classification problem. It can
# help tell us how we predict what users are going to play in the future.



