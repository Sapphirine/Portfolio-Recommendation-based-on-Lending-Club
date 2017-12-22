

from __future__ import print_function
import sys
if sys.version >= '3':
    long = int


# In[382]:

from pyspark import SparkContext, SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession


# In[3]:

conf = SparkConf('local').setAppName('MyApp')
sc = SparkContext(conf=conf)
if __name__ == "__main__":
    spark = SparkSession        .builder        .appName("Recommendation")        .getOrCreate()


# In[425]:

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os

from sklearn import decomposition
import matplotlib.pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler


# In[63]:
#cold start dataset, cointains over 20 features
pwd = '/Users/RUIest/Desktop/big_data_project/lending_data_clean_v3.csv'
lines = spark.read.text(pwd).rdd
parts = lines.map(lambda row: row.value.split(','))
lend_RDD = parts.map(lambda p: Row(lable=int(p[22]),featuresList=(float(p[0]), float(p[1]),float(p[2]),float(p[3]),float(p[4]),float(p[5]),float(p[6]),
                                                                  float(p[8],float(p[9],float(p[10],float(p[11],float(p[12],float(p[13],float(p[14],float(p[15],float(p[16],
                                                                  float(p[17],float(p[18],float(p[19],float(p[20],float(p[21])))


# In[338]:

# Create a DataFrame
lending_df = spark.createDataFrame(lend_RDD)
lending_df.show(10)


# In[339]:

# Convert feature type to vector
lending_df_vectors = lending_df.rdd.map(lambda row: Row(
    label=row["lable"],
    features = Vectors.dense(row["featuresList"])
)).toDF()


# In[340]:

lending_df_vectors


# In[341]:

# Scale the data 
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)


# In[342]:

scalerModel = scaler.fit(lending_df_vectors)


# In[343]:

scaled_lending = scalerModel.transform(lending_df_vectors)


# In[344]:

scaled_lending.show(10)


# In[345]:

# Split train and test dataset
splits = scaled_lending.select("label","scaledFeatures").randomSplit([0.75, 0.25],32)
train = splits[0]
test = splits[1]


#### Naive Bayes Classification

# In[216]:

from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(smoothing=1.0,featuresCol="scaledFeatures",modelType="multinomial")
model = nb.fit(train)
predictions = model.transform(test)
predictions.show(20)


# In[228]:

# Built Evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
predictions.show(5,False)

print("Test set accuracy = " + str(accuracy))


# ###### Logistic Regression Classifier

# In[348]:

from pyspark.ml.classification import LogisticRegression

blor = LogisticRegression(maxIter=5, regParam=0.01, regType='l1',featuresCol='scaledFeatures')
# Ridge regression
rlor = LogisticRegression(maxIter=5, regParam=0.01, regType='l2',featuresCol='scaledFeatures')
model = blor.fit(train)

help(blor)

# In[349]:

predictions_blor = model.transform(test)
predictions_blor.show(20)


# In[350]:

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(predictions_blor)
predictions_blor.show(10,False)


# In[351]:

print("Test set accuracy = " + str(accuracy))


# ###### Random Forest Classification

# In[304]:

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=4,maxDepth=3,featuresCol="scaledFeatures",seed=10)
rf_model = rf.fit(train)


# In[308]:

predictions_rf = rf_model.transform(test)
predictions_rf.show(20)


# In[309]:

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(predictions_rf)
predictions_rf.show(10,False)


# In[307]:

print("Test set accuracy = " + str(accuracy))



# User interface with lease variables
# In[310]:

pwd = '/Users/RUIest/Desktop/big_data_project/lending_data_clean_v5.csv'
lines = spark.read.text(pwd).rdd

# In[311]:

lines.first()


# In[313]:

parts = lines.map(lambda row: row.value.split())


# In[314]:

lend_user_RDD = parts.map(lambda p: Row(lable=int(p[13]),
                                   featuresList=(int(p[0]), float(p[1]),float(p[2]),
                                                 int(p[3]),int(p[4]),int(p[5]),
                                                 float(p[6]),int(p[7]),int(p[8]),
                                                 float(p[10]))))


# In[315]:

lend_user_df = spark.createDataFrame(lend_user_RDD)
lend_user_df.show(10)


# In[318]:

lend_user_df_vectors = lend_user_df.rdd.map(lambda row: Row(
    label=row["lable"],
    features = Vectors.dense(row["featuresList"])
)).toDF()


# In[320]:

lend_user_df_vectors.show(10)


# In[321]:

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)


# In[322]:

scalerModel = scaler.fit(lend_user_df_vectors)


# In[325]:

scaled_lend_user = scalerModel.transform(lend_user_df_vectors)
scaled_lend_user.show(10)


# In[352]:

(train_lend_user, test_lend_user) = scaled_lend_user.select('label','scaledFeatures').randomSplit([0.75, 0.25],67)


# In[357]:

train_lend_user.count()


# In[358]:

test_lend_user.count()


# ###### Naive Bayes Classification

# In[361]:

from pyspark.ml.classification import NaiveBayes
(train_lend_user, test_lend_user) = scaled_lend_user.select('label','scaledFeatures').randomSplit([0.75, 0.25],21)
nb = NaiveBayes(smoothing=1.0,featuresCol="scaledFeatures",modelType="multinomial")
model_nb = nb.fit(train_lend_user)
predictions_nb = model_nb.transform(test_lend_user)


# In[362]:

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy_nb = evaluator.evaluate(predictions_nb)
print("Test set accuracy = " + str(accuracy_nb))


# ###### Logistic Regression

# In[365]:

# ###### Lasso regression
from pyspark.ml.classification import LogisticRegression
(train_lend_user, test_lend_user) = scaled_lend_user.select('label','scaledFeatures').randomSplit([0.75, 0.25],39)
blor_model = blor.fit(train_lend_user)
predictions_blor = blor_model.transform(test_lend_user)


# In[366]:

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy_blor = evaluator.evaluate(predictions_blor)
print("Test set accuracy = " + str(accuracy_blor))



# In[367]:

# ###### Ridge regression*
from pyspark.ml.classification import LogisticRegression
(train_lend_user, test_lend_user) = scaled_lend_user.select('label','scaledFeatures').randomSplit([0.75, 0.25],79)
rlor_model = rlor.fit(train_lend_user)
predictions_rlor = rlor_model.transform(test_lend_user)


# In[368]:

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy_rlor = evaluator.evaluate(predictions_rlor)
print("Test set accuracy = " + str(accuracy_rlor))



# ##### Random Forest

# In[371]:

from pyspark.ml.classification import RandomForestClassifier
(train_lend_user, test_lend_user) = scaled_lend_user.select('label','scaledFeatures').randomSplit([0.8, 0.2],67)
rf = RandomForestClassifier(numTrees=7,maxDepth=4,featuresCol="scaledFeatures",seed=1234)
rf_model = rf.fit(train_lend_user)
predictions_rf = rf_model.transform(test_lend_user)
predictions_rf.show(20)


# In[372]:

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy_rf = evaluator.evaluate(predictions_rf)
print("Test set accuracy = " + str(accuracy_rf))

