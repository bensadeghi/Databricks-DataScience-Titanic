# Databricks notebook source
# MAGIC %md
# MAGIC # Machine Learning Workflows
# MAGIC 
# MAGIC Machine learning practitioners generally follow an iterative workflow.  This lesson walks through that workflow at a high level before exploring train/test splits, a baseline model, and evaluation.
# MAGIC 
# MAGIC #### Agenda:
# MAGIC * Define the data analytics development cycle
# MAGIC * Perform a split between training and test data
# MAGIC * Track model development with MLflow
# MAGIC * Train a baseline model
# MAGIC * Evaluate a baseline model's performance
# MAGIC * Train a decision tree
# MAGIC * Evaluate decision tree model's performance

# COMMAND ----------

# MAGIC %run ./Classroom-Setup

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### The Development Cycle
# MAGIC 
# MAGIC Data scientists follow an iterative workflow that keeps their work closely aligned to both business problems and their data.  This cycle begins with a thorough understanding of the business problem and the data itself, a process called _exploratory data analysis_.  Once the motivating business question and data are understood, the next step is preparing the data for modeling.  This includes removing or imputing missing values and outliers as well as creating features to train the model on.  The majority of a data scientist's work is spent in these earlier steps.
# MAGIC 
# MAGIC After preparing the features in a way that the model can benefit from, the modeling stage uses those features to determine the best way to represent the data.  The various models are then evaluated and this whole process is repeated until the best solution is developed and deployed into production.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/CRISP-DM.png" style="height: 500px; margin:10px"/></div>
# MAGIC 
# MAGIC The above model addresses the high-level development cycle of data products.  This lesson addresses how to implement this at more practical level.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> <a href="https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining" target="_blank">See the Cross-Industry Standard Process for Data Mining</a> for details on the method above.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Train/Test Split
# MAGIC 
# MAGIC To implement the development cycle detailed above, data scientists first divide their data randomly into two subsets.  This allows for the evaluation of the model on unseen data.<br><br>
# MAGIC 
# MAGIC 1. The **training set** is used to train the model on
# MAGIC 2. The **test set** is used to test how well the model performs on unseen data
# MAGIC 
# MAGIC This split avoids the memorization of data, known as **overfitting**.  Overfitting occurs when our model learns patterns caused by random chance rather than true signal.  By evaluating our model's performance on unseen data, we can minimize overfitting.
# MAGIC 
# MAGIC Splitting training and test data should be done so that the amount of data in the test set is a good sample of the overall data.  **A split of 80% of your data in the training set and 20% in the test set is a good place to start.**
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/train-test-split.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC Import the cleansed Titanic dataset

# COMMAND ----------

titanicDF = spark.read.table("titanic_clean").cache()

display(titanicDF)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Split the dataset into two DataFrames:<br><br>
# MAGIC 
# MAGIC 1. `trainDF`: our training data
# MAGIC 2. `testDF`: our test data
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Using a seed ensures that the random split we conduct will be the same split if we rerun the code again.  Reproducible experiments are a hallmark of good science.<br>
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Conventions using other machine learning tools often entail creating 4 objects: `X_train`, `y_train`, `X_test`, and `y_test` where your features `X` are separate from your label `y`.  Since Spark is distributed, the Spark convention keeps the features and labels together when the split is performed.

# COMMAND ----------

trainDF, testDF = titanicDF.randomSplit([0.8, 0.2], seed=10)

print("We have {} training examples and {} test examples.".format(trainDF.count(), testDF.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline Model
# MAGIC 
# MAGIC A **baseline model** offers an educated best guess to improve upon as different models are trained and evaluated.  It represents the simplest model we can create.  This is generally approached as the center of the data.  In the case of regression, this could involve predicting the average of the outcome regardless of the features it sees.  In the case of classification, the center of the data is the mode, or the most common class.  
# MAGIC 
# MAGIC A baseline model could also be a random value or a preexisting model.  Through each new model, we can track improvements with respect to this baseline.

# COMMAND ----------

# MAGIC %md
# MAGIC Create a baseline model by calculating the most common Survival status (rounding of average) in the training dataset.

# COMMAND ----------

from pyspark.sql.functions import avg

trainAvg = trainDF.select(avg("Survived")).first()[0]
trainAvg = float(round(trainAvg))

print("Common Survival Status: {}".format(trainAvg))

# COMMAND ----------

# MAGIC %md
# MAGIC Take the average calculated on the training dataset and append it as the column `prediction` on the test dataset.

# COMMAND ----------

from pyspark.sql.functions import lit

testPredictionDF = testDF.withColumn("prediction", lit(trainAvg))

display(testPredictionDF)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Evaluation and Improvement
# MAGIC 
# MAGIC Evaluation offers a way of measuring how well predictions match the observed data.  In other words, an evaluation metric measures how closely predicted responses are to the true response.
# MAGIC 
# MAGIC There are a number of different evaluation metrics.  The most common evaluation metric in classification tasks is **accuracy**.  This is calculated by the sum of correct predictions, divided by the total number of predictions. Technically:
# MAGIC 
# MAGIC Accuracy = (TP + TN) / (TP + TN + FP + FN)
# MAGIC 
# MAGIC where: TP = True positive; FP = False positive; TN = True negative; FN = False negative 
# MAGIC 
# MAGIC See [Evaluation of binary classifiers](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
# MAGIC 
# MAGIC Since we care about how our model performs on unseen data, we are more concerned about the test error, or the accuracy calculated on the unseen data.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Define the evaluator with the prediction column, label column, and accuracy metric.

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="Survived", metricName="accuracy")

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate `testPredictionDF` using the `.evaluator()` method, and track the accuracy metric with MLflow

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name="baseline_model"):
  # Log the dummy model parameter value
  mlflow.log_param("maxDepth", 0)
  
  # Log corresponding model accuracy
  accuracy = evaluator.evaluate(testPredictionDF)
  mlflow.log_metric("accuracy", accuracy)
  print("Accuracy on the test set for the baseline model: {}".format(accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC This score indicates that the accuracy between the true survival status and the prediction of the baseline is about 54%.<br>
# MAGIC That's not great, but it's also not too bad for a naive approach.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree for Classification

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC [Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning) is a method commonly used in data mining. The goal is to create a model that predicts the value of a target variable based on several input variables. An example is shown in the diagram below. Each interior node corresponds to one of the input variables; there are edges to children for each of the possible values of that input variable. Each leaf represents a value of the target variable given the values of the input variables represented by the path from the root to the leaf.
# MAGIC 
# MAGIC ![x](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)
# MAGIC 
# MAGIC A decision tree is a simple representation for classifying examples. For this section, assume that all of the input features have finite discrete domains, and there is a single target feature called the "classification". Each element of the domain of the classification is called a class. A decision tree or a classification tree is a tree in which each internal (non-leaf) node is labeled with an input feature. The arcs coming from a node labeled with an input feature are labeled with each of the possible values of the target or output feature or the arc leads to a subordinate decision node on a different input feature. Each leaf of the tree is labeled with a class or a probability distribution over the classes, signifying that the data set has been classified by the tree into either a specific class.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's build a decision tree using the training data set

# COMMAND ----------

import mlflow.spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# Vectorize the features (all columns excluding the first one, Survived)
features = trainDF.columns[1:]
assembler = VectorAssembler(inputCols=features, outputCol="features")
assembledTrainDF = assembler.transform(trainDF)

# Start new MLflow run
mlflow.start_run(run_name="shallow_tree")

# Set decision tree `maxDepth` parameter to 2, logging with MLflow
maxDepth = 2
mlflow.log_param("maxDepth", maxDepth)

# Train a decision tree
dtc = DecisionTreeClassifier(featuresCol="features", labelCol="Survived", maxDepth=maxDepth)

# Log and save model as an MLflow artifact
dtcModel = dtc.fit(assembledTrainDF)
mlflow.spark.log_model(dtcModel, "model")

# Print the constructed tree
print(dtcModel.toDebugString)

# COMMAND ----------

# Visualize the decision tree

display(dtcModel)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Feature Importance

# COMMAND ----------

# zip the list of features with their scores
scores = zip(assembler.getInputCols(), dtcModel.featureImportances)

# and pretty print theem
for x in scores: print("%-15s = %s" % x)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Model Performance

# COMMAND ----------

# Vectorize the features of test set
assembledTestDF = assembler.transform(testDF)

# Make predictions using vectorized test set
testPredictionDF = dtcModel.transform(assembledTestDF)

display(testPredictionDF)

# COMMAND ----------

# Generation accuracy metric and log it in MLflow
accuracy = evaluator.evaluate(testPredictionDF)
mlflow.log_metric("accuracy", accuracy)
print("Accuracy on the test set for the decision tree model: {}".format(accuracy))

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC [Model Selection]($./05-Model-Selection)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>