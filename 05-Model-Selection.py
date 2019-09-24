# Databricks notebook source
# MAGIC %md
# MAGIC # Model Selection
# MAGIC 
# MAGIC Building machine learning solutions involves testing a number of different models.  This lesson explores tuning hyperparameters and cross-validation in order to select the optimal model as well as saving models and predictions.
# MAGIC 
# MAGIC ### Agenda:
# MAGIC * Define hyperparameters and motivate their role in machine learning
# MAGIC * Tune hyperparameters using grid search
# MAGIC * Validate model performance using cross-validation
# MAGIC * Save a trained model

# COMMAND ----------

# MAGIC %run ./Classroom-Setup

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Tuning, Validating and Saving
# MAGIC 
# MAGIC In earlier lessons, we addressed the methodological mistake of training _and_ evaluating a model on the same data.  This leads to **overfitting,** where the model performs well on data it has already seen but fails to predict anything useful on data it has not already seen.  To solve this, we proposed the train/test split where we divided our dataset between a training set used to train the model and a test set used to evaluate the model's performance on unseen data.  In this lesson, we will explore a more rigorous solution to problem of overfitting.
# MAGIC 
# MAGIC A **hyperparameter** is a parameter used in a machine learning algorithm that is set before the learning process begins.  In other words, a machine learning algorithm cannot learn hyperparameters from the data itself.  Hyperparameters need to be tested and validated by training multiple models.  Common hyperparameters include the number of iterations and the complexity of the model.  **Hyperparameter tuning** is the process of choosing the hyperparameter that performs the best on our loss function, or the way we penalize an algorithm for being wrong.
# MAGIC 
# MAGIC If we were to train a number of different models with different hyperparameters and then evaluate their performance on the test set, we would still risk overfitting because we might choose the hyperparameter that just so happens to perform the best on the data we have in our dataset.  To solve this, we can use _k_ subsets of our training set to train our model, a process called **_k_-fold cross-validation.** 
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-1/cross-validation.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC In this lesson, we will divide our dataset into _k_ "folds" in order to choose the best hyperparameters for our machine learning model.  We will then save the trained model and its predictions.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Hyperparameter Tuning
# MAGIC 
# MAGIC Hyperparameter tuning is the process of of choosing the optimal hyperparameters for a machine learning algorithm.  Each algorithm has different hyperparameters to tune.  You can explore these hyperparameters by using the `.explainParams()` method on a model.
# MAGIC 
# MAGIC **Grid search** is the process of exhaustively trying every combination of hyperparameters.  It takes all of the values we want to test and combines them in every possible way so that we test them using cross-validation.
# MAGIC 
# MAGIC Start by performing a train/test split on the Boston dataset and building a pipeline for linear regression.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See <a href="https://en.wikipedia.org/wiki/Hyperparameter_optimization" target="_blank">the Wikipedia article on hyperparameter optimization</a> for more information.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

titanicDF = spark.read.table("titanic_clean")

trainDF, testDF = titanicDF.randomSplit([0.8, 0.2], seed=10)

assembler = VectorAssembler(inputCols=titanicDF.columns[1:], outputCol="features")
dtc = DecisionTreeClassifier(featuresCol="features", labelCol="Survived")

pipeline = Pipeline(stages = [assembler, dtc])

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the model parameters using the `.explainParams()` method.

# COMMAND ----------

print(dtc.explainParams())

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC `ParamGridBuilder()` allows us to string together all of the different possible hyperparameters we would like to test.  In this case, we can test the maximum number of iterations, whether we want to use an intercept with the y axis, and whether we want to standardize our features.
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Since grid search works through exhaustively building a model for each combination of parameters, it quickly becomes a lot of different unique combinations of parameters.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (ParamGridBuilder()
  .addGrid(dtc.maxDepth, [2, 3, 4, 5, 6])
  .build()
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now `paramGrid` contains all of the combinations we will test in the next step.  Take a look at what it contains.

# COMMAND ----------

paramGrid

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Cross-Validation
# MAGIC 
# MAGIC There are a number of different ways of conducting cross-validation, allowing us to trade off between computational expense and model performance.  An exhaustive approach to cross-validation would include every possible split of the training set.  More commonly, _k_-fold cross-validation is used where the training dataset is divided into _k_ smaller sets, or folds.  A model is then trained on _k_-1 folds of the training data and the last fold is used to evaluate its performance.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See <a href="https://en.wikipedia.org/wiki/Cross-validation_(statistics)" target="_blank">the Wikipedia article on Cross-Validation</a> for more information.

# COMMAND ----------

# MAGIC %md
# MAGIC Create a `MulticlassClassificationEvaluator()` to evaluate our grid search experiments and a `CrossValidator()` to build our models.

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="Survived", metricName="accuracy")

cv = CrossValidator(
  estimator = pipeline,             # Estimator (individual model or pipeline)
  estimatorParamMaps = paramGrid,   # Grid of parameters to try (grid search)
  evaluator=evaluator,              # Evaluator
  numFolds = 3,                     # Set k to 3
  seed = 11                         # Seed to sure our results are the same if ran again
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Fit the `CrossValidator()`
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This will train a large number of models.  If your cluster size is too small, it could take a while.

# COMMAND ----------

cvModel = cv.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Take a look at the scores from the different experiments
# MAGIC This can be done via the [MLflow](https://databricks.com/blog/2018/06/05/introducing-mlflow-an-open-source-machine-learning-platform.html) sidebar/UI, or with the code in the next cell.<br><br>
# MAGIC ![x](https://docs.databricks.com/_images/mlflow-notebook-experiments.gif)

# COMMAND ----------

for params, score in zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics):
  print("".join([param.name+"\t"+str(params[param])+"\t" for param in params]))
  print("\tScore: {}".format(score))

# COMMAND ----------

# MAGIC %md
# MAGIC You can then access the best model using the `.bestModel` attribute.

# COMMAND ----------

bestModel = cvModel.bestModel
bestModel.stages[-1]    # decision tree model details, also can use .explainParams()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving Models
# MAGIC 
# MAGIC Spark can save both the trained model we created as well as the predictions.  For online predictions such as on a stream of new data, saving the trained model and using it with Spark Streaming is a common application.

# COMMAND ----------

# MAGIC %md
# MAGIC Save the best model.

# COMMAND ----------

modelPath = "/titanic/cvPipelineModel"
dbutils.fs.rm(modelPath, recurse=True)

cvModel.bestModel.save(modelPath)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at where it saved.

# COMMAND ----------

dbutils.fs.ls(modelPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC [Model Serving]($./06-Model-Serving)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2019 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>