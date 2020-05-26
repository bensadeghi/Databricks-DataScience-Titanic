# Databricks notebook source
# MAGIC %md
# MAGIC # Model Selection
# MAGIC 
# MAGIC Building machine learning solutions involves testing a number of different models.  This lesson explores tuning hyperparameters and cross-validation in order to select the optimal model as well as saving models and predictions.
# MAGIC 
# MAGIC ### Agenda:
# MAGIC * Use MLflow to manage the model lifecycle
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

titanicDF = spark.read.table("titanic_clean").cache()

trainDF, testDF = titanicDF.randomSplit([0.8, 0.2], seed=1)

assembler = VectorAssembler(inputCols=titanicDF.columns[1:], outputCol="features")
dtc = DecisionTreeClassifier(featuresCol="features", labelCol="Survived")

pipeline = Pipeline(stages = [assembler, dtc])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC `ParamGridBuilder()` allows us to string together all of the different possible hyperparameters we would like to test.  In this case, we can test the maximum number of iterations, whether we want to use an intercept with the y axis, and whether we want to standardize our features.
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Since grid search works through exhaustively building a model for each combination of parameters, it quickly becomes a lot of different unique combinations of parameters.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (ParamGridBuilder()
  .addGrid(dtc.maxDepth, [2, 3, 4, 5, 6])
  .addGrid(dtc.maxBins, [10, 25, 50, 75])
  .build()
)

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
  evaluator = evaluator,              # Evaluator
  numFolds = 5,                     # Set k to 5
  seed = 10                         # Seed to sure our results are the same if ran again
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
# MAGIC This can be done via the [MLflow](https://databricks.com/blog/2018/06/05/introducing-mlflow-an-open-source-machine-learning-platform.html) sidebar/UI<br><br>
# MAGIC ![x](https://docs.databricks.com/_images/mlflow-notebook-experiments.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC You can then access the best model using the `.bestModel` attribute.

# COMMAND ----------

bestModel = cvModel.bestModel.stages[-1]
print(bestModel)

# get the best value for maxDepth parameter
bestDepth = bestModel.getOrDefault("maxDepth")
bestBins = bestModel.getOrDefault("maxBins")

# COMMAND ----------

# MAGIC %md
# MAGIC Build final model using the entire training dataset and evaluate its performance using the test set
# MAGIC 
# MAGIC Log parameters, metrics, and the model iteself in MLflow

# COMMAND ----------

import mlflow.spark

with mlflow.start_run(run_name="final_model") as run:
  runID = run.info.run_uuid
  
  # train model
  dtc = DecisionTreeClassifier(featuresCol="features", labelCol="Survived", maxDepth=bestDepth, maxBins=bestBins)
  pipeline = Pipeline(stages = [assembler, dtc])
  finalModel = pipeline.fit(trainDF)
  
  # log parameters and model
  mlflow.log_param("maxDepth", bestDepth)
  mlflow.log_param("maxBins", bestBins)
  mlflow.spark.log_model(finalModel, "model")
  
  # generate and log metrics
  testPredictionDF = finalModel.transform(testDF)
  accuracy = evaluator.evaluate(testPredictionDF)
  mlflow.log_metric("accuracy", accuracy)
  print("Accuracy on the test set for the decision tree model: {}".format(accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Model
# MAGIC 
# MAGIC #### Create a new registered model using the API
# MAGIC 
# MAGIC The following cells use the `mlflow.register_model()` function to create a new registered model whose name begins with the string `Titanic-DecisionTree`. This also creates a new model version (e.g., `Version 1` of `Titanic-Model`).

# COMMAND ----------

modelName = "Titanic-Model__" + userName

artifactPath = "model"
modelURI = "runs:/{run_id}/{artifact_path}".format(run_id=runID, artifact_path=artifactPath)

modelDetails = mlflow.register_model(model_uri=modelURI, name=modelName)

# COMMAND ----------

# MAGIC %md After creating a model version, it may take a short period of time to become ready. Certain operations, such as model stage transitions, require the model to be in the `READY` state. Other operations, such as adding a description or fetching model details, can be performed before the model version is ready (e.g., while it is in the `PENDING_REGISTRATION` state).
# MAGIC 
# MAGIC The following cell uses the `MlflowClient.get_model_version()` function to wait until the model is ready.

# COMMAND ----------

import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

def wait_until_ready(model_name, model_version):
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)
  
wait_until_ready(modelDetails.name, modelDetails.version)

# COMMAND ----------

# MAGIC %md ### Perform a model stage transition
# MAGIC 
# MAGIC The MLflow Model Registry defines several model stages: **None**, **Staging**, **Production**, and **Archived**. Each stage has a unique meaning. For example, **Staging** is meant for model testing, while **Production** is for models that have completed the testing or review processes and have been deployed to applications.

# COMMAND ----------

client.transition_model_version_stage(
  name = modelDetails.name,
  version = modelDetails.version,
  stage='Production',
)

# COMMAND ----------

# MAGIC %md The MLflow Model Registry allows multiple model versions to share the same stage. When referencing a model by stage, the Model Registry will use the latest model version (the model version with the largest version ID). The `MlflowClient.get_latest_versions()` function fetches the latest model version for a given stage or set of stages. The following cell uses this function to print the latest version of the power forecasting model that is in the `Production` stage.

# COMMAND ----------

latestVersionInfo = client.get_latest_versions(modelName, stages=["Production"])
latestVersion = latestVersionInfo[0].version
print("The latest production version of the model '%s' is '%s'." % (modelName, latestVersion))

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