import json
import mlflow
import time
import shap
import os
import pickle

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score

from models.scorer import home_credit_scorer
from scripts.patch_shap import patched_beeswarm

load_dotenv()


def plot_auc_conf(estimator, X_test, y_test, display: bool = False):
    """
    Function :
    Calculates the area under the roc curve, returns always the auroc value and
    optionnaly the figure containing both the AUROC and the confusion matrix.

    Args :
    - estimator : classfier previously fitted on training data
    - X_test : test data matrix
    - y_test : test labels
    - display : bool, displays or not the figure (false : return plt.gcf())

    Returns :
    - auroc : area under the roc curve
    - None or plt.gcf()
    """

    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), dpi=150)

    # AUROC
    y_pred_proba = estimator.predict_proba(X_test)[:, 1]

    # false positive rate, true positive rate, AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auroc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color="#000331", lw=1.5, label=f"ROC curve (AUC = {auroc})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1, linestyle=":")

    # CONF
    y_pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax2)

    ###
    # Titles/Lables
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix")
    #
    ###

    if not display:
        fig = plt.gcf()
        plt.close()
        return auroc, fig
    elif display:
        plt.show()
        return auroc, None


def plot_auc_conf_keras(model, X_test, y_test, display: bool = False):
    """
    Function:
    Calculates the area under the roc curve, returns always the auroc value and
    optionnaly the figure containing both the AUROC and the confusion matrix.
    Specific to Keras (although similar to the ensemble estimator, it might be preferable to keep
    the two functions as separate)

    Args:
    - model : a TensorFlow model object
    - X_test : test data matrix
    - y_test : test labels
    - display : bool, displays or not the figure (false : return plt.gcf())

    Returns:
    - auroc : area under the roc curve
    - None or plt.gcf()
    """

    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), dpi=150)

    y_pred_proba = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auroc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color="#000331", lw=1.5, label=f"ROC curve (AUC = {auroc})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1, linestyle=":")

    # CONF
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax2)

    ###
    # Titles/Lables
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix")
    #
    ###

    if not display:
        fig = plt.gcf()
        plt.close()
        return auroc, fig
    elif display:
        plt.show()
        return auroc, None


def plot_epochs(history):
    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(8, 4),
        dpi=150
        )

    plt.ioff()

    training_recall = history.history["recall"]
    validation_recall = history.history["val_recall"]
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    ax1.plot(training_recall, label="training recall")
    ax1.plot(validation_recall, label="validation recall")

    ax2.plot(training_loss, label="training loss")
    ax2.plot(validation_loss, label="validation loss")

    ###
    # Titles/Lables
    ax1.legend()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Recall")
    ax2.legend()
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    fig.suptitle("Evolution of Training and Validation accuracy per epoch")
    #
    ###

    return plt.gcf()


def describe_run(
        template_path: str, model_name: str, data_version: str,
        imb_learn_method: str, column_drop_na_threshold: float
        ) -> str:
    """
    Function :
    Formats the template passed as argument and returns a description to be added to the mlrun

    Args :
    - template_path : path to the json template
    - model_name : the name of the model
    - data_version : the version of the dataset used
    - imb_learn_method : the method used to solve class imbalance, if any
    - column_drop_na_threshold : the threshold used to delete variables if na ratio > thresh

    Returns :
    - description : the description ready to be passed to mlflow to describe the run, as str
    """

    with open(template_path, "r") as template:
        template = json.load(template)

    if column_drop_na_threshold > 0:
        na_threshold_text = f"Columns with a NaN count above `{column_drop_na_threshold * 100}%` were dropped."
    else:
        na_threshold_text = ""

    description = template["description"].format(
        model_name=model_name,
        data_version=data_version,
        imb_learn_method=imb_learn_method,
        column_drop_na_threshold=column_drop_na_threshold,
        na_threshold_text=na_threshold_text
    )

    return description


def get_shap_waterfall(clf, X_train):
    """
    Takes a tree based classifier and the X_train of the classifier,
    returns the waterfall and summary plots of shap

    Args:
    - clf, a tree based classifier
    - X_train, the data used to train the clf

    Returns :
    - shap summary matplotlib figure
    """
    plt.ioff()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_train)

    waterfall_fig = shap.plots.waterfall(shap_values[0], max_display=20, show=False)
    waterfall_fig.figure.set_size_inches(8, 8)
    waterfall_fig.figure.set_dpi(150)

    waterfall_fig.suptitle("Impact des differentes variables sur le modele, classées par ordre d'importance")
    return waterfall_fig


def get_shap_summary(clf, X_train):
    """
    Takes a tree based classifier and the X_train of the classifier,
    returns the waterfall and summary plots of shap

    Args:
    - clf, a tree based classifier
    - X_train, the data used to train the clf

    Returns :
    - shap summary matplotlib figure
    """

    plt.ioff()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_train)

    summary_fig = patched_beeswarm(shap_values, max_display=20, plot_size=(8, 8), color_bar=False)
    summary_fig.set_dpi(150)
    summary_fig.tight_layout()
    summary_fig.suptitle("Blue -> Red = Low -> High")
    return summary_fig


def get_shap_summary_keras(model, X_train, X_test, display: bool = None):
    """
    Takes a tf classifier DNN  and the X_train and X_test of the classifier,
    returns the summary plots of shap values ranked by absolute importance (lim 20).
    Since it takes quite a long time to compute, returns the shap_values to use with
    waterfall or other.

    Args:
    - model, a tensorflow dnn
    - X_train, the data used to train the clf

    Returns :
    - shap summary matplotlib figure (or none if display)
    - shap_values : the shap explainer of the values to use in waterfall for local use
    """

    # bg dataset
    background = X_train.sample(n=2500, random_state=123)

    explainer = shap.DeepExplainer(model, background.values)
    shap_values = explainer.shap_values(X_test.values)
    # Shap mean values :
    mean_shap_values = np.mean(shap_values[0], axis=0)

    # Ranking the values by absolute value (thanks numpy)
    sorted_idx = np.argsort(np.abs(mean_shap_values))[::-1]
    sorted_features = X_train.columns[sorted_idx]
    sorted_shap_values = mean_shap_values[sorted_idx]

    plt.ioff()

    fig, ax1 = plt.subplots(
        figsize=(8, 8),
        dpi=155
        )

    colors = ["grey" if val < 0 else "#000331" for val in mean_shap_values]

    ax1.barh(
        sorted_features[:20],
        sorted_shap_values[:20],
        align="center",
        color=colors
        )

    ###
    # Titles/Lables
    ax1.set_ylim(-1, len(X_train.columns[:20]))
    ax1.set_xlabel("SHAP Value, impact on model")
    ax1.set_ylabel("Feature name")
    ax1.set_title("Global DNN Feature Importance")
    ax1.invert_yaxis()  # Most important at the top
    #
    ###

    if display:
        plt.show()
        return shap_values
    elif not display:
        summary_fig = plt.gcf()
        return summary_fig, shap_values


def train_and_log(
        estimator, X_train, X_test, y_train, y_test, model_name="classifier",
        dataset_version="default", imb_method="None", na_thresh=0, params: dict = None,
        ):
    """
    Trains and predict the model based on X_train/test and y_train/test + classifier
    Logs the model name (default is classifier) using mlflow as well as the params and
    the metrics (accuracy, f1, recall) using mlflow. Handles description creation with json template

    Args :
    - estimator : classifier (supported are : sklearn, xgboost and catboost)
    - X_train : the training data
    - X_test : the test data
    - y_train : the training labels
    - y_test : the test labels
    - model_name : default = classifier, the name of the model as will appear on mlflow's logs
    - id_dict : dictionnary of ids to log as train/test split for replication, optionnal
    - dataset_version : default = default, the version of the dataset
    - imb_method : default = None, the class imbalance method used
    - na_thresh : default = 0, float between 0 and 1, informs that variables
    containing more than this thresh have been dropped
    - params : the parameters to pass to the estimator

    Returns :
    - metrics : dictionnary of the evaluated metrics
    - model : the classifier fitted on X_train/y_train and params
    """

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    description = describe_run(
        template_path="../templates/description_mlflow.json",
        model_name=model_name,
        data_version=dataset_version,
        imb_learn_method=imb_method,
        column_drop_na_threshold=na_thresh,
        )

    with mlflow.start_run(run_name=model_name, description=description):
        start = time.perf_counter()
        mlflow.log_params(params)

        if estimator.__module__.startswith("xgboost"):
            classifier = estimator(**params, eval_metric=home_credit_scorer)
        elif estimator.__module__.startswith("lightgbm") or \
                estimator.__module__.startswith("catboost") or estimator.__module__.startswith("sklearn"):
            classifier = estimator(**params)
        classifier.fit(X=X_train, y=y_train)

        # Metrics :
        home_credit_score = home_credit_scorer(estimator=classifier, X=X_test, y_true=y_test)
        accuracy = accuracy_score(y_test, classifier.predict(X_test))
        f1 = f1_score(y_test, classifier.predict(X_test), average="macro")
        recall = recall_score(y_test, classifier.predict(X_test), average="macro")
        auroc, auroc_conf_fig = plot_auc_conf(estimator=classifier, X_test=X_test, y_test=y_test, display=False)

        metrics = {
            "home_credit_score": home_credit_score, "accuracy": accuracy,
            "f1": f1, "recall": recall, "auroc": auroc
            }

        # MLFlow log :
        mlflow.log_metrics(metrics)

        mlflow.log_figure(auroc_conf_fig, "AUROC_Conf_matrix.png")

        # waterfall = get_shap_waterfall(clf=classifier, X_train=X_train)
        # mlflow.log_figure(waterfall, "waterfall_shap.png")

        summary = get_shap_summary(clf=classifier, X_train=X_train)
        mlflow.log_figure(summary, "summary_shap.png")

        model_type = type(classifier)

        if model_type.__module__.startswith("sklearn"):
            artifact_path = "sklearn-model"
            mlflow.sklearn.log_model(classifier, artifact_path=artifact_path, registered_model_name=model_name)
        elif model_type.__module__.startswith("xgboost"):
            artifact_path = "xGboost-model"
            mlflow.xgboost.log_model(classifier, artifact_path=artifact_path, registered_model_name=model_name)
        elif model_type.__module__.startswith("catboost"):
            artifact_path = "catboost-model"
            mlflow.catboost.log_model(classifier, artifact_path=artifact_path, registered_model_name=model_name)
        elif model_type.__module__.startswith("lightgbm"):
            artifact_path = "lightgbm-model"
            mlflow.lightgbm.log_model(classifier, artifact_path=artifact_path, registered_model_name=model_name)

        run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/{artifact_path}"

        end = time.perf_counter()

        mlflow.log_param(key="processing time", value=end - start)

        mlflow.register_model(model_uri=model_uri, name=model_name)

        return metrics, classifier


def train_and_log_keras(
        model, X_train, X_test, y_train, y_test, model_summary, training_time: float,
        history, home_credit_score, feature_list, model_name="dnn_classifier",
        dataset_version="default", imb_method="None", na_thresh=0,
        ):
    """
    Trains and predict the model based on X_train/test and y_train/test + classifier (keras DNN)
    Logs the model name (default is dnn_classifier) using mlflow as well as the params and
    the metrics (accuracy, f1, recall) using mlflow. Handles description creation with json template

    Args :
    - model : keras model, fitted
    - X_train : the training data
    - X_test : the test data
    - y_train : the training labels
    - y_test : the test labels
    - training_time : the time to train the model
    - model_name : default = classifier, the name of the model as will appear on mlflow's logs
    - dataset_version : default = default, the version of the dataset
    - imb_method : default = None, the class imbalance method used
    - na_thresh : default = 0, float between 0 and 1, informs that variables
    containing more than this thresh have been dropped
    - feature list : the list of features used by the model

    Returns :
    - metrics : dictionnary of the evaluated metrics
    - model : the classifier fitted on X_train/y_train and params
    """

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    description = describe_run(
        template_path="../templates/description_mlflow.json",
        model_name=model_name,
        data_version=dataset_version,
        imb_learn_method=imb_method,
        column_drop_na_threshold=na_thresh,
        )

    with mlflow.start_run(run_name=model_name, description=description):

        # Metrics :
        y_pred = np.round(model.predict(X_test))

        home_credit_score = home_credit_score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        auroc, auroc_conf_fig = plot_auc_conf_keras(model=model, X_test=X_test, y_test=y_test, display=False)

        metrics = {
            "home_credit_score": home_credit_score, "accuracy": accuracy,
            "f1": f1, "recall": recall, "auroc": auroc
            }

        # MLFlow log :
        mlflow.log_metrics(metrics)

        with open("model_summary.txt", "w") as f:
            f.write(model_summary)

        # log model summary file as an artifact in MLFlow

        serialized_list = pickle.dumps(feature_list)
        with open("feature_list.pkl", "wb") as f:
            f.write(serialized_list)

        mlflow.log_artifact("feature_list.pkl")
        mlflow.log_artifact("model_summary.txt")

        mlflow.log_figure(auroc_conf_fig, "AUROC_Conf_matrix.png")

        epoch_fig = plot_epochs(history=history)

        mlflow.log_figure(epoch_fig, "epoch_eval_fig.png")

        # waterfall = get_shap_waterfall(clf=classifier, X_train=X_train)
        # mlflow.log_figure(waterfall, "waterfall_shap.png")

        summary_fig, shap_values = get_shap_summary_keras(
            model=model, X_train=X_train,
            X_test=X_test,
            display=False
            )

        summary_fig.tight_layout()

        mlflow.log_figure(summary_fig, "summary_shap.png")

        model_type = type(model)

        if model_type.__module__.startswith("keras"):
            artifact_path = "keras-model"
            mlflow.tensorflow.log_model(model=model, artifact_path=artifact_path, registered_model_name=model_name)

        else:
            pass

        run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/{artifact_path}"
        shap_file = "shap_values.npy"

        np.save(shap_file, shap_values)

        mlflow.log_artifact(shap_file, artifact_path="shap_values")

        mlflow.log_param(key="processing time", value=training_time)

        mlflow.register_model(model_uri=model_uri, name=model_name)

        return metrics, model
