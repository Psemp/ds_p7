# Deploy a scoring model - modelisation repo

# 1 : Data :
- The data originates from [this kaggle competition](https://www.kaggle.com/c/home-credit-default-risk/data).
- Preprocessing is derived from [@JS Aiguar](https://www.kaggle.com/jsaguiar) preprocessing from this competition. The source code can be found in the `source.txt` file of this repo or [**here**](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script)
- Some additionnal preprocessing steps have been taken, some parts of it modified iteratively to fit the problem. The process is detailled as much as possible using docstrings in the file `scripts/preprocessing.py`

# 2 : The class imbalance issue :
- There is a factor ~11 imbalance (11 times more clients that have no reported problem repaying their debts). Even using mostly Ensemble methods and DNNs, the problem needs to be addressed.
- Attempted but not kept : loop over the dataset with many successive different algorithms execution (TL, OSS, NCR, ENN, details on imblearn doc.). It smoothes the data after several iterations and likely oversimplifies the model.
- Near Miss (version 1) was retained as it is a good balance between speed, information and class harmony. It detects potential misclassifiable instances in the data and eliminates them.

# 3 : Modelisation :
3 models have been tested and optimized :

- xGboost classifier
- Catboost classifier
- Keras DNN (Input, 512, 30%, 256, 20%, Output (sigmoid))

# 4 : Logging :
*MLFow is used to keep track of the models, it is configured to track :* <br>

- The estimator parameters for reproductibility
- The **test** AUC and confusion matrix (test AUC on xGboost is high but further tests on hold out hints that the model can ideed generalize)
- SHAP values for Feature importance
- columns used for training
- SHAP plot
- diverse metrics including the custom scorer and its threshold found via CV (+ F1, recall, accuracy and AUC)

# 5 : Evidently for data drift :

- Evidently is used to calculate the datadrift under the assumption that training set is known data and application_test is current data.
- Cross checking between high importance features and highly drifted variables shows that the data collection might be perfectible but doesnt cross the 50% drift threshold - constant monitoring is advised however.
- The two reports as HTML are showed in a separate folder
