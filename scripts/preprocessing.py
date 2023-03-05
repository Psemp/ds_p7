###
# Most of the code is derived from https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script
# Some variables and functionnalities have been reworked for optimization (generalization, bins, encoding)
# Accent was made on modularity and clarity
# In this use case, it is expected to pass the complete application dataset to "preprocess_application" and split
# train/test after (i.e. : test has target = np.nan, train has 1 && 0)
###

import warnings
import time

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer


def is_binary(dataframe: pd.DataFrame, column: str) -> bool:
    """
    Args:
    - dataframe : dataframe which column is analyzed
    - column : the column to analyze

    Function:
    - Checks if there are only two values in a given dataframe's column

    Returns:
    True
    if only two values (beside NaN) are found, else return False

    """

    if dataframe[column].dropna().unique().__len__() == 2:
        return True
    else:
        return False


def get_binary_column(dataframe: pd.DataFrame) -> list:
    """
    Args:
    - dataframe : the pandas dataframe object to analyze

    Function:
    - finds and lists the binary columns in the dataframe

    Returns:
    - list of binary columns
    """

    binary_cols = []
    columns = dataframe.columns
    for col in columns:
        if is_binary(dataframe=dataframe, column=col):
            binary_cols.append(col)

    return binary_cols


def one_hot_encode(
        dataframe: pd.DataFrame, to_encode: list,
        prefix: str = None, keep_na: bool = True,
        drop_original: bool = False,
        ) -> None:
    """
    Function :
    Uses one hot encoding on categorical variables (pandas.get_dummies). Applies it dataframe wide
    on a list of columns

    Args:
    - dataframe : the dataframe to encode
    - to_encode : list of the columns to encode, must be of dtype `object` or `category`
    - prefix : override default prefix, default = None : original column name will be kept if len(to_encode) > 1
    - keep_na : if `True`, creates a column for NaNs, default is set to `True` (not pandas default option !)
    - drop_original : Whether to keep the original columns (`to_encode`) or to drop them, default = `False`

    Returns:
    - None
    """
    if not isinstance(to_encode, list):
        raise TypeError(f"to_encode must be of type list, not type {type(to_encode)}")

    if len(to_encode) < 1:
        raise Exception("Warning : to_encode is an empty list")

    for col in to_encode:
        if dataframe[col].dtype != "object" and dataframe[col].dtype != "category":
            raise TypeError(f"Column of type {dataframe[col].dtype} not expected. Adjust.")

    encoded = pd.get_dummies(dataframe[to_encode], prefix=prefix, dummy_na=keep_na)
    dataframe[encoded.columns] = encoded
    if drop_original:
        dataframe.drop(columns=to_encode, inplace=True)


def get_cols_missing_thresh(dataframe: pd.DataFrame, threshold: float) -> list:
    """
    Returns the columns of a dataframe which contain more than threshold (0->1 float) missing values
    """
    if threshold >= 1 or threshold <= 0:
        raise Exception("Threshold must be greater than 0 and smaller than 1")

    drop_list = []
    dataframe_length = len(dataframe)

    for column in dataframe.columns:
        if dataframe[column].isna().sum() / dataframe_length >= threshold:
            drop_list.append(column)

    return drop_list


def variable_creation_application(dataframe: pd.DataFrame) -> None:
    """
    Creates new variables as suggested by the kaggle kernel (quoted as source)

    Args :

    - dataframe : the pandas dataframe of train+test
    """
    dataframe["DAYS_EMPLOYED_PERC"] = dataframe["DAYS_EMPLOYED"] / dataframe["DAYS_BIRTH"]
    dataframe["INCOME_PER_PERSON"] = dataframe["AMT_INCOME_TOTAL"] / dataframe["CNT_FAM_MEMBERS"]
    dataframe["PAYMENT_RATE"] = dataframe["AMT_ANNUITY"] / dataframe["AMT_CREDIT"]


def calc_sentinel(col, range_small=10_000):
    """
    Function :
    Calculates the sentinel value of a column for basic nan imputation,
    A small range of values will mean a sentinel of -1, a large range will
    mean a larger sentinel (-999), if the column already contains a negative value,
    no sentinel will be created.

    Args:
    col : the column to evaluate with apply
    range_small : the range above which the large sentinel will be preferred

    Returns :
    sentinel : value for column imputation
    """

    if col.min() < 0:
        return np.nan

    values_range = col.dropna().max() - col.dropna().min()
    if values_range <= range_small:
        sentinel = -1
    else:
        sentinel = -999
    return sentinel


def calc_sentinel_neg(col, range_small=10_000):
    """
    Function:
    Calculates the sentinel value of a column for basic NaN imputation.
    A small range of negative values will mean a sentinel of -1,
    a large range will mean a larger sentinel (-999).
    If all values in the column are not negative, no sentinel will be created.

    Args:
    - col: the column to evaluate with apply
    - range_small: the range above which the large sentinel will be preferred

    Returns:
    - sentinel: value for column imputation
    """

    # Check if all values in the column are negative
    if (col.dropna() < 0).all():
        values_range = abs(col[col < 0].max() - col[col < 0].min())
        if values_range <= range_small:
            sentinel = 1
        else:
            sentinel = 999
    else:
        sentinel = np.nan

    return sentinel


def impute_knn_parallel(dataframe, col):
    knn_imputer = KNNImputer(n_neighbors=5)
    col_imputed = knn_imputer.fit_transform(dataframe[[col]])
    return col_imputed


def preprocess_application(dataframe: pd.DataFrame, handle_na=True) -> pd.DataFrame:
    """
    Args : dataframe of the application (train and/or test)

    Function :
    Preprocesses the dataset as a pandas dataframe, steps :
    - Finds and factorizes binary columns
    - Uses one hot encoding to encode categorical (non-bin) variables
    - Creates variables (cf. function variable_creation_application)
    - Replaces 365243 by np.nan for days
    - can impute nans, method in dictionnary
    Args:
    - handle_na : method to handle na values, true by default

    Returns :
    - preprocessed dataframe
    """

    application_binary = get_binary_column(dataframe=dataframe)

    # Only two contracts :
    dataframe["IS_CASH_LOAN"] = dataframe["NAME_CONTRACT_TYPE"].map({"Cash loans": 1, "Revolving loans": 0})
    dataframe.drop(columns=["NAME_CONTRACT_TYPE"], inplace=True)

    application_binary.remove("NAME_CONTRACT_TYPE")

    # Essential columns
    subset_na = ["AMT_ANNUITY", "AMT_GOODS_PRICE"]
    dataframe.dropna(subset=subset_na, inplace=True)

    for col in application_binary:
        if dataframe[col].dtype != int:
            values, _ = dataframe[col].factorize()
            dataframe[col] = values
        if dataframe[col].dtype != int:
            dataframe[col] = dataframe[col].astype(int)

    application_object = dataframe.select_dtypes(include="object").columns.tolist()

    one_hot_encode(
        dataframe=dataframe,
        to_encode=application_object,
        prefix=None,
        keep_na=True,
        drop_original=True
        )

    dataframe["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    if handle_na:
        impute_methods = {
            "DAYS_EMPLOYED": "median",
            "FLOORSMAX_AVG": "mean",
            "LIVINGAREA_AVG": "mean",
            "FLOORSMAX_MODE": "mean",
            "LIVINGAREA_MODE": "mean",
            "FLOORSMAX_MEDI": "median",
            "LIVINGAREA_MEDI": "median",
            "TOTALAREA_MODE": "median",
            "OBS_30_CNT_SOCIAL_CIRCLE": "mean",
            "DEF_30_CNT_SOCIAL_CIRCLE": "mean",
            "OBS_60_CNT_SOCIAL_CIRCLE": "mean",
            "DEF_60_CNT_SOCIAL_CIRCLE": "mean",
        }

        print("Imputation ...")
        for col, impute_method in impute_methods.items():
            if impute_method == "mean":
                dataframe[col].fillna(dataframe[col].mean(), inplace=True)
            elif impute_method == "median":
                dataframe[col].fillna(dataframe[col].median(), inplace=True)

    variable_creation_application(dataframe=dataframe)

    return dataframe


def preprocess_bureau(bureau_path: str) -> pd.DataFrame:
    """
    Args :
    - bureau_path : string, the path to bureau.csv

    Function :
    - encodes categorical variables
    - aggregates numerical values according to dict of functions
    - aggregates and returns a dataframe grouped by "SK_ID_CURR"

    Returns :

    - df_bureau_agg : pandas dataframe of aggregated and grouped variables
    """

    df_bureau = pd.read_csv(filepath_or_buffer=bureau_path)

    # Finding and encoding categorical columns :
    bureau_object = df_bureau.select_dtypes(include="object").columns
    one_hot_encode(
        dataframe=df_bureau,
        to_encode=bureau_object.tolist(),
        prefix=None,
        keep_na=True,
        drop_original=True
    )

    # Columns of bureau.csv to aggreagate with associated functions :
    bureau_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_ANNUITY": ["max", "mean", "sum"],
        "AMT_CREDIT_MAX_OVERDUE": ["sum"],
        "AMT_CREDIT_SUM": ["mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
    }

    # Adding categorical aggregations (mean value) :
    # any col that startwith object_col has been encoded and removed
    encoded_cols = [col for col in df_bureau.columns if any(col.startswith(obj_col) for obj_col in bureau_object)]

    for col in encoded_cols:
        bureau_aggregations[col] = ["mean"]

    df_bureau_agg = df_bureau.groupby("SK_ID_CURR").agg(bureau_aggregations)
    df_bureau_agg.columns = pd.Index([f"{col[0]}_{col[1]}" for col in df_bureau_agg.columns.tolist()])

    return df_bureau_agg


def preprocess_credit_card(credit_card_path: str) -> pd.DataFrame:
    """
    Args :
    - credit_card_path : string, the path to credit_card_balance.csv

    Function :
    - encodes categorical variables
    - aggregates numerical values according to dict of functions
    - aggregates and returns a dataframe grouped by "SK_ID_CURR"

    Returns :

    - df_credit_card_agg : pandas dataframe of aggregated and grouped variables
    """

    df_credit_card = pd.read_csv(filepath_or_buffer=credit_card_path)

    df_credit_card.drop(["SK_ID_PREV"], axis=1, inplace=True)

    # Finding and encoding categorical columns :
    credit_card_object = df_credit_card.select_dtypes(include="object").columns

    one_hot_encode(
        dataframe=df_credit_card,
        to_encode=credit_card_object.tolist(),
        prefix=None,
        keep_na=True,
        drop_original=True
    )

    # columns to aggregate, mostly using sum in case of multiple cards by cx
    credit_card_aggregations = {
        "AMT_BALANCE": ["mean", "sum"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "sum"],
        "AMT_DRAWINGS_ATM_CURRENT": ["mean", "sum"],
        "AMT_DRAWINGS_CURRENT": ["mean", "sum"],
        "AMT_DRAWINGS_OTHER_CURRENT": ["sum"],
        "AMT_DRAWINGS_POS_CURRENT": ["sum"],
        "AMT_INST_MIN_REGULARITY": ["mean"],
        "AMT_PAYMENT_CURRENT": ["mean"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["mean"],
        "AMT_RECEIVABLE_PRINCIPAL": ["sum"],
        "AMT_RECIVABLE": ["sum"],
        "AMT_TOTAL_RECEIVABLE": ["sum"],
        "CNT_DRAWINGS_ATM_CURRENT": ["sum"],
        "CNT_DRAWINGS_CURRENT": ["sum"],
        "CNT_DRAWINGS_OTHER_CURRENT": ["sum"],
        "CNT_DRAWINGS_POS_CURRENT": ["sum"],
        "CNT_INSTALMENT_MATURE_CUM": ["sum"],
        "SK_DPD": ["mean", "max"],
        "SK_DPD_DEF": ["mean", "max"]
    }

    # Adding categorical aggregations (mean value) :
    # only one encoded col : NAME_CONTRACT_STATUS
    encoded_cols = [col for col in df_credit_card.columns if col.startswith("NAME_CONTRACT_STATUS_")]

    for col in encoded_cols:
        credit_card_aggregations[col] = ["mean"]

    df_credit_card_agg = df_credit_card.groupby("SK_ID_CURR").agg(credit_card_aggregations)
    df_credit_card_agg.columns = pd.Index([f"{col[0]}_{col[1]}" for col in df_credit_card_agg.columns.tolist()])

    return df_credit_card_agg


def preprocess_pos(pos_path: str) -> pd.DataFrame:
    """
    Args :
    - pos_path : string, the path to POS_CASH_balance.csv

    Function :
    - encodes categorical variables
    - aggregates numerical values according to dict of functions
    - aggregates and returns a dataframe grouped by "SK_ID_CURR"
    - counts the number of accounts posseded by each ID

    Returns :

    - df_pos_agg : pandas dataframe of aggregated and grouped variables
    """

    df_pos = pd.read_csv(filepath_or_buffer=pos_path)

    # Finding and encoding categorical columns :
    pos_object = df_pos.select_dtypes(include="object").columns

    one_hot_encode(
        dataframe=df_pos,
        to_encode=pos_object.tolist(),
        prefix=None,
        keep_na=True,
        drop_original=True
    )

    # columns to aggregate, mostly using sum in case of multiple cards by cx
    pos_aggregations = {
        "MONTHS_BALANCE": ["max", "mean"],
        "SK_DPD": ["max", "mean"],
        "CNT_INSTALMENT_FUTURE": ["sum"],
        "SK_DPD_DEF": ["max", "mean"]
    }

    # Adding categorical aggregations (mean value) :
    encoded_cols = [col for col in df_pos.columns if any(col.startswith(obj_col) for obj_col in pos_object)]
    for col in encoded_cols:
        pos_aggregations[col] = ["mean"]

    df_pos_agg = df_pos.groupby("SK_ID_CURR").agg(pos_aggregations)
    df_pos_agg.columns = pd.Index([f"{col[0]}_{col[1]}" for col in df_pos_agg.columns.tolist()])
    # Count pos cash accounts
    df_pos_agg["number_accounts"] = df_pos.groupby("SK_ID_CURR").size()

    return df_pos_agg


def preprocess_previous_app(previous_app_path: str) -> pd.DataFrame:
    """
    Args :
    - previous_app_path : string, the path to previous_application.csv

    Function :
    - encodes categorical variables
    - aggregates numerical values according to dict of functions
    - aggregates and returns a dataframe grouped by "SK_ID_CURR"
    - counts the number of applications made by each ID

    Returns :

    - df_app_agg : pandas dataframe of aggregated and grouped variables
    """

    df_app = pd.read_csv(filepath_or_buffer=previous_app_path)

    # Finding and encoding categorical columns :
    app_object = df_app.select_dtypes(include="object").columns
    one_hot_encode(
        dataframe=df_app,
        to_encode=app_object.tolist(),
        prefix=None,
        keep_na=True,
        drop_original=True
    )

    # Days columns 365.243 values -> nan
    for col in df_app.columns:
        if col.startswith("DAYS_"):
            df_app[col].replace(365243, np.nan, inplace=True)

    # columns to aggregate, mostly using sum in case of multiple cards by cx
    app_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "DAYS_FIRST_DRAWING": ["mean"],
        "DAYS_FIRST_DUE": ["mean"],
        "DAYS_LAST_DUE_1ST_VERSION": ["mean"],
        "DAYS_LAST_DUE": ["mean"],
        "DAYS_TERMINATION": ["mean"]
    }

    # Adding categorical aggregations (mean value) :
    encoded_cols = [col for col in df_app.columns if any(col.startswith(obj_col) for obj_col in app_object)]

    for col in encoded_cols:
        app_aggregations[col] = ["mean"]

    with warnings.catch_warnings():  # Silence fragmentation warning
        warnings.simplefilter("ignore")

        df_app_agg = df_app.groupby("SK_ID_CURR").agg(app_aggregations)
        df_app_agg.columns = pd.Index([f"{col[0]}_{col[1]}" for col in df_app_agg.columns.tolist()])
        df_app_agg["number_applications"] = df_app.groupby("SK_ID_CURR").size()

    return df_app_agg


def preprocess_installments(installments_path: str) -> pd.DataFrame:
    """
    Args :
    - installments_path : string, the path to installments_payments.csv

    Function :
    - Aggregates most numerical values
    - No ohe required : no categorical variables
    - Counts the number of accounts per ID

    Returns :
    - df_inst_agg
    """

    df_installments = pd.read_csv(filepath_or_buffer=installments_path)

    # Days past due and days before due (no negative values)
    df_installments["DPD"] = df_installments["DAYS_ENTRY_PAYMENT"] - df_installments["DAYS_INSTALMENT"]
    df_installments["DBD"] = df_installments["DAYS_INSTALMENT"] - df_installments["DAYS_ENTRY_PAYMENT"]
    df_installments["DPD"] = df_installments["DPD"].apply(lambda x: x if x > 0 else 0)
    df_installments["DBD"] = df_installments["DBD"].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"]
    }

    df_inst_agg = df_installments.groupby("SK_ID_CURR").agg(aggregations)
    df_inst_agg.columns = pd.Index([f"{col[0]}_{col[1]}" for col in df_inst_agg.columns.tolist()])

    # Count installments accounts
    df_inst_agg["number_accounts"] = df_installments.groupby("SK_ID_CURR").size()

    return df_inst_agg


def prefixer(dataframe: pd.DataFrame, prefix: str, ignore: list = None) -> None:
    """
    Quickly adds a prefix to the columns of a dataframe

    Args:
    - dataframe : the dataframe containing the cols needing a prefix
    - prefix : string, the prefix as prefix_col (_ is included)
    - ignore : list of the columns not to be renamed

    Returns:
    - None
    """

    if ignore is None:
        ignore = []

    columns = [col for col in dataframe.columns if col not in ignore]
    new_columns = [f"{prefix}_{col}" for col in columns]

    # dict mapping original column names to prefixd ones
    rename_dict = dict(zip(columns, new_columns))

    dataframe.rename(columns=rename_dict, inplace=True)


def make_dataset(
        train_path: str,
        test_path: str,
        bureau_path: str,
        credit_card_path: str,
        pos_path: str,
        installments_path: str,
        previous_app_path: str,
        output_path: str,
        output_format: str = "both"
        ) -> None:
    """
    Preprocesses the dataset by :
    - concatenating application_train & application_test
    - preprocessing :
        - bureau.csv
        - credit_card.csv
        - installments_payments.csv
        - POS_CASH_balance.csv
        - previous_application.csv
    - joining the preprocessed datasets on SK_ID_CURR
    - drops columns that contain 50%+ of nans
    - saving the whole dataset as csv, pickle or both
    - saving the datasets with target (train) and without target (test) as well (csv/pkl/both)

    Args:
    - train_path: string, the path to application_train.csv
    - test_path: string, the path to application_test.csv
    - bureau_path: string, the path to bureau.csv
    - credit_card_path: string, the path to credit_card_balance.csv
    - pos_path: string, the path to POS_CASH_balance.csv
    - previous_app_path: string, the path to previous_application.csv
    - output_path: string, the output path (relative or absolute, with the filename but not the extension)
    - output_format: string = "both", the expected output format, must be csv, pkl or both
    """

    if output_format != "csv" and output_format != "pkl" and output_format != "both":
        raise Exception(f"{output_format} is not a valid output format, please use 'csv', 'pkl' or 'both'")

    df_train = pd.read_csv(filepath_or_buffer=train_path)
    df_test = pd.read_csv(filepath_or_buffer=test_path)

    df_application = pd.concat([df_train, df_test]).reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    df_application = preprocess_application(dataframe=df_application)

    # Checks if df_application is fragmented / memory usage too high (+100mb)
    if df_application.memory_usage(deep=True).sum() > 100 * 1024 * 1024:
        df_application = df_application.copy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_bureau = preprocess_bureau(bureau_path=bureau_path)
        df_credit_card = preprocess_credit_card(credit_card_path=credit_card_path)
        df_pos = preprocess_pos(pos_path=pos_path)
        df_installments = preprocess_installments(installments_path=installments_path)
        df_prev_app = preprocess_previous_app(previous_app_path=previous_app_path)

    # + bureau :
    prefixer(
        dataframe=df_bureau,
        prefix="BUR",
        ignore=["SK_ID_CURR"]
    )

    df_application = df_application.join(df_bureau, how="left", on="SK_ID_CURR")

    # + credit card :
    prefixer(
        dataframe=df_credit_card,
        prefix="CARD",
        ignore=["SK_ID_CURR"]
    )

    df_application = df_application.join(df_credit_card, how="left", on="SK_ID_CURR")

    # + pos
    prefixer(
        dataframe=df_pos,
        prefix="POS",
        ignore=["SK_ID_CURR"]
    )

    df_application = df_application.join(df_pos, how="left", on="SK_ID_CURR")

    # + installments
    prefixer(
        dataframe=df_installments,
        prefix="INST",
        ignore=["SK_ID_CURR"]
    )

    df_application = df_application.join(df_installments, how="left", on="SK_ID_CURR")

    # + prev app
    prefixer(
        dataframe=df_prev_app,
        prefix="PREV",
        ignore=["SK_ID_CURR"]
    )

    df_application = df_application.join(df_prev_app, how="left", on="SK_ID_CURR")

    # Na handler
    drop_list = get_cols_missing_thresh(dataframe=df_application, threshold=0.5)
    df_application.drop(columns=drop_list, inplace=True)
    df_application["BUR_CREDIT_CURRENCY_currency 4_mean"].fillna(value=-1, inplace=True)

    df_unknown = df_application[df_application["TARGET"] == -1]
    df_unknown.reset_index(drop=True, inplace=True)

    df_model = df_application[df_application["TARGET"] != -1]
    df_model.reset_index(drop=True, inplace=True)

    df_unknown = df_unknown.drop(columns=["TARGET"])

    if output_format == "csv":
        # Full dataset :
        df_application.to_csv(path_or_buf=f"{output_path}.csv", index=False)
        # Unknowns :
        df_unknown.to_csv(path_or_buf=f"{output_path}_test.csv", index=False)
        # Knowns :
        df_model.to_csv(path_or_buf=f"{output_path}_train.csv", index=False)

    elif output_format == "pkl":
        # Full dataset :
        df_application.to_pickle(path=f"{output_path}.pkl")
        # Unknowns :
        df_unknown.to_pickle(path=f"{output_path}_test.pkl")
        # Knowns :
        df_model.to_pickle(path=f"{output_path}_train.pkl")

    elif output_format == "both":
        # Full dataset :
        df_application.to_csv(path_or_buf=f"{output_path}.csv", index=False)
        df_application.to_pickle(path=f"{output_path}.pkl")
        # Unknowns :
        df_unknown.to_csv(path_or_buf=f"{output_path}_test.csv", index=False)
        df_unknown.to_pickle(path=f"{output_path}_test.pkl")
        # Knowns :
        df_model.to_csv(path_or_buf=f"{output_path}_train.csv", index=False)
        df_model.to_pickle(path=f"{output_path}_train.pkl")


if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        start = time.perf_counter()
        print("Processing ...")

        make_dataset(
            train_path="data/application_train.csv",
            test_path="data/application_test.csv",
            bureau_path="data/bureau.csv",
            credit_card_path="data/credit_card_balance.csv",
            installments_path="data/installments_payments.csv",
            pos_path="data/POS_CASH_balance.csv",
            previous_app_path="data/previous_application.csv",
            output_format="both",
            output_path="data/home_credit_data"
        )

        print("Done")
        end = time.perf_counter()
        print(f"Executed in {end - start} seconds")
