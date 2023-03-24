import pandas as pd

from sklearn.preprocessing import StandardScaler


df_source = pd.read_pickle(filepath_or_buffer="data/home_credit_data.pkl")
df_train = pd.read_pickle(filepath_or_buffer="data/df_hc_nm_imputed.pkl")


# we need to redefine binary cols as variables that use 0, 1 or -1 (sentinel) :
def detect_binary_cols_with_sentinel(dataframe: pd.DataFrame):
    """
    Detects binary columns in a pandas dataframe.

    Args:
    - df: pandas dataframe.

    Returns:
    - list of binary column names.
    """

    binary_cols = []

    for col in dataframe.columns:
        unique_vals = dataframe[col].dropna().unique()
        if len(unique_vals) == 3 and set(unique_vals) == {0, 1, -1}:
            binary_cols.append(col)
        elif len(unique_vals == 2) and set(unique_vals) == {0, 1}:
            binary_cols.append(col)

    return binary_cols


ignore_cols = detect_binary_cols_with_sentinel(dataframe=df_source)
ignore_cols.append("SK_ID_CURR")

training_cols = df_train.columns

keep_col = training_cols

keep_col.append("SK_ID_CURR")

df_source_model_cols = df_source[keep_col]

numeric_to_scale = [col for col in df_train.columns if col not in ignore_cols]

# Using the same scaler for both train and test
scaler = StandardScaler()

df_source_model_cols[numeric_to_scale] = scaler.fit_transform(df_source_model_cols[numeric_to_scale])
df_train[numeric_to_scale] = scaler.transform(df_train[numeric_to_scale])

df_train.to_pickle(path="data/df_train_hc_nm_imputed_scaled.pkl")

df_source.to_pickle(path="data/test_hc_nm_imputed_scaled.pkl")
