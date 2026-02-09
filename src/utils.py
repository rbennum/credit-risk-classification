import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split


def load_data(fname: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame and display its dimensions.

    :param fname: The file path or buffer of the CSV file to be read.
    :type fname: str
    :return: A DataFrame containing the loaded data.
    :rtype: pandas.DataFrame
    """
    data = pd.read_csv(fname)
    print(f"Data Shape: [{data.shape}]")
    return data


def split_feature_target(
    data: pd.DataFrame, target_col="loan_status"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features (X) and target (y).

    :param data: Input DataFrame.
    :type data: pd.DataFrame
    :param target_col: Target column name, defaults to "loan_status".
    :type target_col: str, optional
    :return: Feature set (X) and target series (y).
    :rtype: typle[pd.DataFrame, pd.Series]
    """
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    print(f"Original data shape: {data.shape}")
    print(f"X data shape: {X.shape}")
    print(f"y data shape: {y.shape}")
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into X_train, X_test, y_train, y_test.

    :param X: A feature dataset.
    :type X: pd.DataFrame
    :param y: A target dataset.
    :type y: pd.Series
    :param test_size: Represents the number of test samples.
    :type test_size: float
    :param random_state: Controls the shuffling applied to the data, defaults to None.
    :type random_state: int, optional
    :return: X_train, X_test, y_train, y_test. In that order.
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"X train shape: {X_train.shape}")
    print(f"X test shape: {X_test.shape}")
    print(f"y test shape: {y_train.shape}")
    print(f"y test shape: {y_test.shape}\n")
    return X_train, X_test, y_train, y_test


def serialize_data(data: pd.DataFrame | pd.Series, path: str):
    """
    Serialize the input into a file.

    :param data: Data to be serialized.
    :type data: pd.DataFrame | pd.Series
    :param path: File path.
    :type path: str
    """
    parent_dir = os.path.dirname(path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    joblib.dump(data, filename=path, compress=3)
    print(f"Data serialized on {path}.")


def deserialize_data(path: str) -> pd.DataFrame | pd.Series:
    """
    Deserialize a file into DataFrame/Series.

    :param path: File path.
    :type path: str
    :raises TypeError: If the deserialized object is not a pandas type.
    :return: The restored pandas object.
    :rtype: pd.DataFrame | pd.Series
    """
    data = joblib.load(path)

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(f"Expected DataFrame/Series, got {type(data)}")

    return data
