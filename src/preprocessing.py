import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def ohe_transform(dataset, subset, prefix, ohe):
    """
    Transforms a specific column in a DataFrame using a fitted OneHotEncoder.

    This function validates inputs, creates new columns with a specified prefix,
    concatenates them to the original DataFrame, and removes the original source column.

    :param dataset: The input dataset containing the column to be transformed.
    :type dataset: pd.DataFrame
    :param subset: The name of the column within the dataset to be encoded.
    :type subset: str
    :param prefix: The prefix string to be added to the newly created one-hot columns.
    :type prefix: str
    :param ohe: A previously fitted OneHotEncoder instance.
    :type ohe: sklearn.preprocessing.OneHotEncoder
    :raises RuntimeError: If input types are incorrect or if the subset column
        is not found in the DataFrame.
    :return: A new DataFrame with the original column replaced by one-hot encoded columns.
    :rtype: pd.DataFrame
    """
    if not isinstance(dataset, pd.DataFrame):
        raise RuntimeError(
            "Fungsi ohe_transform: parameter dataset harus bertipe DataFrame!"
        )

    if not isinstance(ohe, OneHotEncoder):
        raise RuntimeError(
            "Fungsi ohe_transform: parameter ohe harus bertipe OneHotEncoder!"
        )

    if not isinstance(prefix, str):
        raise RuntimeError("Fungsi ohe_transform: parameter prefix harus bertipe str!")

    if not isinstance(subset, str):
        raise RuntimeError("Fungsi ohe_transform: parameter subset harus bertipe str!")

    try:
        _ = dataset.columns.get_loc(subset)
    except:
        raise RuntimeError(
            "Fungsi ohe_transform: parameter subset string namun data tidak ditemukan dalam daftar kolom yang terdapat pada parameter dataset."
        )

    print("Fungsi ohe_transform: parameter telah divalidasi.")
    dataset = dataset.copy()
    print(
        f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {list(dataset.columns)}"
    )
    col_names = [f"{prefix}_{cat}" for cat in ohe.categories_[0].tolist()]  # type: ignore
    encoded = pd.DataFrame(
        ohe.transform(dataset[[subset]]).toarray(),  # type: ignore
        columns=col_names,
        index=dataset.index,
    )
    dataset = pd.concat([dataset, encoded], axis=1)
    dataset.drop(columns=[subset], inplace=True)
    print(
        f"Fungsi ohe_transform: daftar nama kolom setelah dilakukan pengkodean adalah {list(dataset.columns)}\n"
    )

    return dataset
