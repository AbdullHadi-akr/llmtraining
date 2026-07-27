import os
import pandas as pd


def get_op_path(op_num):
    """
    Finds the OP folder.

    Example:
    OP03 -> ./storeOps/OP03
    """

    op_str = f"OP{int(op_num):02d}"

    candidates = [
        os.path.join(".", "storeOps", op_str),
        os.path.join(".", "storeOPs", op_str),
    ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        f"Could not find OP folder. Checked:\n{candidates}"
    )


def find_csv(folder, keyword):
    """
    Finds a CSV file containing the given keyword.
    """

    for file in os.listdir(folder):
        if (
            file.lower().endswith(".csv")
            and keyword.lower() in file.lower()
        ):
            return os.path.join(folder, file)

    raise FileNotFoundError(
        f"No CSV containing '{keyword}' found in {folder}"
    )


def load_input_signals(op_num):
    """
    Loads the complete Input dataframe.
    """

    folder = get_op_path(op_num)
    file_path = find_csv(folder, "input")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"File is empty: {file_path}")

    return df


def get_input_values(op_num):
    """
    Returns first row as numpy array.
    """

    df = load_input_signals(op_num)

    return df.iloc[0].to_numpy(dtype=float)


def load_fluid_properties(op_num):
    """
    Loads the fluid properties file.
    """

    folder = get_op_path(op_num)
    file_path = find_csv(folder, "fluid")

    df = pd.read_csv(file_path, sep=";")

    if df.empty:
        raise ValueError(f"File is empty: {file_path}")

    return df


def get_fluid_value(op_num, header_name):
    """
    Returns a fluid property by column name.
    """

    df = load_fluid_properties(op_num)

    if header_name not in df.columns:
        raise KeyError(
            f"Column '{header_name}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    return df.iloc[0][header_name]