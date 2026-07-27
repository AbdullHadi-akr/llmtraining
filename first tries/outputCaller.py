import os
import pandas as pd


def get_op_path(op_num):
    """
    Finds the OP folder.
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


def _load_csv(op_num, keyword):
    """
    Generic CSV loader.
    """

    folder = get_op_path(op_num)
    file_path = find_csv(folder, keyword)

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"File is empty: {file_path}")

    return df


def load_FMU1(op_num):
    """
    Loads FMU1 dataframe.
    """

    return _load_csv(op_num, "fmu1")


def load_FMU2(op_num):
    """
    Loads FMU2 dataframe.
    """

    return _load_csv(op_num, "fmu2")


def get_FMU1_value(op_num, header_name, row):
    """
    Returns a FMU1 value.
    """

    df = load_FMU1(op_num)

    if header_name not in df.columns:
        raise KeyError(
            f"Column '{header_name}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if row < 0 or row >= len(df):
        raise IndexError(
            f"Row {row} out of range (0-{len(df)-1})"
        )

    return df.at[row, header_name]


def get_FMU2_value(op_num, header_name, row):
    """
    Returns a FMU2 value.
    """

    df = load_FMU2(op_num)

    if header_name not in df.columns:
        raise KeyError(
            f"Column '{header_name}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if row < 0 or row >= len(df):
        raise IndexError(
            f"Row {row} out of range (0-{len(df)-1})"
        )

    return df.at[row, header_name]


def get_FMU_row_count(op_num):
    """
    Returns number of valid FMU rows.
    """

    df = load_FMU1(op_num)

    df = df.dropna(how="all")

    return len(df)