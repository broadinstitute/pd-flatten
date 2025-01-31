from __future__ import annotations

import pandas as pd


def pd_flatten(
    df,
    explode_lists: bool = True,
    expand_dicts: bool = True,
    except_cols: list[str] | None = None,
    sep: str = "__",
    name_columns_with_parent: bool = True,
) -> pd.DataFrame:
    """
    Flatten a data frame by recursively exploding lists to separate rows and expanding
    dictionaries to separate columns.

    :param df: a data frame
    :param explode_lists: whether to split lists to separate rows
    :param expand_dicts: whether to split dictionaries to separate columns
    :param except_cols: an optional list of columns to exclude from flattening
    :param sep: a separator character to use between `parent_key` and its column names
    :param name_columns_with_parent: whether to "namespace" nested column names using
    their parents' column names
    :return: a flattened data frame
    """

    if except_cols is None:
        except_cols = []

    def do_explode_lists(this_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check each column of a data frame for lists and explode those values to separate
        rows.

        :param this_df: a data frame
        :return: the data frame with list values exploded to separate rows
        """

        for c in this_df.columns:
            if c not in except_cols and bool(
                this_df[c].apply(lambda x: isinstance(x, list)).any()
            ):
                this_df = this_df.explode(c).reset_index(drop=True)

        return this_df

    def do_expand_dicts(this_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check each column of a data frame for dictionaries and expand those values to
        separate columns.

        :param this_df: a data frame
        :return: the data frame with list values expanded to separate columns
        """

        for c in this_df.columns:
            if c not in except_cols and bool(
                this_df[c].apply(lambda x: isinstance(x, dict)).any()
            ):
                # replace NA's with empty dictionaries so that `pd.Series` doesn't
                # create an extraneous series named `0`
                this_df[c] = this_df[c].fillna(pd.Series([{}] * len(df)))

                expanded = this_df[c].apply(pd.Series)

                if name_columns_with_parent:
                    # "namespace" column names by their nested paths
                    expanded = expanded.add_prefix(f"{c}{sep}")

                # ensure that we aren't joining nested a column that has the same name
                # as one of the higher-level columns
                dup_cols = set(this_df.columns).intersection(set(expanded.columns))

                if len(dup_cols) > 0:
                    raise NameError(
                        f"Column names {dup_cols} on the column path `{c}` are "
                        "duplicated. Try calling `pd_flatten` with "
                        "`name_columns_with_parent=True`."
                    )

                this_df = this_df.drop(columns=[c]).join(expanded)

        return this_df

    prev_shape = None

    while prev_shape != df.shape:
        # continue iterating until we the number of rows and cols is unchanged
        prev_shape = df.shape

        if explode_lists:
            df = do_explode_lists(df)
        if expand_dicts:
            df = do_expand_dicts(df)

    return df
