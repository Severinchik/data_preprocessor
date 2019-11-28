from typing import Dict, List, Tuple
import numpy as np

from transformers import ColumnTransformer, RowTransformer, XTransformer


class Mean(ColumnTransformer):
    def __init__(self, columns: List):
        """

        :param columns:
        """
        name = 'mean_({})'
        super().__init__(name, columns)
        self.n = 0

    def update(self, row: Dict):
        """

        :param row:
        :return:
        """
        cols = self._row_cols_intersection(row)
        if not cols:
            raise Exception('Nothing to update')

        args = np.array([[row[col], self._get_old_value(col)] for col in cols])
        x, old_vals = args[:, 0], args[:, 1]
        new_vals = (self.n * old_vals + x) / (self.n + 1)

        for col, new_val in zip(cols, new_vals):
            self.vals[col] = new_val

        self.n += 1


class Std(ColumnTransformer):
    def __init__(self, columns: List, mean: Mean):
        """

        :param columns:
        :param mean:
        """
        name = 'std_({})'
        super().__init__(name, columns)
        self.mean_stat = Mean(columns)
        self.mean = mean

    def update(self, row: Dict):
        """

        :param row:
        :return:
        """
        cols = self._row_cols_intersection(row)
        if not cols:
            raise Exception('Nothing to update')

        args = np.array([[row[col], self.mean.vals[col]] for col in cols])
        x, means = args[:, 0], args[:, 1]
        new_vals = (x - means) ** 2
        new_row = dict(zip(cols, new_vals))

        self.mean_stat.update(new_row)

        for col in cols:
            self.vals[col] = np.sqrt(self.mean_stat.vals[col])


class MaxIndex(RowTransformer):
    def __init__(self, columns: List, replace=False):
        """

        :param columns:
        """
        name = '{}_max_index'
        super().__init__(name, columns, replace)

    def transform(self, row: Dict, feature_name: str):
        """

        :param row:
        :param feature_name:
        :return:
        """
        cols = self._row_cols_intersection(row)
        if not cols:
            raise Exception('Nothing to transform')

        x = np.array([row[col] for col in cols])

        result = {self.feature_name(feature_name): np.argmax(x)}

        if self.replace:
            return result

        return {**row, **result}


class ZScore(XTransformer):
    def __init__(self, columns: List, mean: Mean, std: Std, replace=False):
        """

        :param columns:
        :param mean:
        :param std:
        """
        name = '{}_stand'
        super().__init__(name, columns, replace)
        self.mean = mean
        self.std = std

    def transform(self, row: Dict):
        """

        :param row:
        :return:
        """
        cols = self._row_cols_intersection(row)
        if not cols:
            raise Exception('Nothing to transform')

        args = np.array([[row[col], self.mean.vals[col], self.std.vals[col]] for col in cols])
        x, mean, std = args[:, 0], args[:, 1], args[:, 2]

        new_vals = (x - mean) / std
        result = dict(zip([self.feature_name(col) for col in cols], new_vals))

        if self.replace:
            new_row = {key: val for key, val in row.items() if key not in self._col_set}
            return {**new_row, **result}

        return {**row, **result}


class AbsMeanDiff(XTransformer):
    def __init__(self, columns: List, mean: Mean, replace=False):
        name = '{}_abs_mean_diff'
        super().__init__(name, columns, replace)
        self.mean = mean

    def transform(self, row: Dict):
        cols = self._row_cols_intersection(row)
        if not cols:
            raise Exception('Nothing to transform')

        args = np.array([[row[col], self.mean.vals[col]] for col in cols])
        x, mean = args[:, 0], args[:, 1]

        new_vals = np.abs(x - mean)

        result = dict(zip([self.feature_name(col) for col in cols], new_vals))

        if self.replace:
            return result

        return {**row, **result}
