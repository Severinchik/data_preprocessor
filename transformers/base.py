from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np


class FeatureOperator(ABC):
    def __init__(self, name: str, columns: List):
        """
        Base operator ofr all feature transformations

        :param name:
        :param columns:
        """
        self.name = name
        self.columns = columns
        self._col_set = set(columns)

    def _row_cols_intersection(self, row):
        """

        :param row:
        :return:
        """
        return list(sorted(set(row.keys()) & self._col_set, key=lambda x: (len(x), x)))

    def feature_name(self, feature_name) -> str:
        """

        :param feature_name:
        :return:
        """
        return self.name.format(feature_name)


class ColumnTransformer(FeatureOperator, ABC):
    def __init__(self, name: str, columns: List, init_val=0):
        """
        Steam based statistics aggregator
        Gathers data from row to row for each needed column

        :param name:
        :param columns:
        :param init_val:
        """
        super().__init__(name, columns)
        self.vals = defaultdict(lambda: np.nan)
        self._init_val = init_val

    def _get_old_value(self, feature_name):
        """

        :param feature_name:
        :return:
        """
        old_val = self.vals[feature_name]
        if np.isnan(old_val):
            old_val = self._init_val
        return old_val

    @abstractmethod
    def update(self, row: Dict):
        """

        :param row:
        :return:
        """
        pass


class RowTransformer(FeatureOperator, ABC):
    def __init__(self, name: str, columns: List, replace=False):
        """
        Applies transformation on row

        :param name:
        :param columns:
        :param replace:
        """
        super().__init__(name, columns)
        self.replace = replace

    @abstractmethod
    def transform(self, row: Dict, feature_name: str):
        """

        :param row:
        :param feature_name:
        :return:
        """
        pass


class XTransformer(FeatureOperator, ABC):
    def __init__(self, name: str, columns: List, replace=False):
        """
        Applies transformation on each element in a row

        :param name:
        :param columns:
        :param replace:
        """
        super().__init__(name, columns)
        self.replace = replace

    @abstractmethod
    def transform(self, row: Dict):
        """

        :param row:
        :return:
        """
        pass
