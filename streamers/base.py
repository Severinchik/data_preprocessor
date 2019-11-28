from typing import Dict
import numpy as np

from config import Config


class StageRowReader:
    def __init__(self, file_name: str):
        """

        :param file_name:
        """
        self.file_name = file_name
        file = open(self.file_name, 'r')
        self.columns = file.readline().rstrip().split(Config.COL_SEP)
        file.close()

    def _features_map(self, features):
        """

        :param features:
        :return:
        """
        return dict(zip(self.columns, features))

    def generate(self):
        """

        :return:
        """
        with open(self.file_name, 'r') as file:
            file.readline()
            for line in file:
                features = np.array(line.rstrip().split(Config.COL_SEP), dtype=np.float64)

                yield self._features_map(features)


class RowWriter:
    def __init__(self, file_name):
        """

        :param file_name:
        """
        self.file_name = file_name
        self.file = open(self.file_name, 'w')
        self.columns = None

    def write_line(self, row: Dict):
        """

        :param row:
        :return:
        """
        if self.columns is None:
            self.columns = row.keys()
            self.file.write('\t'.join(self.columns) + '\n')

        self.file.write('\t'.join([str(row[col]) for col in self.columns]) + '\n')

    def close(self):
        """

        :return:
        """
        self.file.close()
