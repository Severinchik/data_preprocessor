import numpy as np
import tempfile
from pymongo import UpdateOne

from config import Config
from utils import DAO


class FeatureMerger:
    def __init__(self):
        """

        """
        self.file_name = Config.FILE_IN
        self.dao = DAO()

        self.columns = ['job_id']
        self._cols_set = set(self.columns)

    def update(self, job_id, f_group, features):
        """

        :param job_id:
        :param f_group:
        :param features:
        :return:
        """
        result = {'job_id': job_id}
        features_map = {f'feature_{f_group}_{i}': f_val for i, f_val in enumerate(features)}

        merged = {**result, **features_map}
        f_names_set = set(merged.keys())
        f_diff = f_names_set - self._cols_set

        if f_diff:
            new_columns = sorted(f_diff, key=lambda x: (len(x), x))
            self._cols_set |= f_diff
            self.columns += new_columns

        self.dao.update(
            UpdateOne({'job_id': job_id}, {'$set': features_map}, upsert=True)
        )

    def screen(self):
        """

        :return:
        """
        with open(self.file_name, 'r') as file:
            file.readline()
            for line in file:
                job_id, features_str = line.rstrip().split(Config.COL_SEP)
                features_list = features_str.split(Config.FEAT_SEP)
                f_group, features = features_list[0], list(map(lambda x: int(x), features_list[1:]))
                self.update(job_id, f_group, features)

            self.dao.write_and_flush_bulk()

    def transform_out(self):
        """

        :return:
        """
        _, file = tempfile.mkstemp()

        with open(file, 'w') as f:
            f.write('\t'.join(self.columns) + '\n')
            for row in self.dao.collection.find():
                f.write('\t'.join([str(row.get(col, np.nan)) for col in self.columns]) + '\n')

        return file
