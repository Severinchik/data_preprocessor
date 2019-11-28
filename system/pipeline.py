import os
from typing import Dict, List
import tempfile
from tqdm import tqdm

from system import Specification, Stage
from streamers import FeatureMerger, StageRowReader, RowWriter
from transformers import ColumnTransformer, RowTransformer, XTransformer


class Pipeline:
    def __init__(self, file_out: str, stages: List[List[Dict]]):
        """

        :param file_out:
        :param stages:
        """
        print('Feature groups screening and syncing..')
        feature_merger = FeatureMerger()
        feature_merger.screen()
        print('Generating transformed tmp file..')
        file = feature_merger.transform_out()

        self.file_in = file
        self.file_out = file_out
        self.stages = [list(map(lambda spec: Specification(spec), stage)) for stage in stages]
        self.pipeline_len = len(self.stages)

    def process(self):
        """

        :return:
        """
        results = {}
        in_file = self.file_in
        _, out_file = tempfile.mkstemp()

        for i, stage_specs in tqdm(enumerate(self.stages)):
            if i == self.pipeline_len - 1:
                out_file = self.file_out
            else:
                _, out_file = tempfile.mkstemp()

            stage = Stage(stage_specs, results)
            reader = StageRowReader(in_file)
            writer = RowWriter(out_file)

            for row in reader.generate():
                for transformer, spec in stage.flow:
                    if issubclass(spec.transformer, XTransformer):
                        row = transformer.transform(row)

                    if issubclass(spec.transformer, RowTransformer):
                        row = transformer.transform(row, spec.feature_name)

                    if issubclass(spec.transformer, ColumnTransformer):
                        transformer.update(row)

                writer.write_line(row)
            writer.close()

            for transformer, spec in stage.flow:
                if issubclass(spec.transformer, ColumnTransformer):
                    results[spec.res_key] = transformer

            os.remove(in_file)
            in_file = out_file
