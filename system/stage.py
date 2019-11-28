from typing import Dict, List, Tuple

from transformers import ColumnTransformer, RowTransformer, XTransformer


class Specification:
    def __init__(self, specification):
        """

        :param specification:
        """
        self.transformer = specification['transformer']
        self.columns = specification['columns']
        self.feature_name = specification.get('feature_name')
        self.inputs = specification.get('inputs')
        self.replace = specification.get('replace', False)
        self.res_key = specification.get('res_key')


class Stage:
    def __init__(self, stage_specs: List[Specification], results: Dict):
        """

        :param stage_specs:
        :param results:
        """
        self.flow = []

        for spec in stage_specs:
            kwargs = {}

            if spec.inputs is not None:
                kwargs = {key: results[val] for key, val in spec.inputs.items()}

                if spec.replace and (
                        issubclass(spec.transformer, RowTransformer) or
                        issubclass(spec.transformer, XTransformer)
                ):
                    kwargs = {**kwargs, 'replace': spec.replace}

            self.flow.append((
                spec.transformer(spec.columns, **kwargs),
                spec
            ))
