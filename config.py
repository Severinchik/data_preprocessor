import os

from transformers import Mean, Std, ZScore, MaxIndex, AbsMeanDiff


class Config:
    FILE_IN = os.path.join('data', 'train.tsv')
    FILE_OUT = os.path.join('data', 'test_proc.tsv')
    COL_SEP = '\t'
    FEAT_SEP = ','
    BULK_SIZE = 100
    PIPELINE_SPEC = [
        [
            {
                'transformer': MaxIndex,
                'columns': [f'feature_2_{i}' for i in range(256)],
                'feature_name': 'feature_2'
            },
            {
                'transformer': Mean,
                'columns': [f'feature_2_{i}' for i in range(256)] + ['feature_2_max_index'],
                'res_key': 0
            }
        ],
        [
            {
                'transformer': AbsMeanDiff,
                'columns': ['feature_2_max_index'],
                'inputs': {'mean': 0},
                'replace': False
            },
            {
                'transformer': Std,
                'columns': [f'feature_2_{i}' for i in range(256)],
                'inputs': {'mean': 0},
                'res_key': 1
            }

        ],
        [
            {
                'transformer': ZScore,
                'columns': [f'feature_2_{i}' for i in range(256)],
                'inputs': {'mean': 0, 'std': 1},
                'replace': True
            }
        ]
    ]
