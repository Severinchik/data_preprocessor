from config import Config
from system import Pipeline


if __name__ == '__main__':
    pipeline = Pipeline(Config.FILE_OUT, Config.PIPELINE_SPEC)
    pipeline.process()
