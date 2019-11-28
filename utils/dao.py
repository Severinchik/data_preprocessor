from datetime import datetime

from credentials import Credentials
from config import Config
from pymongo import MongoClient, ASCENDING


class DAO:
    def __init__(self):
        """
        Needed only once in order to merge feature groups by job index
        Decided to use Mongo as a replacement for RAM:
            1. fast and easy to use
            2. used mlab
            3. didn't want to install and configure caching for redis, which
                better solution for this case than mongodb
        """
        client = MongoClient(Credentials.MONGO_HOST,
                             Credentials.MONGO_PORT,
                             username=Credentials.MONGO_USER,
                             password=Credentials.MONGO_PASS,
                             authSource=Credentials.MONGO_DB_NAME,
                             authMechanism='SCRAM-SHA-1',
                             retryWrites=False)

        collection_name = f'exp-{str(datetime.now())}'
        self.collection = client[Credentials.MONGO_DB_NAME][collection_name]
        self.collection.create_index([('job_id', ASCENDING)], unique=True)
        self.bulk_size = Config.BULK_SIZE
        self.bulk = []

    def update(self, operation):
        self.bulk.append(operation)
        if len(self.bulk) == self.bulk_size:
            self.write_and_flush_bulk()

    def write_and_flush_bulk(self):
        if self.bulk:
            self.collection.bulk_write(self.bulk)
            self.bulk = []
