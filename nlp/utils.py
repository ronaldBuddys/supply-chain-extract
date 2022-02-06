
# connect to mongodb atlas (remote database)
from pymongo import MongoClient


def get_database(username, password, clustername):
    # source: https://www.mongodb.com/languages/python

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    # TODO: review the particulars of connection string to understand impact - i.e. databasename was removed
    CONNECTION_STRING = f"mongodb+srv://{username}:{password}@{clustername}.mongodb.net/?retryWrites=true&w=majority"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient

    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client
