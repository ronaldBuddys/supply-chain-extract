
# connect to mongodb atlas (remote database)
import os.path

from pymongo import MongoClient
import pandas as pd
import json
import re
from OpenPermID import OpenPermID
from nlp import get_configs_path

def get_database(username, password, clustername):
    # source: https://www.mongodb.com/languages/python

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    # TODO: review the particulars of connection string to understand impact - i.e. databasename was removed
    CONNECTION_STRING = f"mongodb+srv://{username}:{password}@{clustername}.mongodb.net/?retryWrites=true&w=majority"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient

    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client


def search_company(ticker, api_key=None, num=10):

    # TODO: could make the first bit of this function it's own, and then
    #  get have other functions/methods to get other specific info
    if api_key is None:
        print("api_key is None, will read from configs/keys.json")
        keys_file = get_configs_path("keys.json")

        assert os.path.exists(keys_file), f"file:\n{keys_file}\ndoes not exist"
        with open(keys_file, "r") as f:
            keys = json.load(f)

        assert "refinitiv" in keys, "'refinitiv' not found in keys.json\n" \
                                   "to get one follow instructions found at\n" \
                                   "https://github.com/Refinitiv-API-Samples/Article.OpenPermID.Python.APIs"
        api_key = keys["refinitiv"]

    # initialise OpenPermID object, and set access token (api_key)
    opid = OpenPermID()
    opid.set_access_token(api_key)

    # query
    q = f"ticker: {ticker}"
    print(f"query = {q}")

    # run query
    res, err = opid.search(q, entityType='all', format="dataframe",
                           start=1, num=num, order='rel')

    if err is not None:
        print(f"there was an error:\n{err}")
        return None

    orgs = res["organizations"]
    if len(orgs) == 0:
        print("there were no search results")
        return None

    # get the company ID
    id_base_url = 'https://permid.org/'

    # take first results - most relevant?
    org = orgs.iloc[0, :].to_dict()
    org["id"] = re.sub(f"^{id_base_url}", "", org["@id"])
    # HARDCODED: remove a predict of #- (will this allways be the case
    org["id"] = re.sub(f"^\d-", "", org["id"])

    return org


if __name__ == "__main__":

    # search for company information
    ticker_info = search_company(ticker="GM")

