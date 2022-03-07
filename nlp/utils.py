
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


def make_reg_tree(c, cur_node=1, max_node=1):

    if cur_node >= max_node:
        return {"|".join(c): c}
    else:
        n = len(c)
        _ = {"|".join(c): {

            **make_reg_tree(c[:n//2],
                            cur_node=cur_node+1,
                            max_node=max_node),
            **make_reg_tree(c[n//2:],
                            cur_node=cur_node+1,
                            max_node=max_node)},
        }
        return _

def get_list_from_tree(text, rtree, out=None):

    if out is None:
        out = []

    for k, v in rtree.items():
        if isinstance(v, dict):
            if re.search(k, text):
                get_list_from_tree(text, v, out)
        else:
            if re.search(k, text):
                out += v
    return out

if __name__ == "__main__":

    # search for company information
    ticker_info = search_company(ticker="GM")

    # ---
    # searching for (many) names in text - using regex
    # ---
    # TODO: add unit test for make_reg_tree and get_list_from_tree

    # 'company' names
    c = ['a', 'b', 'c', 'd',
         'e', 'f', 'h', 'i',
         'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q',
         "r", "s", "t"]

    # make a tree, combining names with | and splitting up the list as go down to leafs
    rtree = make_reg_tree(c, cur_node=0, max_node=3)

    print(json.dumps(rtree, indent=4))

    # simple text to search for 'company' names
    text = "asdf"

    # aim is to get a reduced set of company names to search
    # - as it can be very slow to search for each company one by one
    short_list = get_list_from_tree(text, rtree)
