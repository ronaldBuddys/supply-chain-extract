# example of querying mongodb

import json
import os
import sys
import pandas as pd

try:
    # python package (nlp) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


from nlp.utils import get_database
from nlp import get_configs_path


if __name__ == "__main__":

    # get credentials
    with open(get_configs_path("mongo.json"), "r+") as f:
        mdb_cred = json.load(f)

    # get mongodb client - for connections
    client = get_database(username=mdb_cred["username"],
                          password=mdb_cred["password"],
                          clustername=mdb_cred["cluster_name"])

    # database names
    print(f"database names: {client.list_database_names()}")

    # database
    db = client["refinitiv"]
    print("collections in database")
    print(db.list_collection_names())

    # collection
    cl = db['VCHAINS']

    # find one - any
    d = cl.find_one()

    # find many - store in dataframe
    vc = pd.DataFrame(list(cl.find(filter={})))
    # drop _id col
    vc.drop("_id", axis=1, inplace=True)

    vc.loc[vc["Parent Id"] == 4298546138]