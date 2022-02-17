
import re
import time
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bson import ObjectId


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
from nlp import get_configs_path, get_data_path, get_plots_path


if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)

    # require cc_download_articles is in nlp/data/
    # data_dir = get_data_path("cc_download_articles")
    # data_dir = "/mnt/hd1/data/cc_download_articles"
    # assert os.path.exists(data_dir), f"data_dir:\n{data_dir}\ndoes not exist, get from "

    # ----
    # connect to database
    # ----

    # get credentials
    with open(get_configs_path("mongo.json"), "r+") as f:
        mdb_cred = json.load(f)

        # get mongodb client - for connections
    client = get_database(username=mdb_cred["username"],
                          password=mdb_cred["password"],
                          clustername=mdb_cred["cluster_name"])

    # --
    # get the company names from refinitiv value chains
    # ---

    print(f"database names: {client.list_database_names()}")

    # find many - store in dataframe
    vc = pd.DataFrame(list(client["refinitiv"]['VCHAINS'].find(filter={})))
    # drop _id col
    vc.drop("_id", axis=1, inplace=True)

    # ----
    # analyse names found in sources
    # ----

    # news articles and related collections found in: 'new_articles' database
    arts_db = client["news_articles"]
    arts_db.name

    print(f"the collections found in database '{arts_db.name}':\n{arts_db.list_collection_names()}")

    print("getting all the number names found in each source")
    tmp = []
    for ns in arts_db['names_found_in_source'].find():
        if 'found' in ns:
            tmp += [{'source': ns['source'], 'num_names': len(ns['found'])}]

    df = pd.DataFrame(tmp)

    df.sort_values("num_names", ascending=False, inplace=True)

    _ = df.head(50)
    # fig, axs = plt.figure()
    _.plot.bar(x='source', y='num_names', figsize=(12,12))
    plt.xticks(rotation=60, ha='right')
    plt.xlabel("source")
    plt.ylabel("count of times name found in any article from source")
    plt.title(f"Company Names found in any Article by Source\n top: {len(_)} out of {len(df)} sources")
    plt.tight_layout()
    plt.savefig(get_plots_path("names_found_per_source_from_cc_news.png"))

    # ---
    # articles
    # ---

    print(f"number of articles found with a mention of company name: {len(list(arts_db['articles'].find()))}")

    # TODO: use 'names_found_in_articles' collection to get articles mentioning name
    #  and search for the suppliers / customers  usign data from refinitiv
