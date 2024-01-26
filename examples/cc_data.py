# parse data from cc_news dataset from hugging face

import re
import time
import json
import os
import sys
import pandas as pd
import numpy as np

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
from nlp import get_configs_path, get_data_path


if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)

    # require cc_download_articles is in nlp/data/
    # data_dir = get_data_path("cc_download_articles")
    # CHANGE THIS IF NEED BE!
    data_dir = "/mnt/hd1/data/cc_download_articles"
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

    # database
    cl = client["refinitiv"]['VCHAINS']
    # find many - store in dataframe
    vc = pd.DataFrame(list(cl.find(filter={})))
    # drop _id col
    vc.drop("_id", axis=1, inplace=True)

    # get all the company names
    # TODO: here want to allow to select names and aliases (shortened names)
    company_names = vc["Parent Name"].unique()
    # vc.sort_values("Revenue (USD)", ascending=False, inplace=True)
    # cn = vc[["Company Name", "Revenue (USD)"]].drop_duplicates()
    # company_names = cn["Company Name"].values[:200]
    # company_names = vc["Parent Name"].unique()
    # company_names = [vc["Parent Name"].unique(), vc["Company Name"].unique()]
    # company_names = np.concatenate(company_names)
    # company_names = np.unique(company_names.astype(str))

    # -------
    # search files for mention of company names
    # -------

    # write results to "news_articles" database
    art_db = client["news_articles"]

    # will write documents to several collections
    # articles: company_names if an article is found to contain any company will write
    articles = art_db["articles"]
    # names_found_in: contain a Company Name and the article ids (array) where that name was found
    names_found_in_articles = art_db["names_found_in_articles"]
    # names_searched_in: names already search in article - to avoid duplication
    # names_searched_in = art_db["names_searched_in"]
    # names_found_in_source: keep track of company names found in source
    # - can rule out sources where no company names have been found
    names_found_in_source = art_db["names_found_in_source"]


    # count = 0
    # 'walk' down the folder structure - will check
    for root, sub_dir, files in os.walk(data_dir, topdown=True):

        # source assumed to be basename of root directory
        # - this is the case for cc_news data from hugging face
        source = os.path.basename(root)

        print("*" * 1300)
        print(f"checking source:\n{source}")

        # find all json files
        json_files = [i for i in files if re.search("\.json$", i, re.IGNORECASE)]

        # keep track of the number of companys found in source
        names_found = []

        # find previously searched files in source
        # NOTE: it is assumed all the files in source directory were searched!
        already_searched = names_found_in_source.find_one(filter={"source": source})
        search_names = company_names
        if already_searched is not None:
            search_names = company_names[~np.in1d(company_names, already_searched['searched'])]

        if len(search_names) == 0:
            print("already searched names for this source, skipping")
            continue

        for jf in json_files:
            # the assumption is the file name is unique
            # TODO: validate this assumption
            # id = re.sub("\.json$", "", jf, re.IGNORECASE)

            # read in article - very quick (~1ms)
            with open(os.path.join(root, jf), "r") as _:
                a = json.load(_)

            # add the json file name - assumed to be unique! will use a identifier
            a["json_file"] = jf

            # check the names that have already be search for in this article


            # if the company name has already been searched for, skip searching it again
            # [ for cn in company_names if names_found_in_articles.find_one("")]

            # search for 500 company names ~ 30 - 50ms
            # t1 = time.perf_counter()
            for cn in search_names:
                if re.search(cn, a["maintext"]):
                    print("-"*10)
                    print(f"{cn} - found in:\n{a['title']}\nfrom: {source}")

                    names_found += [cn]
                    # see if this article (using json_file is identifier) has been uploaded
                    art_found = articles.find_one(filter={"json_file": jf})
                    # # if article is not in database, then add
                    if art_found is None:
                        print("adding article")
                        _id = articles.insert_one(document=a)
                        art_id = _id.inserted_id
                    # otherwise, get the article _id
                    else:
                        art_id = art_found["_id"]
                    # add to names_found_in_articles
                    names_found_in_articles.update_one(filter={"name": cn},
                                                       update={'$addToSet': {'articles': art_id}},
                                                       upsert=True)

        # write the names searched and found in all source articles
        names_found = np.unique(names_found).tolist()
        names_found_in_source.update_one(filter={"source": source},
                                         update={'$addToSet': {"found": {"$each": names_found},
                                                               "searched": {"$each": search_names.tolist()}}},
                                         upsert=True)

    #
    #
    # data_dir = os.path.join(get_parent_path(), "data", "cc_news", "cc_download_articles")
    # data_dir = get_data
    #
    # files = os.listdir(os.path.join(data_dir, "www.reuters.com"))
    #
    # reuters_dir = [i for i in os.listdir(data_dir) if re.search("reuters", i)]
    #
    # for rdir in reuters_dir:
    #     found = []
    #     for file in files:
    #         with open(os.path.join(data_dir, "www.reuters.com", file), "r+") as f:
    #             d = json.load(f)
    #
    #         if re.search("General Motors Co", d["maintext"], re.IGNORECASE):
    #             # print("check this out")
    #             found += [file]
    #
    #         # if re.search("Supply|supplier", d["maintext"], re.IGNORECASE):
    #         #     print("check this out")
    #         #     found += [file]
    #
    #
    # supplier = {}
    # file = found[0]
