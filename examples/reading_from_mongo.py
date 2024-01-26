# example of querying mongodb

import json
import os
import sys

import numpy as np
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

    pd.set_option("display.max_columns", 200)

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
    print(f"type(db): {type(db)}")
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

    #
    df = vc.loc[vc["Parent Id"] == 4298546138]

    # get only US companys
    us_comp =  vc.loc[(vc["Country/Region"] == "United States of America"), ["Company Name", "Identifier"]].drop_duplicates()

    # find american company with few customers
    _ = vc.loc[(vc["Parent Name"].isin(us_comp["Company Name"])) & \
            (vc["Relationship"] == "Customer")]

    # pd.pivot_table(_,
    #                index=["Parent Name"],
    #                values="Company Name",
    #                aggfunc="count")
    #
    # pd.pivot_table(vc.loc[vc["Parent Name"].isin(us_comp["Company Name"])])

    # -----
    # Articles with Names
    # -----

    # connect to  new_articles database
    art_db = client["news_articles"]
    # articles collection
    articles = art_db["articles"]

    # find all articles that have a names_in_text field
    articles_with_names = list(articles.find(filter={"names_in_text": {"$exists": True}}))
    # get all the names found so far
    all_found_names = [i["names_in_text"] for i in articles_with_names]
    all_found_names = np.unique(np.concatenate(all_found_names))

    # get the Parent, Company, Relationship
    df = vc[["Parent Name", "Company Name", "Relationship"]].drop_duplicates()

    # select only the names found in all_found_names
    df = df.loc[(df["Parent Name"].isin(all_found_names)) | (df["Company Name"].isin(all_found_names))]

    # search for each Parent, Company, Relationship
    res = []

    # NOTE: this seems a bit slow, may want to see how can do more quickly
    for i in range(len(df)):
        pcr = df.iloc[i, :]

        # ---
        # search for both names in
        # ---

        p, c, r = pcr

        # is this kind of slow?
        a = articles.find_one(filter={"names_in_text": {"$all": [p, c]}})
        if a is not None:
            print(" | ".join(pcr))
            res.append({**pcr.to_dict(), "_id": a["_id"], "title": a["title"]})

    # store results in data frame
    resdf = pd.DataFrame(res)

    # can get an article by searching for it's _id

    _ = resdf.iloc[-1,:]
    print(_)
    some_id = _["_id"]

    a = articles.find_one(filter={"_id": some_id})

    print(a["maintext"])




    # prev_fetched = pd.DataFrame(list(client['refinitiv']["VCHAINS"].find(filter={})))
    #
    # filename = src_path + "/nlp/data/VCHain_DB_Data_As_of_" + time.strftime("%Y_%m_%d_%H_%M")+".xlsx"
    # print("Saving {}".format(filename))
    # prev_fetched.to_excel(filename)