
# connect to mongodb atlas (remote database)
import os.path

from pymongo import MongoClient
import pandas as pd
import numpy as np
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


def remove_suffix(name, suffixes):
    """helper function for niave_long_to_short_name"""
    for s in suffixes:
        # regex: space, word, space then any character to end
        # or
        name = re.sub(f" {s} .*$| {s}$", "", name)
    return name


def niave_long_to_short_name(all_names):
    """return a dictionary mapping long name to a short name
    using a rules based approach"""

    # TODO: short_name_map needs to be reviewed!, preferable to use some NLP package (spacy?)
    # This is pretty hard coded list of company name 'suffixes'
    # - some of these were taken by counting suffixes occrances, removing those and repeating
    # - others were just a gues
    suffixes = ['Inc', 'Corp', 'Ltd', 'Co', 'PLC', 'SA', 'AG', 'LLC', 'NV', 'SE',
                'ASA', 'Bhd', 'SpA', 'Association', 'Aerospace', 'AB', 'Oyj', "Plc"] + \
               ['Co', 'Holdings', 'Group', 'Technologies', 'International',
                'Systems', 'Energy', 'Communications', 'Airlines', 'Motor',
                'Technology', 'Oil', 'Motors', 'Corp', 'Industries', 'Steel',
                'Holding', 'Airways', 'Aviation', 'Automotive', 'Networks',
                'Electronics', 'Digital', 'BP', 'Electric', 'Aircraft',
                'US', 'Mobile', 'Software', 'Broadcom', 'Brands',
                'Service', 'Semiconductor', 'Petroleum'] + \
               ['Platforms', 'Precision', 'Industry', 'AeroSystems', 'Media', 'Petrochemical']



    #  'International Business Machines Corp' -> 'IBM'
    # 'News Corp' -> 'News Corp'
    # NOTE: if it starts with air it needs two words
    short_name = pd.DataFrame([(n, remove_suffix(n, suffixes)) for n in all_names],
                              columns=["name", "short"])
    # look at longer names
    short_name["len"] = [len(n) for n in short_name["short"]]
    short_name.sort_values("len", ascending=False, inplace=True)

    # making a mapping dictionary
    short_name_map = {i[0]: i[1] for i in zip(short_name["name"], short_name["short"])}

    return short_name_map


def get_knowledge_base_from_value_chain_data(vc, verbose=True):

    if verbose:
        print("generating knowledge base ")
    # NOTE: there can be duplicates in value chain data - a given pair may have multiple entries
    # - but for difference dates
    # here will only take the most recent

    relevant_cols = ["Parent Name", "Company Name", "Relationship", "Confidence Score (%)"]

    if verbose:
        print("taking most recent entries")

    # for each pair get the Last Update Date
    recent = pd.pivot_table(vc[relevant_cols + ["Last Update Date"]],
                            index=["Parent Name", "Company Name", "Relationship"],
                            values=["Last Update Date"],
                            aggfunc="max").reset_index()

    vc = recent.merge(vc[relevant_cols + ["Last Update Date"]],
                      on=["Parent Name", "Company Name", "Relationship", "Last Update Date"],
                      how="left")

    if verbose:
        print("'flipping' value chain: all Relationship = 'Customer' -> 'Supplier' ")
    # select a subset of value chain data to make knowledge base
    kb = vc[relevant_cols + ["Last Update Date"]].copy(True)
    kb.rename(columns={"Parent Name": "entity1", "Company Name": "entity2", "Relationship": "rel"},
              inplace=True)
    # select just customers
    c = kb.loc[kb["rel"] == "Customer"].copy(True)
    # select just suppliers
    s = kb.loc[kb["rel"] == "Supplier"].copy(True)

    # switch customer labels around
    c.rename(columns={"entity1": "entity2", "entity2": "entity1"}, inplace=True)
    c['rel'] = "Supplier"

    # combine
    kb = pd.concat([s, c])

    # drop duplicates
    kb = kb.drop_duplicates()

    # again get the mostly recently updated - appears data can be update in
    # one parent company table, but not other, leading to discrepancies
    kb_recent = pd.pivot_table(kb,
                               index=["entity1", "entity2", "rel"],
                               values="Last Update Date",
                               aggfunc="max").reset_index()
    kb = kb_recent.merge(kb,
                         on=["entity1", "entity2", "rel", "Last Update Date"],
                         how="left")

    # drop any entries where company supplies self
    kb = kb.loc[kb["entity1"] != kb["entity2"]]

    return kb


def get_bidirectional_suppliers(kb, verbose=True):
    """given the knowledge base extract the entity pairs that 'go both ways'
    i.e. (A supplies B) AND (B supplies A)"""

    if verbose:
        print("getting companys that have bi-directional supplier relationship")

    e1s = kb["entity1"].values
    e2s = kb["entity2"].values

    # get the unique entities 1
    e1s_u = np.unique(e1s)

    bi_dir = []

    # for each entity1 - get all the entity 2
    # - then for each of those entity2 see if when
    # - it's entity1 does it contain the other (org e1)
    for i, e1 in enumerate(np.unique(e1s_u)):
        if (i % 100 == 0) & verbose:
            print(f"{i}/{len(e1s_u)}")

        # all of e1's e2s
        e1_e2s = e2s[e1s==e1]

        # for each of the entity2, check if / when it's e1
        for e1_e2 in e1_e2s:
            #
            if e1 in e2s[e1s == e1_e2]:
                # print((e1, e1_e2))
                bi_dir.append((e1, e1_e2))

    # for each bi-directional entity pair
    out = []
    for bd in bi_dir:
        e1, e2 = bd

        # b1 = ((kb["entity1"] == e1) & (kb["entity2"] == e2)).values.any()
        # b2 = ((kb["entity1"] == e2) & (kb["entity2"] == e1)).values.any()

        b1 = ((e1s == e1) & (e2s == e2)).any()
        b2 = ((e1s == e2) & (e2s == e1)).any()

        assert b1 & b2, "expected both"

        k1 = kb.loc[(kb["entity1"] == e1) & (kb["entity2"] == e2)]
        k2 = kb.loc[(kb["entity1"] == e2) & (kb["entity2"] == e1)]

        out.append(pd.concat([k1, k2]))

    res = pd.concat(out)

    # due to bi-directional nature there will be duplicates
    # - drop those!
    res = res.drop_duplicates()

    return res


def get_most_confidence_bidirectional_pair(bi_dir):
    """given the input from bidirectional_supplier_relations.csv
    or the output from get_bidirectional_suppliers
    add a column indicating most confident of the pairs

    NOTE: pairs will be equal confidence will have both directions
    with 'most_conf' = True
    """
    # require the bi-directional relationships only have Supplier
    assert np.all(bi_dir["rel"].unique() == "Supplier"), "expect only 'rel' value to be 'Supplier'"

    # re-name columns (not needed)
    # bi_dir.rename(columns={"entity1": "entity1_full", "entity2": "entity2_full"},
    #               inplace=True)

    # get each pair - sorted
    bi_dir["pair"] = bi_dir[["entity1", "entity2"]].apply(lambda x: "|".join(np.sort(x)), axis=1)

    # get max confidence per pair
    bi_dir_max_c = pd.pivot_table(bi_dir,
                                  index=["pair"],
                                  values="Confidence Score (%)",
                                  aggfunc="max").reset_index()
    bi_dir_max_c.rename(columns={"Confidence Score (%)": "max_conf"}, inplace=True)

    # merge on the max confidence per pair
    bd = bi_dir.merge(bi_dir_max_c[["pair", "max_conf"]],
                      on=["pair"],
                      how="left")

    # add column to indicate if confidence score is maximal
    bd['most_conf'] = bd['Confidence Score (%)'] == bd['max_conf']

    return bd


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
