# given a json file / collection of sentences - run some basic analysis


import json
import os
import re
import time
import gzip
import itertools


import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


try:
    # python package (supply_chain_extract) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


from supply_chain_extract.utils import get_database, niave_long_to_short_name, get_knowledge_base_from_value_chain_data
from supply_chain_extract import get_configs_path, get_data_path


if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_colwidth", 70)

    # ----
    # parameters
    # ----

    # if True will read articles from local json file: data/articles.json
    # - this can reduce burden on the remote database (network usage)
    read_local_articles = True

    # make number of sentences to combine
    max_num_sentences_to_combine = 5

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

    # ---
    # read in value chain data / knowledge base
    # ---

    vc = pd.DataFrame(list(client["knowledge_base"]["KB"].find(filter={})))

    # there are some missing comany names? exclude those
    vc = vc.loc[~pd.isnull(vc['Company Name'])]

    # the knowledge base
    kb = get_knowledge_base_from_value_chain_data(vc)

    # ---
    # read in full_sentences store locally
    # ---

    assert os.path.exists(get_data_path("full_sentences.json")), \
        f"looks like: {get_data_path('full_sentences.json')}, copy from google drive data/full_sentences.json"

    with open(get_data_path("full_sentences.json"), "r") as f:
        full_sents = json.load(f)

    # ---
    # read in article data - for reference
    # ---

    # get locally stored data
    assert os.path.exists(get_data_path("articles.json")), \
        f"{get_data_path('articles.json')} does not exist, get from the google drive and save locally"

    with open(get_data_path("articles.json"), "r") as f:
        article_list = json.load(f)

    articles = {re.sub("\.json$", "", f["json_file"]): f
                for f in article_list if "names_in_text" in f}

    # ---
    # get counts of occurrences / summary statistics
    # ---

    keys = list(full_sents[0].keys())

    df = pd.DataFrame(full_sents)
    # exclude the full_sentence - for now
    # df.drop("full_sentence", axis=1, inplace=True)

    print(f"unique relations: {df['relation'].unique()}")

    print(f"number of unique articles: {len(df['article'].unique())}")

    # ---
    # count of relation
    # ---
    relation_count = pd.pivot_table(df,
                                    index="relation",
                                    values="full_sentence",
                                    aggfunc="count")
    print("count of 'relation'")
    print(relation_count)

    # ---
    # sentences per articles
    # ---

    sent_per_art = pd.pivot_table(df, index='article', values="full_sentence", aggfunc='count')
    sent_per_art = sent_per_art.reset_index()
    sent_per_art.sort_values("full_sentence", ascending=False, inplace=True)

    print(f"average extract sentences per article: {sent_per_art['full_sentence'].mean():.2f}")

    plt.plot(sent_per_art['full_sentence'].values)
    plt.title("sentences per article")
    plt.ylabel("sentences in article")
    plt.show()

    # there is heavy skew
    print("top articles:")
    print(sent_per_art.head(20))

    sent_per_art.loc[sent_per_art['full_sentence'] < 10, 'full_sentence'].mean()

    # ---
    # sentence spread - entity pair can exist across multiple sentence - consider those
    # ---

    num_sent = pd.pivot_table(df,
                              index='num_sentence',
                              values="full_sentence",
                              aggfunc="count")

    plt.plot(num_sent)
    plt.title("number of 'full sentence' spread across actual sentences")
    plt.xlabel("'actual' sentence count (from article)")
    plt.show()

    # ---
    # sentences per pair
    # ---

    sentence_per_triple = pd.pivot_table(df,
                                         index=["entity1", "entity2", "relation"],
                                         values=["full_sentence"],
                                         aggfunc="count").reset_index()

    sentence_per_triple['full_sentence'].mean()


    # may want to consider:
    # - articles with low(ish) number of companies
    # - sentences that are short


    # np.quantile(sent_per_art['full_sentence'].values, q=0.99)
    #
    # sent_per_art.loc[sent_per_art['full_sentence'] < 50]
    #
    # articles[sent_per_art['article'].values[0]]
    #
    # df.loc[df['article'] == sent_per_art['article'].values[0]]
    #
    # articles['18d8635ec7c9a5def5fe672feecd2e8743ed63d48445aa9424ffce5540eb9640']
    #
    # # sent_per_art['articles'][]
    #
    # df.loc[df['article'] == '18d8635ec7c9a5def5fe672feecd2e8743ed63d48445aa9424ffce5540eb9640']