# add id and metrics to sentences


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

# https://stackoverflow.com/questions/17388213/find-the-similarity-metric-between-two-strings
from difflib import SequenceMatcher
import Levenshtein

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# https://stackoverflow.com/questions/6690739/high-performance-fuzzy-string-comparison-in-python-use-levenshtein-or-difflib
def keep_only_unique_text(e1, e2, fs, sim_thresh=0.9, verbose=True):
    """"""


    b = (fs['entity1_full'] == e1) & (fs['entity2_full'] == e2)
    tmp = fs.loc[b, :]
    short_text = tmp["short_text"].values

    if verbose:
        print(e1, e2)
        print(f"started with {b.sum()}")

    # keep array - initially all True
    keep = np.ones(len(short_text), dtype=bool)
    # store values for keeping track of / investigating
    # chk = []
    # for each entry compare to others
    for i in range(len(short_text)):
        # if the keep flag has been changed to False just contine
        if keep[i]:
            # for all the follow 'short_text' check how similar strings are
            for j in range(i+1, len(keep)):
                # if the one to check still has keep flag = True
                if keep[j]:
                    # measure distance
                    # s = similar(short_text[i], short_text[j])
                    s = Levenshtein.ratio(short_text[i], short_text[j])
                    # if above threshold, set keep flag to False (i.e. drop it)
                    if s > sim_thresh:
                        keep[j] = False
                        # print(s)
                        # chk += [(i, j)]
                else:
                    pass
        else:
            pass

    if verbose:
        print(f"keeping: {keep.sum()}")

    res = tmp.loc[keep]
    return res


if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_colwidth", 70)

    # ----
    # parameters
    # ----

    # if True will read articles from local json file: data/articles.json
    # - this can reduce burden on the remote database (network usage)
    read_local_articles = True

    # characters to abbreviate entity names with  - used in making 'id'
    abbrv_char = 4

    # aggregate function for dealing with overlapping sentences (those that start with the same)
    # could use max if want
    over_lap_aggfunc = 'min'

    # drop if too similar threshold
    # - values to keep must be below this one
    sim_thresh = 0.9

    # ---
    # read in value chain data / knowledge base
    # ---

    # read in locally stored valued chaines
    vc = pd.read_csv(get_data_path("VCHAINS.csv"))

    # the knowledge base
    kb = get_knowledge_base_from_value_chain_data(vc)

    # ---
    # read in full_sentences store locally
    # ---

    assert os.path.exists(get_data_path("full_sentences.json")), \
        f"looks like: {get_data_path('full_sentences.json')}, copy from google drive data/full_sentences.json"

    with open(get_data_path("full_sentences.json"), "r") as f:
        full_sents = json.load(f)

    fs = pd.DataFrame(full_sents)

    print(f"there were {len(fs)} sentences read in")
    # drop duplicates
    fs = fs.drop_duplicates()

    print(f"after dropping duplicates there are: {len(fs)}")

    # ---
    # add a sentence 'id' - based on attributes of
    # ---

    print("adding a sentence 'id'")

    # generate a condensed name - used for generating sentence id
    # NOTE: some companies may map to the same abbreviated name
    fs['e1'] = ["".join([i[:abbrv_char] for i in e.split(" ") if len(i) > 0])
                for e in fs["entity1_full"].values]
    fs['e2'] = ["".join([i[:abbrv_char] for i in e.split(" ") if len(i) > 0])
                for e in fs["entity2_full"].values]
    fs["sentence_range"] = fs[["start_sent", "end_sent"]].apply(lambda x: "|".join([str(i) for i in x]), axis=1)

    id_col = ["article", "date_publish", "source_domain", "e1", "e2", "sentence_range"]
    fs['id'] = fs[id_col].apply(lambda x: "_".join([re.sub(" |:|-", "", i) for i in x]), axis=1)

    # check the uniqueness of id
    id_count = pd.pivot_table(fs,
                              index='id',
                              values='full_sentence',
                              aggfunc='count')

    print("checking for none unique sentence 'id'")
    id_count.sort_values('full_sentence', ascending=False, inplace=True)
    id_count.reset_index(inplace=True)

    print(f"there are {(id_count['full_sentence'] > 1).sum()} / {len(id_count['full_sentence'])} 'id's "
          f"that have more than one sentence")
    print("These will be dropped for now")

    drop_id = id_count.loc[id_count["full_sentence"] > 1, "id"].values

    fs = fs.loc[~fs['id'].isin(drop_id)]

    # ---
    # overlapping sentences
    # ---

    print("handling overlapping sentences")

    # for a given article - source - date - entity pair
    fs['id_'] = fs['id'].apply(lambda x: "_".join(x.split("_")[:-1]))

    id2_count = pd.pivot_table(fs,
                               index='id_',
                               values='full_sentence',
                               aggfunc='count')
    id2_count.sort_values('full_sentence', ascending=False, inplace=True)
    id2_count.reset_index(inplace=True)

    # for each id_ find all the text that start with a given sentence
    # by taking the min of the end_sent we're finding the shortest text
    # that starts at start_sent for the given id_ (article, source, date, pair)
    start_sent = pd.pivot_table(fs,
                                index=["id_", "start_sent"],
                                values=["end_sent"],
                                aggfunc=over_lap_aggfunc).reset_index()
    # take a subset of the data
    fs = start_sent.merge(fs,
                          on=['id_', "start_sent", "end_sent"],
                          how="left")

    # ----
    # for each entity pair / triple - try to get only unique text
    # ----

    fs['short_text'] = ["".join([i[0] for i in e.split(" ") if len(i) > 0])
                        for e in fs['full_sentence']]

    e_pairs = fs[['entity1_full', 'entity2_full']].drop_duplicates()

    # store a dataframe (subset) of unique* sentences (per entity pair)
    u_sent = []
    e_pairs.index = np.arange(len(e_pairs))
    for idx, row in e_pairs.iterrows():
        print(f"{idx+1}/{len(e_pairs)}")
        e1, e2 = row['entity1_full'], row['entity2_full']
        u_sent += [keep_only_unique_text(e1, e2, fs)]

    us = pd.concat(u_sent)

    # ---
    # format full_sentence strings
    # ---

    # us_tmp = us.copy(True)

    # drop everything between () and/or {}, including the brackets
    us["full_sentence"] = [re.sub("[\{\(].*?[\}\)]", "", i)
                           for i in us["full_sentence"]]

    # standardise the quotes ” -> " and ’ -> '
    us["full_sentence"] = [re.sub("’", "'", i)
                           for i in us["full_sentence"]]

    us["full_sentence"] = [re.sub('”', '"', i)
                           for i in us["full_sentence"]]
    # double space
    us["full_sentence"] = [re.sub('  ', ' ', i)
                           for i in us["full_sentence"]]

    # ---
    # add some additional metrics
    # ---

    # number of characters
    us["num_chars"] = [len(i) for i in us['full_sentence']]

    # (approximate) number of tokens
    us["num_tokens"] = [len(i.split(" ")) for i in us['full_sentence']]

    # --
    # drop short_text
    # --

    us.drop("short_text", axis=1, inplace=True)

    # ---
    # write to file
    # ---

    res = us.to_dict("records")

    with open(get_data_path("processed_sentences.json"), "w") as f:
        json.dump(res, f, indent=4)


    # ----
    # example per triple
    # ----

    # ept = pd.pivot_table(us.loc[us['relation'] == "Supplier"],
    #                      index=['entity1_full', "entity2_full", "relation"],
    #                      values='full_sentence',
    #                      aggfunc="count")
    # ept.sort_values("full_sentence", ascending=False, inplace=True)
    #
    # ept.describe()

