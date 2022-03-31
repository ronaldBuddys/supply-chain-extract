# check if kb has bi-directional relation

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
    # python package (nlp) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


from nlp.utils import get_bidirectional_suppliers, get_knowledge_base_from_value_chain_data
from nlp import get_data_path

if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)

    # ---
    # read in value chain data / knowledge base
    # ---

    # read in locally stored valued chaines
    vc = pd.read_csv(get_data_path("VCHAINS.csv"))

    # make knowledge base
    kb = get_knowledge_base_from_value_chain_data(vc)

    # ---
    # get bi-directional pairs
    # ---

    res = get_bidirectional_suppliers(kb)

    # write to file
    res.to_csv(get_data_path("bidirectional_supplier_relations.csv"), index=False)

    # ---
    # determine how many pairs which are bi-directional are in corpuse
    # ---
    #
    # ---
    # read in full_sentences store locally
    # ---
    # sent_file = get_data_path("processed_sentences.json")
    #
    # assert os.path.exists(sent_file), \
    #     f"looks like: {sent_file}, copy from google drive data/{os.path.basename(sent_file)}"
    #
    # with open(sent_file, "r") as f:
    #     full_sents = json.load(f)
    #
    # df = pd.DataFrame(full_sents)
    #
    # # ---
    #
    # tmp = res.merge(df,
    #                 left_on=["entity1", "entity2"],
    #                 right_on=["entity1_full", "entity2_full"],
    #                 how="left")
    #
    # tmp = tmp.loc[~pd.isnull(tmp["id"])]
