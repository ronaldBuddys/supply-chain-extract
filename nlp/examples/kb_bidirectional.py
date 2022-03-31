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


from nlp.utils import get_database, niave_long_to_short_name, get_knowledge_base_from_value_chain_data
from nlp import get_configs_path, get_data_path

if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)


    # ---
    # read in value chain data / knowledge base
    # ---

    # read in locally stored valued chaines
    vc = pd.read_csv(get_data_path("VCHAINS.csv"))

    # the knowledge base
    kb = get_knowledge_base_from_value_chain_data(vc)


    # ---
    #
    # ---

    e1s = kb["entity1"].values
    e2s = kb["entity2"].values

    # get the unique entities 1
    e1s_u = np.unique(e1s)

    bi_dir = []

    # for each entity1 - get all the entity 2
    # - then for each of those entity2 see if when
    # - it's entity1 does it contain the other (org e1)
    for i, e1 in enumerate(np.unique(e1s_u)):
        if i % 100 == 0:
            print(f"{i}/{len(e1s_u)}")

        # all of e1's e2s
        e1_e2s = e2s[e1s==e1]

        # for each of the entity2, check if / when it's e1
        for e1_e2 in e1_e2s:
            #
            if e1 in e2s[e1s == e1_e2]:
                # print((e1, e1_e2))
                bi_dir.append((e1, e1_e2))

    e1, e2 = bi_dir[1]

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


    res.to_csv(get_data_path("bidirectional_supplier_relations.csv"), index=False)
