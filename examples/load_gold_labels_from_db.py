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


from supply_chain_extract.utils import get_database, niave_long_to_short_name, get_knowledge_base_from_value_chain_data, get_most_confidence_bidirectional_pair
from supply_chain_extract import get_configs_path, get_data_path




# get credentials
with open(get_configs_path("mongo0.json"), "r+") as f:
    mdb_cred = json.load(f)

# # get mongodb client - for connections
client = get_database(username=mdb_cred["username"],
                        password=mdb_cred["password"],
                        clustername=mdb_cred["cluster_name"])

art_db = client["news_articles"]
# get gold labels
gl = pd.DataFrame(list(art_db['gold_labels'].find(filter={})))
gold_labels = gl.loc[:,['label_id','gold_label']]

#pd.pivot_table(gl, index="gold_label", values="label_id", aggfunc="count") / len(gl)

print("Storing gold_labels.csv in data folder")
gold_labels.to_csv(get_data_path("gold_labels.csv"),index=False)