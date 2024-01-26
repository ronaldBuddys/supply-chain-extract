# example use of New York Times api


import json
import os
import sys
import re
import time
import warnings
import numpy as np
import pandas as pd

import pyautogui
import pymongo

import selenium
from selenium.webdriver.common.keys import Keys
from supply_chain_extract.utils import get_database
from supply_chain_extract import get_configs_path


if __name__ == "__main__":

    # ---
    # connect to db
    # ---
    # with open(get_configs_path("mongo.json"), "r+") as f:
    #     mdb_cred = json.load(f)
    #
    # # get mongodb client - for connections
    # client = get_database(username=mdb_cred["username"],
    #                       password=mdb_cred["password"],
    #                       clustername=mdb_cred["cluster_name"])
    # # check session info - just to check connection
    # client.server_info()
    # print("connected to mongodb")

    # ---
    # get nyt api key from file
    # ---
    key_name = "mediastack"
    keys_file = get_configs_path("keys.json")
    assert os.path.exists(keys_file), f"file:\n{keys_file}\ndoes not exist"
    with open(keys_file, "r") as f:
        keys = json.load(f)

    assert key_name in keys, f"'{key_name}'not found in:\n{keys_file}"
    api_key = keys[key_name]

    # ---
    # search api
    # ---

    # https://mediastack.com/documentation

    # ? access_key = YOUR_ACCESS_KEY
    # & date = 2020-02-19

    # base_url = "http://api.mediastack.com/v1/news"
    # search_url = f"{base_url}?access_key={api_key}"
    # kword = 'keywords=general motors'
    # search_url += f"&{kword}"

    import http.client, urllib.parse

    conn = http.client.HTTPConnection('api.mediastack.com')

    params = urllib.parse.urlencode({
        'access_key': api_key,
        # 'categories': '-general,-sports',
        'keywords': "Ford",
        'sort': 'published_desc',
        "language": "en",
        'limit': 100,
    })

    conn.request('GET', '/v1/news?{}'.format(params))

    res = conn.getresponse()
    data = res.read()

    rs = json.loads(data.decode('utf-8'))
    print(json.dumps(rs, indent=4))


