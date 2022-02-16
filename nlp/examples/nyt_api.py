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
from nlp.utils import get_database
from nlp import get_configs_path


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

    keys_file = get_configs_path("keys.json")
    assert os.path.exists(keys_file), f"file:\n{keys_file}\ndoes not exist"
    with open(keys_file, "r") as f:
        keys = json.load(f)

    assert "nyt" in keys, f"'nyt' not found in:\n{keys_file}"
    api_key = keys['nyt']

    # ---
    # search api
    # ---

    # https://developer.nytimes.com/docs/articlesearch-product/1/overview

    base_url = "https://api.nytimes.com/"
    article_search_url = f"{base_url}svc/search/v2/articlesearch.json?"
    query = "q=election"
    query = 'fq=news_desk:("Sports") AND glocations:("NEW YORK CITY")'

    search_url = f'{article_search_url}{query}&api-key={api_key}'



