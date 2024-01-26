
import newsplease

import re
import time
import json
import os
import sys
import logging
import hashlib
import pandas as pd
import numpy as np
import datetime

from bson import ObjectId
from newsplease.crawler import commoncrawl_crawler as commoncrawl_crawler
from newsplease.crawler.commoncrawl_extractor import CommonCrawlExtractor
from newsplease.crawler.commoncrawl_crawler import __start_commoncrawl_extractor

try:
    # python package (supply_chain_extract) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


from supply_chain_extract import get_parent_path, get_configs_path
from supply_chain_extract.utils import get_database


def __setup__():
    """
    Setup
    :return:
    """
    os.makedirs(my_local_download_dir_article, exist_ok=True)


def callback_on_warc_completed(*args, **kwargs):
    """
    This function will be invoked for each WARC file that was processed completely. Parameters represent total values,
    i.e., cumulated over all all previously processed WARC files.
    :param warc_path:
    :param counter_article_passed:
    :param counter_article_discarded:
    :param counter_article_error:
    :param counter_article_total:
    :param counter_warc_processed:
    :return:
    """
    pass


def __get_pretty_filepath(path, article):
    """
    Pretty might be an euphemism, but this function tries to avoid too long filenames, while keeping some structure.
    :param path:
    :param name:
    :return:
    """
    short_filename = hashlib.sha256(article.filename.encode()).hexdigest()
    sub_dir = article.source_domain
    final_path = os.path.join(path, sub_dir)
    os.makedirs(final_path, exist_ok=True)
    return os.path.join(final_path, short_filename + '.json')

def on_valid_article_extracted(article):
    """
    This function will be invoked for each article that was extracted successfully from the archived data and that
    satisfies the filter criteria.
    :param article:
    :return:
    """
    # do whatever you need to do with the article (e.g., save it to disk, store it in ElasticSearch, etc.)
    with open(__get_pretty_filepath(my_local_download_dir_article, article), 'w', encoding='utf-8') as outfile:
        if my_json_export_style == 0:
            json.dump(article.__dict__, outfile, default=str, separators=(',', ':'), ensure_ascii=False)
        elif my_json_export_style == 1:
            json.dump(article.__dict__, outfile, default=str, indent=4, sort_keys=True, ensure_ascii=False)
        # ...


def get_config(sysargv, argpos=1, default="commoncrawl.json", verbose=True):
    """allow config to be passed in as argument to script"""

    conf_found = False
    try:
        conf_file = sysargv[argpos]

        # check file exists
        file_exists = False
        if os.path.exists(conf_file):
            if verbose:
                print(f"conf_file:\n{conf_file}\nexists")
            file_exists = True
        else:
            if verbose:
                print(f"file: {conf_file}, does not exist, checking cwd")
            conf_file = os.path.join(os.getcwd(), conf_file)
            if os.path.exists(conf_file):
                if verbose:
                    print(f"conf_file:\n{conf_file}\nexists")
                file_exists = True

        # if the file exists: make sure it is a json file
        if file_exists:
            if re.search("\.json$", conf_file, re.IGNORECASE):
                if verbose:
                    print("conf_file is the right type (json)")
                conf_found = True
            else:
                if verbose:
                    print("conf_file is NOT the right type (json)")

    except IndexError:
        if verbose:
            print("index error reading conf from arg")
        conf_found = False

    if not conf_found:
        if verbose:
            print(f"using default configuration (from package): {default}")
        conf_file = get_configs_path(default)
        assert os.path.exists(conf_file), f"conf_file:\n{conf_file}\ndoes not exist"

    # ---
    # read in config
    # ---

    with open(conf_file, "r") as f:
        cc_config = json.load(f)

    return cc_config


def get_unfetched_commoncrawl_files(art_db, use_dates=None):
    filter = {}
    projection = {"name": 1, "fetched": 1, "date": 1}
    _ = art_db["common_crawl_files"].find(filter,  projection)
    ccf = pd.DataFrame(list(_))
    ccf["date"] = pd.to_datetime(ccf["date"])

    # select subset of dates - Last Update Date
    if use_dates is not None:
        ccf = ccf.loc[ccf["date"].isin(use_dates)]

    # consider only those files that have not been fetched yet
    ccf = ccf.loc[~ccf["fetched"]]
    return ccf


if __name__ == "__main__":

    # TODO: clean up this file - remove
    # TODO: require data_dir exists
    # TODO: allow parameters to be set from config
    # TODO: all config to be passed in as an argument instead of reading from package
    # TODO: wrap up to allow for better try, except

    pd.set_option("display.max_columns", 200)
    # ----
    # common crawl config
    # ----

    # read in configuration file
    cc_config = get_config(sysargv=sys.argv, argpos=1, verbose=True)

    print("*" * 50)
    print("using config file")
    print(json.dumps(cc_config, indent=4))

    # use subset of all date (i.e. only the 'updates' found in the value chain data)
    use_subset_of_dates = cc_config.get("only_use_dates_from_supply_chain_data", False)


    # download_dir = cc_config.get("download_dir", None)
    #
    #
    # if download_dir is None:
    #     print("download_dir is None, will use 'data' dir in parent directory of package")
    #     download_dir = os.path.join(get_parent_path(), "data")
    # print(f"using download_dir:\n{download_dir}")
    #
    # os.makedirs(download_dir, exist_ok=True)

    # ----
    # connect to database to determine which dates to download
    # ----

    # get credentials
    with open(get_configs_path("mongo.json"), "r+") as f:
        mdb_cred = json.load(f)

    # get mongodb client - for connections
    client = get_database(username=mdb_cred["username"],
                          password=mdb_cred["password"],
                          clustername=mdb_cred["cluster_name"])
    art_db = client["news_articles"]

    # get all entries from value chain data
    # TODO: select only relevant 'columns' using a 'projection' in the find()
    t0 = time.time()
    vc = pd.DataFrame(list(client["knowledge_base"]["VCHAINS"].find(filter={})))
    t1 = time.time()

    # ----
    # filter dates to fetch - i.e. get update_dates
    # ----

    if use_subset_of_dates:
        # TODO: let these selection criteria be specified in config
        # filter only those that are after
        vc["Last Update Date"] = pd.to_datetime(vc["Last Update Date"])
        # CC-NEWS available from August 2016?
        vc = vc.loc[vc["Last Update Date"] >= "2016-10-01"]
        # take only the high confidence entries
        vc = vc.loc[vc["Confidence Score (%)"] > 0.95]

        # get the update dates
        update_dates = vc["Last Update Date"].unique()
    # if update_dates is set to None will search for all dates
    else:
        update_dates = None

    # ----
    # get commoncrawl files that can be downloaded
    # ----

    # get the commoncrawl files that have not been fetched
    ccf = get_unfetched_commoncrawl_files(art_db, use_dates=update_dates)

    # # remove the previously fetched dates
    # fetched_dates = list(["fetched_dates"].find())
    # fetched_dates = [f["date"] for f in fetched_dates]
    # fetched_dates = np.array(fetched_dates).astype("datetime64[ns]")
    #
    # if len(fetched_dates) == 0:
    #     print("no dates previously fetched")

    # remove previously fetched
    # update_dates = update_dates[~np.in1d(update_dates, fetched_dates)]

    counter = 0
    max_count = len(ccf)
    # TODO: this can be tidied up
    while len(ccf) > 0:
        counter += 1

        # get the current date
        todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # pick a random date - to reduce chances of two people fetching same
        # date at same time

        # ---
        # convert date to fetched to datetime
        # ---
        #
        # if update_dates is None:
        #     ran_date = np.random.choice(ccf['date'].values, 1)
        # else:
        #     ran_date = np.random.choice(update_dates, 1)
        # # TODO: get stackoverflow ref
        # start_date = datetime.datetime.utcfromtimestamp(int(ran_date)/1e9)
        #
        # date_str = pd.to_datetime(update_dates[0]).strftime("%Y-%m-%d")
        #
        # # --
        # # add / update document for this date specifying started, but did not finish
        # # --
        # # TODO: include the names/websites which were searched/filtered for?
        # doc = {"date": date_str, "started": True, "finished": False}
        # client["news_articles"]["fetched_dates"].update_one(filter={"date": date_str},
        #                                                     update={"$set": doc},
        #                                                     upsert=True)

        # this was taken from new-please repo: news-please/newsplease/examples/commoncrawl_crawler.py
        # TODO: have the below parameters specified in a config
        ############ YOUR CONFIG ############
        # download dir for warc files
        my_local_download_dir_warc = cc_config["my_local_download_dir_warc"]
        os.makedirs(my_local_download_dir_warc, exist_ok=True)

        # download dir for articles
        my_local_download_dir_article = cc_config["my_local_download_dir_article"]
        os.makedirs(my_local_download_dir_warc, exist_ok=True)

        # hosts (if None or empty list, any host is OK)
        my_filter_valid_hosts = cc_config.get("my_filter_valid_hosts",
                                              ["www.reuters.com", "uk.reuters.com"])
        print("using my_filter_valid_hosts")
        print(my_filter_valid_hosts)

        # start date (if None, any date is OK as start date), as datetime
        my_filter_start_date = None # datetime.datetime(2016, 1, 1)

        # end date (if None, any date is OK as end date), as datetime
        my_filter_end_date = None  # datetime.datetime(2016, 12, 31)

        # Only .warc files published within [my_warc_files_start_date, my_warc_files_end_date) will be downloaded.
        # Note that the date a warc file has been published does not imply it contains only news
        # articles from that date. Instead, you must assume that the warc file can contain articles
        # from ANY time before the warc file was published, e.g., a warc file published in August 2020
        # may contain news articles from December 2016.
        # my_warc_files_start_date = start_date  # example: datetime.datetime(2020, 3, 1)
        # my_warc_files_end_date = start_date + datetime.timedelta(1) # example: datetime.datetime(2020, 3, 2)
        # if date filtering is strict and news-please could not detect the date of an article, the article will be discarded
        my_filter_strict_date = cc_config.get("my_filter_strict_date", True)
        # if True, the script checks whether a file has been downloaded already and uses that file instead of downloading
        # again. Note that there is no check whether the file has been downloaded completely or is valid!
        my_reuse_previously_downloaded_files = cc_config.get("my_reuse_previously_downloaded_files", True)
        # continue after error
        my_continue_after_error = cc_config.get("my_continue_after_error", True)
        # show the progress of downloading the WARC files
        my_show_download_progress = cc_config.get("my_show_download_progress", True)
        # log_level
        my_log_level = logging.INFO
        # json export style
        my_json_export_style = cc_config.get("my_json_export_style", 1)  # 0 (minimize), 1 (pretty)
        # number of extraction processes
        my_number_of_extraction_processes = cc_config.get("my_number_of_extraction_processes", 1)
        # if True, the WARC file will be deleted after all articles have been extracted from it
        my_delete_warc_after_extraction = cc_config.get("my_delete_warc_after_extraction", True)
        # if True, will continue extraction from the latest fully downloaded but not fully extracted WARC files and then
        # crawling new WARC files. This assumes that the filter criteria have not been changed since the previous run!
        my_continue_process = cc_config.get("my_continue_process", True)
        # if True, will crawl and extract main image of each article. Note that the WARC files
        # do not contain any images, so that news-please will crawl the current image from
        # the articles online webpage, if this option is enabled.
        my_fetch_images = cc_config.get("my_fetch_images", False)
        ############ END YOUR CONFIG #########


        # take from newsplease/examples
        # __setup__()
        # try:
        #     commoncrawl_crawler.crawl_from_commoncrawl(on_valid_article_extracted,
        #                                                callback_on_warc_completed=callback_on_warc_completed,
        #                                                valid_hosts=my_filter_valid_hosts,
        #                                                start_date=my_filter_start_date,
        #                                                end_date=my_filter_end_date,
        #                                                warc_files_start_date=my_warc_files_start_date,
        #                                                warc_files_end_date=my_warc_files_end_date,
        #                                                strict_date=my_filter_strict_date,
        #                                                reuse_previously_downloaded_files=my_reuse_previously_downloaded_files,
        #                                                local_download_dir_warc=my_local_download_dir_warc,
        #                                                continue_after_error=my_continue_after_error,
        #                                                show_download_progress=my_show_download_progress,
        #                                                number_of_extraction_processes=my_number_of_extraction_processes,
        #                                                log_level=my_log_level,
        #                                                delete_warc_after_extraction=my_delete_warc_after_extraction,
        #                                                continue_process=my_continue_process,
        #                                                fetch_images=my_fetch_images)
        # except FileNotFoundError:
        #     print("bad date?")
        #     print(start_date)
        #     doc["error_message"] = "FileNotFoundError"


        # select a random(?) warc file
        warc_file = np.random.choice(ccf["name"].values, 1)[0]
        base_url = 'https://commoncrawl.s3.amazonaws.com/'
        warc_download_url = base_url + warc_file

        # write to mongo this warc file has* been fetched
        art_db["common_crawl_files"].update_one(filter={"name": warc_file},
                                                update={"$set": {"fetched": True}})

        try:
            __log_pathname_fully_extracted_warcs = None
            extractor_cls=CommonCrawlExtractor
            # for warc_download_url in warc_download_urls:

            __start_commoncrawl_extractor(warc_download_url,
                                          callback_on_article_extracted=on_valid_article_extracted,
                                          callback_on_warc_completed=callback_on_warc_completed,
                                          valid_hosts=my_filter_valid_hosts,
                                          start_date=my_filter_start_date,
                                          end_date=my_filter_end_date,
                                          strict_date=my_filter_strict_date,
                                          reuse_previously_downloaded_files=my_reuse_previously_downloaded_files,
                                          local_download_dir_warc=my_local_download_dir_warc,
                                          continue_after_error=my_continue_after_error,
                                          show_download_progress=my_show_download_progress,
                                          log_level=my_log_level,
                                          delete_warc_after_extraction=my_delete_warc_after_extraction,
                                          log_pathname_fully_extracted_warcs=__log_pathname_fully_extracted_warcs,
                                          extractor_cls=extractor_cls,
                                          fetch_images=my_fetch_images)

            # add the date fetched
            art_db["common_crawl_files"].update_one(filter={"name": warc_file},
                                                    update={"$set": {"fetched_date": todays_date}})

        except Exception as e:
            print("-"*50)
            print("-"*50)

            print(f"error occured\n will state current warc_file: {warc_file} fetched=False")
            print(e)

            # write to mongo this warc file has* been fetched
            art_db["common_crawl_files"].update_one(filter={"name": warc_file},
                                                    update={"$set": {"fetched": False}})

            # assert False, "broke on purpose (remove this line)"

        # -----
        # update doc in database indicating finished searching for that date
        # -----

        # update
        # art_db["common_crawl_files"].update_one(filter={"_id": doc["_id"]},
        #                                         update={"$set": {"fetched": True}})


        # # TODO: add the in
        # doc["finished"] = True
        # client["news_articles"]["fetched_dates"].update_one(filter={"date": date_str},
        #                                                    update={"$set": doc},
        #                                                    upsert=True)

        # # remove the previously fetched dates
        # fetched_dates = list(client["news_articles"]["fetched_dates"].find())
        # fetched_dates = [f["date"] for f in fetched_dates]
        # fetched_dates = np.array(fetched_dates).astype("datetime64[ns]")
        #
        # # remove previously fetched
        # update_dates = update_dates[~np.in1d(update_dates, fetched_dates)]

        # get the commoncrawl files that have not been fetched
        ccf = get_unfetched_commoncrawl_files(art_db, use_dates=update_dates)

        # if counter happens to exceed max count then set ccf to [] - which will end loop
        if counter >= max_count:
            ccf = []

    print("FINSIHED!")
    client.close()

