# read in locally stored articles
# get a list of company names to search for (taken from value chain data)
# avoid searching previously searched names - as this can be quite slow
# store company names ('set name') searched in article - to help avoid duplicating search in future
# store articles with at least one company name found

import re
import time
import json
import os
import sys
import pandas as pd
import numpy as np

from pymongo import UpdateOne
# from bson import ObjectId


try:
    # python package (supply_chain_extract) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


from supply_chain_extract.utils import get_database, get_list_from_tree, make_reg_tree
from supply_chain_extract import get_configs_path, get_data_path


if __name__ == "__main__":

    # TODO: investigate why not all articles have 'names_in_text' entry - they should
    # TODO: bulk upload for articles searched is very slow
    # TODO: move document with 'docname' in 'articles_searched' into own database

    pd.set_option("display.max_columns", 200)

    # ----
    # parameters
    # ----

    # search only for the parent names?
    parents_only = False

    # host / source
    host = 'www.reuters.com'
    # host = 'uk.reuters.com'

    # 'batch' size - how many full articles to read from file at once
    # - to avoid using too much memory in reading all
    batch_size = 1000

    # require cc_download_articles is in supply_chain_extract/data/
    # data_dir = get_data_path("cc_download_articles")
    # CHANGE THIS IF NEED BE!
    data_dir = "/home/buddy/workspace/datasets/commoncrawl/cc_download_articles"
    # assert os.path.exists(data_dir), f"data_dir:\n{data_dir}\ndoes not exist, get from "

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

    art_db = client["news_articles"]

    # --
    # get the company names from knowledge_base value chains
    # ---
    print(f"database names: {client.list_database_names()}")

    # database
    # find many - store in dataframe - is this now slow?
    t1 = time.time()
    # get the value chain data
    vc_col = client["knowledge_base"]['VCHAINS']
    vc = pd.DataFrame(list(vc_col.find(filter={})))
    t2 = time.time()
    print(f"time to read in all value chains: {t2-t1:.2f} seconds")

    # t1 = time.time()
    # vc_col.find_one()
    # art_db["articles"].find_one()
    # t2=time.time()
    # print(t2 -t1)

    # drop _id col
    vc.drop("_id", axis=1, inplace=True)

    # ---
    # select company names from data
    # ---

    if parents_only:
        company_names = vc["Parent Name"].unique()
    else:
        company_names = np.concatenate([vc["Parent Name"].unique(),
                                        vc.loc[vc['Confidence Score (%)'] > 0.9, "Company Name"].unique()])
        # there are some nans in the data, on the Company Name side?
        company_names = company_names[~pd.isnull(company_names)]
        company_names = np.unique(company_names)

    print(f"have: {len(company_names)} to search")

    # --
    # check if articles are unique
    # --

    #  is it possible multiple articles are found on multiple days, and if yes, can we skip some days?

    # get the 'route directory'
    root = os.path.join(data_dir, host)

    assert os.path.exists(root), f"root:\n{root}\ndoes not exist, check paths"

    # source assumed to be basename of root directory
    # - this is the case for cc_news data from hugging face
    # NOTE: this is redundant, just use 'host' ?
    # source = os.path.basename(root)
    # TODO: avoid using effectively duplicated variable here
    source = host

    # identify the files in the directory
    files = os.listdir(root)
    print("*" * 50)
    print(f"checking source:\n{source}")

    # find all json files
    json_files = [i for i in files if re.search("\.json$", i, re.IGNORECASE)]

    print(f"there are {len(json_files)} (json) articles found in\n{root}")

    # ----
    # read in previous company name 'set names'
    # ----

    # will store company names belonging to a set in a single document and then
    # refer to the 'set name' searched in article
    # - to avoid storing many long arrays in database which can slow things down

    set_names = art_db["articles_searched"].find_one(filter={"docname": "setnames"})
    # determine if the current set of names matches exactly another set previously used
    # to search files
    found_matching_set = False
    for k, v in set_names["sets"].items():
        # TODO: double check this logic, could use interest1d or union1d and then check size
        if np.in1d(company_names, v).all() & (len(company_names) == len(v)):
            print(f"current name set matched exactly: {k}")
            found_matching_set = True
            setname = k
            break

    if not found_matching_set:
        # get the largest number in set name
        largest_set_name = max([int(re.sub("\D", "", k)) for k in set_names["sets"].keys()])
        setname = f"set_names{largest_set_name+1}"

        # add the names in the current set to the 'setnames' doc
        set_names['sets'][setname] = company_names.tolist()
        doc_filter = {"docname": "setnames"}
        print(f"adding: {setname} to document containing:\n{json.dumps(doc_filter, indent=4)}\nunder 'sets'")
        # TODO: is there are more efficient way to update a nest field, than setting entire document? probably
        art_db["articles_searched"].update_one(filter={"docname": "setnames"},
                                               update={"$set": set_names},
                                               upsert=True)

    # get all the documents that have been previously searched for
    # some set of company names already
    # tmp = [ {k: v for k, v in d.items() if k != "_id"}
    #         for d in art_db["articles_searched"].find(filter={"docname": {"$exists": False}})]
    #
    # # https://stackoverflow.com/questions/58940431/unpack-list-of-dictionaries-in-python
    # # list to dict - want to unpack each dict in list into large dict
    # prev_searched = {}
    # for i in tmp:
    #     prev_searched.update(i)

    # ----
    # get information on files (json articles) on file system
    # ----

    # NOTE: this can be RAM intensive (up to 3.5GB per 550k articles)
    # - don't want to read in all (full) files at once
    # NOTE: can be slow if reading from HDD rather than SSD

    # only take a subset of columns - to save on memory
    # these will be used to determine which files to consider (effectively avoid duplicates)
    key_cols = ["date_download", "date_publish", "title", "json_file"]
    all_files = {}

    # increment over all the json_files found in the source
    print("reading in json files")
    bad_files = []
    for i, jf in enumerate(json_files):

        if i % 10000 == 0:
            print(f"{i} / {len(json_files)}")

        # read in article - very quick (~1ms)
        try:
            with open(os.path.join(root, jf), "r") as _:
                a = json.load(_)
        except json.decoder.JSONDecodeError:
            bad_files.append(jf)
            continue

        # if there is no 'maintext' skip
        if a['maintext'] is None:
            continue
        # select just a subset of keys / entries, enough to determine duplicates
        tmp = {k: a[k] for k in key_cols if k in a}
        tmp["json_file"] = jf
        all_files[jf] = tmp

    print(f"read in {len(all_files)}, there were issues reading in: {len(bad_files)}\n"
          f"(some files were skipped because their 'maintext' was empty)")

    # ------
    # get only the most recent versions of an article - try to remove duplicates (needs review)
    # ---------

    # store (key) data in dataframe - to perform analysis on
    df = pd.DataFrame([{k: a[k] for k in key_cols} for kk, a in all_files.items()])

    # convert dates to datetime64 objects
    date_cols = [i for i in df.columns if re.search("date", i)]
    for dc in date_cols:
        df[dc] = pd.to_datetime(df[dc])
        # get the day, [D], of the datetime
        df[dc+"_day"] = df[dc].values.astype('datetime64[D]')

    # sort data (not needed)
    df.sort_values(["date_publish", "date_download_day"], ascending=True, inplace=True)

    # for a given title, date_publish_day pair get the most recently published (using datetime)
    max_dp = pd.pivot_table(df, index=["title", "date_publish_day"], values=["date_publish"],
                            aggfunc="max").reset_index()

    # get the most recently downloaded for the most recently published (per day)
    pub_cols = max_dp.columns.values.tolist()
    # merge on date date_download - there could be multiple matches here
    recent_dl = max_dp.merge(df[pub_cols + ["date_download"]], on=pub_cols, how="left")
    max_dl = pd.pivot_table(recent_dl,
                            index=["title", "date_publish_day", "date_publish"], values=["date_download"],
                            aggfunc="max").reset_index()

    # get their corresponding json_files
    use_articles = max_dl.merge(df, on=max_dl.columns.values.tolist(), how="left")

    use_files = {jf: all_files[jf] for jf in use_articles["json_file"]}

    print(f"after taking the mostly recently published of\nan article per download date have: {len(use_files)} articles")

    # ----
    # Search articles for company names
    # ----

    # NOTE: articles_searched should have (after running once) documents containing
    # {"json_file": *, "sets_searched": ["set_name<#>", "set_name<#2>"]}


    # articles that have been previously searched - json_file as key and sets_searched as values
    # # - also include id
    article_searched_setname = {a["json_file"]: {"sets_searched": a["sets_searched"],
                                                 "_id": a["_id"]}
                                for a in art_db["articles_searched"].find({"json_file": {"$exists": True}})}


    # combine the setname searched per article - so can use as a key later
    # - combine as a string using : as a separator
    for k in article_searched_setname.keys():
        article_searched_setname[k]["comb_set_name"] = ":".join(article_searched_setname[k]["sets_searched"])

    # get the unique combination of sets names
    unique_set_name_comb = np.unique([v["comb_set_name"]
                                      for k, v in article_searched_setname.items()])

    # for search of these combined set names get the
    # unique company names search - by combining all names in sets into a single array
    # get the previously searched set names and their values
    setname_details = art_db["articles_searched"].find_one({"docname": "setnames"})
    setname_details = setname_details['sets']

    set_name_comb_dict = {snc: np.unique(np.concatenate([setname_details[sn]
                                                         for sn in snc.split(":")]))
                          for snc in unique_set_name_comb}

    # make a dict with keys being combined set names (from above) with
    # values being the values to search which were not in the (combined) set
    comp_not_in_set = {k: company_names[~np.in1d(company_names, v)]
                       for k, v in set_name_comb_dict.items()}

    # for each of the above make a dict containing a 'regular expression tree'
    # use to search article for remaining articles
    rtree_dict = {}
    for k, cnames in comp_not_in_set.items():
        # if there are no company names to search - as some combination of previous searched
        # have covered this set, then set to None (will skip these)
        if len(cnames) == 0:
            rtree_dict[k] = None
        else:
            max_node = np.floor(1 + np.log(len(cnames)/50) / np.log(2))
            rtree_dict[k] = make_reg_tree(c=cnames.tolist(), max_node=max_node)
    # add rtree for the current set_name - to be used for articles that have not been searched at all
    # - thus have no previous setnames
    max_node = np.floor(1 + np.log(len(company_names)/50) / np.log(2))
    rtree_dict["current_set"] = make_reg_tree(c=company_names.tolist(), max_node=max_node)

    # get all the file names (articles) - with duplicates removed above
    all_files_names = np.array(list(use_files.keys()))

    # find the articles that have already been searched with this given set
    prev_searched_with_setname = [k for k, v in article_searched_setname.items()
                                  if setname in v["sets_searched"]]

    # exclude those that have already been searched with given set name
    files_to_search = all_files_names[~np.in1d(all_files_names, prev_searched_with_setname)]

    # will read the files in 'batches' to avoid using too much memory
    num_batches = len(files_to_search) // batch_size
    # TODO: double check the edge cases with this, i.e. data is evenly divisible by batches, dont add one below?
    batch_data = {i: files_to_search[i*batch_size: (i+1) * batch_size]
                  for i in range(num_batches + 1)}

    print(f"there are: {len(files_to_search)} articles to be searched "
          f"with company name set: {setname}")

    # store articles that contain any name in company_names in a list
    articles_with_names = []

    # get the json_file and 'names_in_text' of all the 'articles' collection
    # - use a projection (the second argument) in find(...) can reduce arguments that come back
    # - and thus network usage - plus it's quicker
    tmp = {f["json_file"]: f.get("names_in_text", None)
           for f in art_db["articles"].find({}, {"json_file": 1, "names_in_text": 1})}

    # get a list of the article json files
    art_jf = [jf for jf in tmp.keys()]

    # get the names_in_text already found for each file
    nint = {k: v
            for k, v in tmp.items()
            if v is not None}

    # keep track of articles to update
    found_count = 0
    t0 = time.perf_counter()
    td_ave = None
    # TODO: create 'batches' of articles to read in at a time

    # increment over data
    for i, bd in batch_data.items():
        # read in the files
        batch_files = {}
        print("reading in files for batch")
        for jf in bd:
            try:
                with open(os.path.join(root, jf), "r") as _:
                    a = json.load(_)
            except json.decoder.JSONDecodeError:
                bad_files.append(jf)
                continue
            a["json_file"] = jf

            batch_files[jf] = a

        # print progress information - after first batch
        if i > 0:
            t1 = time.perf_counter()
            td = t1 - t0
            t0 = t1
            td_ave = td if td_ave is None else (1 - 0.975) * td + 0.975 * td_ave
            # double check this
            remaining_time = td_ave * (len(batch_data) - i -1)# (len(unsearched) - i*batch_size) / 1000
            # TODO: these time estimates are incorrect
            print("-" * 20)
            print(f"{i}/{len(batch_data)} - {100 * i/len(batch_data):.2f}%% "
                  f"time to run: {td:.2f}s, "
                  f"estimated remaining: {remaining_time:.2f}s")

        # increment over the files in the batch
        counter = 0
        # store in a list articles for bulk insert and update
        bulk_insert = []
        bulk_update = []

        # for each file in batch read
        for j, jf in enumerate(batch_files.keys()):

            # TODO: here want to be able to get the previously searched company names
            #  - NOTE: if the number of company names isn't too large (~500) it will run in a reasonable amount of time
            #  - looks like there might be some O(n^2) element in the calculation

            # get the articles
            a = batch_files[jf]

            # create a list to track the names in the articles
            names_found_in_article = []

            # get the previously searched sets -
            if jf in article_searched_setname:
                # get the previously searched combined set name
                csn = article_searched_setname[jf]["comb_set_name"]
            # otherwise, the articles was not searched previously
            else:
                # so use the 'current_set' - for getting the regression tree
                csn = "current_set"

            # HACK: if rtree would be None - i.e. company names were already found
            # in previous set(s) then just skip
            if rtree_dict[csn] is None:
                continue

            # determine a subset of company names to search for - to avoid for searching all names
            # - this could return an empty list
            cnames = get_list_from_tree(text=a["maintext"],
                                        rtree=rtree_dict[csn])

            if len(cnames) > 0:
                counter += 1
                # print(f"{j}: {counter/(j+1):.2f}")

            # search for specific names in article
            for cn in cnames:
                if re.search(cn, a["maintext"]):
                    names_found_in_article.append(cn)

            # if found some names in article : add as value, and add to articles
            if len(names_found_in_article) > 0:

                # does the article already exist in the database?
                # - if so just update the 'names_in_text'
                if jf in art_jf:
                    # check if the names already exist
                    update_names = np.in1d(names_found_in_article, nint.get(jf, [])).all()

                    if update_names:
                        bulk_update += [UpdateOne(filter={"json_file": jf},
                                                  update={'$addToSet':
                                                              {"names_in_text": {"$each": names_found_in_article}}
                                                          })]
                # otherwise, add to list bulk insert the article
                else:
                    a["names_in_text"] = names_found_in_article
                    articles_with_names.append(a)
                    bulk_insert.append(a)

        # end of batch - bulk insert any
        t0_ = time.perf_counter()
        if len(bulk_insert):
            print(f"bulk inserting (new) articles: {len(bulk_insert)}")
            art_db["articles"].insert_many(documents=bulk_insert)

        # ---
        # bulk update articles
        # ---

        if len(bulk_update):
            print(f"bulk updating {len(bulk_update)} articles previously found articles")
            art_db["articles"].bulk_write(bulk_update)

        # ---
        # bulk update "articles_searched" documents
        # ---

        # NOTE: this is slow - but kind of unavoidable?
        # - should avoid searches in the future
        # update / insert setname for articles searched for in batch
        batch_jf_files = list(batch_files.keys())

        # get the _id for the json files (will exist if already searched)
        # - use to add to 'sets_searched' array
        # ref: https://pymongo.readthedocs.io/en/stable/examples/bulk.html
        # TODO: confirm this is working as expected - i.e. adding to existing
        bulk_write = [UpdateOne(filter={"_id": article_searched_setname[bjf]["_id"]},
                                update={
                                    "$addToSet": {"sets_searched": setname}
                                },
                                upsert=False)
                      for bjf in batch_jf_files
                      if bjf in article_searched_setname]

        # if the articles have not been previously searched - just insert them
        insert_many_art_searched = [{"json_file": bjf, "sets_searched": [setname]}
                                    for bjf in batch_jf_files
                                    if bjf not in article_searched_setname]

        if len(bulk_write):
            art_db["articles_searched"].bulk_write(bulk_write)
        # is this needed?
        if len(insert_many_art_searched):
            art_db["articles_searched"].insert_many(insert_many_art_searched)

        t1_ = time.perf_counter()

        print(f"time for (bulk) inserts and updates: {t1_ - t0_:.2f} seconds")





    # TODO: for all the unsearched articles (just searched) need to update they have been
    #  - searched in "articles_searched" collection

    # TODO: write articles with any names to
    # TODO: insert many - exclude any that have already be found
    # TODO: figure out how to use find for multiple matches
    # art_db["articles_with_names"].update_many()
    #
    # print(f"there were: {len(articles_with_names)} articles found containing at least one company name")
    #
    # existing_article = [f["json_file"] for f in art_db["articles"].find()]
    #
    # # find the articles that have not been updated
    # new_articles = [a for a in articles_with_names if a["json_file"] not in existing_article]
    # print(f"adding: {len(new_articles)} to 'articles' collection")
    # art_db["articles"].insert_many(documents=new_articles)
    #
    # # for the articles that are already exist, update names found
    # old_articles = [a for a in articles_with_names if a["json_file"] in existing_article]
    # oa_jf = [oa["json_file"] for oa in old_articles]
    #
    # # read in the articles that already exist that have been searched
    # # - to determine if need to add names
    # prev_articles = {f["json_file"]: f for f in art_db["articles"].find({"json_file": {"$in": oa_jf}})}
    #
    # # keep track of articles where more names were found
    # added_to_existing = 0
    # for oa in old_articles:
    #     # NOTE: this can be quite slow - perhaps consider reading in all articles at once
    #     # a = art_db["articles"].find_one(filter={"json_file": oa["json_file"]})
    #     a = prev_articles[oa["json_file"]]
    #
    #     # all
    #     nit = a.get("names_in_text", [])
    #     if len(nit) == 0:
    #         pass
    #         # print(f"no names in text for article: {a['json_file']}, will add to them")
    #     # if there were names found that are not in 'names_in_text' array, add:
    #     if not np.in1d(oa["names_in_text"],  nit).all():
    #         added_to_existing += 1
    #         # NOTE: updating one makes an assumption there is only one entry to update
    #         art_db["articles"].update_one(filter={"json_file": oa["json_file"]},
    #                                       update={'$addToSet': {
    #                                           "names_in_text": {"$each": oa["names_in_text"]}
    #                                       }},
    #                                       upsert=True)
    #
    # print(f"add names to: {added_to_existing} existing articles (in database)")


    # tmp = {a["json_file"]: a for a in articles_with_names}
    # with open(get_data_path("articles_with_names_tmp1.json"), "w") as f:
    #     json.dump(tmp, f)

    # ----
    # analysis: difference between publish_day and download_day
    # ----
    #
    # df["dl_pub_diff"] = (df["date_download_day"].values.astype("datetime64[D]") - \
    #                      df["date_publish_day"].values.astype("datetime64[D]")).astype(int)
    # dd = df.copy(True)
    # dd.sort_values("dl_pub_diff", ascending=False, inplace=True)
    # dd = dd.loc[~(pd.isnull(dd["date_download_day"]) | pd.isnull(dd["date_publish_day"]))]
    # print(f"fraction of articles with download_day == publish_day: {100 * (dd['dl_pub_diff'] ==  0).mean():.2f}\%")
    #
    # ----
    # get download distribution - from files that are used
    # ---
    #
    # uf = dd.loc[dd["json_file"].isin(list(use_files.keys()))].copy(True)
    # uf["dl_hour"] = [int(pd.to_datetime(i).strftime("%H")) for i in uf["date_download"].values]
    #
    # # most titles appear to fetched during mid day (14)
    # titles_per_hour = pd.pivot_table(uf, index="dl_hour", values="title", aggfunc="count")
    #
    #
    #
    # # ----
    # # search for company names in articles
    # # ----
    #
    # # keep track of the number of companys found in source
    # search_names = company_names
    #
    # # store the json files where a name was found
    # nf = {sn: [] for sn in search_names}
    #
    # # HARDCODED: previously found
    # # this needs to be refactored - should be getting from database
    # with open(get_data_path("files_searched.json"), "r") as f:
    #     pfound = json.load(f)
    #
    # prev_found = pfound["files"]
    #
    # # keep track of all files search
    # all_files_searched = np.array([])
    # # TODO: review this
    # for j, cn in enumerate(search_names):
    #     if j % 10 == 0:
    #         print(f"on name: {j}/{len(search_names)}")
    #
    #     # for given name - find the files that were previously searched
    #     # prev_found = []
    #     # for pf in art_db["files_searched"].find(filter={"names": {"$in": [cn]}}):
    #     #     prev_found += pf["files"]
    #
    #     # print(f"previously search for {cn} in {len(prev_found)} files, will exclude those")
    #
    #     # get the files to search
    #     search_files = np.array(list(use_files.keys()))
    #     # exclude those previous found - assumed all names in set were searched through the same files
    #     search_files = search_files[~np.in1d(search_files, prev_found)]
    #     # print(f"have: {len(search_names)} files to search")
    #
    #     # keep track of all files searched during this iteration
    #     all_files_searched = np.union1d(all_files_searched, search_files)
    #     # increment over the files to search
    #     for i, _ in enumerate(search_files):
    #         # if i % 1000 == 0:
    #         #     print(f"{i} / {len(search_files)}")
    #         k, a = _, use_files[_]
    #
    #         main_text = a["maintext"]
    #         if main_text is None:
    #             continue
    #         # search main text
    #         if re.search(cn, a["maintext"]):
    #             # print("-"*10)
    #             # print(f"{cn} - found in:\n{a['title']}\nfrom: {source}")
    #             nf[cn] += [a["json_file"]]
    #
    # # store the set of names AND the json_files searched
    # files_searched = {
    #     "files": all_files_searched.tolist(),
    #     "names": search_names.tolist()
    # }
    #
    # # TODO: need to refactor this! think of how to store differently
    #
    # # for the given set of names searched add to the files searched
    # # - to avoid searching through again
    # # art_db["files_searched"].update_one(filter={"names": files_searched["names"]},
    # #                                     update={"$addToSet": {"files": {"$each": files_searched["files"]}}},
    # #                                     upsert=True)
    # # art_db.drop("files_searched")
    # #
    # # fs = list(art_db["files_searched"].find())
    #
    # article_count = pd.DataFrame([{"name":k, "count": len(v)} for k,v in nf.items()])
    #
    # # ---
    # # for each company - record suppliers / customers mentioned in same article
    #
    # parent_company = []
    # found_count = 0
    #
    # for i, _ in enumerate(nf.items()):
    #     k, json_files = _
    #
    #     if i % 100 == 0:
    #         print(f"{i} / {len(nf)}")
    #
    #     # print("-"*20)
    #     # print(f"{k} was found in {len(json_files)} articles")
    #     sc = vc.loc[vc["Parent Name"] == k]
    #
    #     # suppliers
    #     suppliers = sc.loc[sc["Relationship"] == "Supplier", "Company Name"].unique()
    #     # remove missing names - why / where does this occur?
    #     suppliers = suppliers[~pd.isnull(suppliers)]
    #     # print(f"checking if any of the {len(suppliers)} are also mentioned")
    #
    #     # read in each file
    #     for jf in json_files:
    #         a = use_files[jf]
    #
    #         for cn in suppliers:
    #             main_text = a["maintext"]
    #
    #             if re.search(cn, a["maintext"]):
    #                 # print("-"*10)
    #                 # print(f"{cn} - found in:\n{a['title']}\nfrom: {source}")
    #                 found_count += 1
    #                 # print(found_count)
    #                 parent_company += [{"Parent Name": k, "Company Name": cn,"title": a["title"], "json_file": jf}]
    #                 # nf[cn] += [a["json_file"]]
    #
    # print(f"found: {found_count} articles with a mention of a Parent Name and a Company (could be duplicates articles)")
    #
    # pc = pd.DataFrame(parent_company)
    #
    # print(f"found: {len(pc['json_file'].unique())} unique articles mentioning at least 1 Parent Name and a Company Name (Supplier)")
    #
    # # -----
    # # write articles to database
    # # -----
    #
    # articles = art_db["articles"]
    #
    # new_articles_add = 0
    #
    #
    # for jf in pc["json_file"].unique():
    #
    #     # get the article
    #     a = all_files[jf]
    #
    #     # get the names found in given file
    #     pc_ = pc.loc[pc["json_file"] == jf]
    #     all_names = np.concatenate([pc_["Parent Name"].values, pc_["Company Name"].values])
    #     all_names = np.unique(all_names)
    #
    #     # a["names_in_text"] = all_names.tolist()
    #
    #     # see if file has already been found
    #     # TODO: is there a better way of doing this?
    #     art_found = articles.find_one(filter={"json_file": jf})
    #
    #     if art_found is None:
    #         # print("adding article")
    #         _id = articles.insert_one(document=a)
    #         art_id = _id.inserted_id
    #         new_articles_add += 1
    #     # otherwise, get the article _id
    #     else:
    #         art_id = art_found["_id"]
    #
    #     # add to the names_in_text array (addToSet won't add any already there)
    #     articles.update_one(filter={"_id": art_id},
    #                         update={'$addToSet': {'names_in_text': {"$each": all_names.tolist()}}},
    #                         upsert=True)
    #
    # # find all articles that have a names_in_text field
    # articles_with_names = list(articles.find(filter={"names_in_text": {"$exists": True}}))
    #
    #
    # print(f"there are now: {len(articles_with_names)} articles that mention a parent and at least one supplier")
    #
    # tmp = {str(a["_id"]): a for a in articles_with_names}
    #
    # for k in tmp.keys():
    #     tmp[k].pop("_id")
    #
    # with open(get_data_path("articles_with_names.json"), "w") as f:
    #     json.dump(tmp, f, indent=4)



    # previously searched files - saved locally - this is a one off
    # with open(get_data_path("files_searched.json"), "r+") as f:
    #     files_searched = json.load(f)
    # cnames = files_searched["names"]
    # articles = files_searched["files"]
    # doc = {
    #     "docname": "setnames",
    #     "sets": {
    #         "set_names1": cnames
    #     }
    # }
    # art_db["articles_searched"].update_one(filter={"docname": "setnames"},
    #                                        update={"$set": doc},
    #                                        upsert=True)
    # # for each article - state
    # tmp = [{k: ["set_names1"]} for k in articles]
    # art_db["articles_searched"].insert_many(tmp)
    # for k in articles:
    #     art_db["articles_searched"].insert_one({k: ["set_names1"]})

    # tmp = {k: ["set_name1"] for k in all_files.keys()}
    # with open(get_data_path("article_setname.json"), "w") as f:
    #     json.dump(tmp, f)


    # DON'T DO THIS - using 'json_file' as an index turns out to be memory intensive!
    # - used "_id" instead - as it's an index (and low memory)
    # # make an index of the json_file  in articles to search
    # #  - this was done once to help improve bulk_write to "articles_searched"
    # import pymongo
    # art_db["articles_searched"].create_index([("json_file", pymongo.ASCENDING)])
    # art_db["articles_searched"].drop_index([("json_file", pymongo.ASCENDING)])

    # HACK: some articles searched were added without 'json_file' field - drop those
    # art_db["articles_searched"].delete_many(filter={"json_file": {"$exists": False}, "sets_searched": {"$exists": True}})


    # tmp = [(a["json_file"], a["_id"])
    #        for a in art_db["articles_searched"].find({"json_file": {"$exists": True}}) ]
    # asdf = pd.DataFrame(tmp, columns=["json_file", "_id"])
    #
    # asc = pd.pivot_table(asdf,
    #                index=["json_file"],
    #                values="_id",
    #                aggfunc="count")
    #
    # asc.sort_values("_id", ascending=False)

    # art_db["articles_searched"].find_one({"json_file": {"$exists": True}},
    #                                      {"json_file": 0,
    #                                       "sets_searched": 1})
    # art_db["articles_searched"].find_one({"json_file": {"$exists": True}}, {"sets_searched": 1})
