# script to extract sentences from articles
# - the aim will be extract single 'sentences' from an article
# - store in dict / json with 'subject', 'object' and 'relationship' stated
# - also provide article name and source


import json
import os
import re
import time
import gzip
import itertools


import sys
import numpy as np
import pandas as pd

import spacy



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


import pandas as pd


def find_all_articles_with_name(articles, name):
    return [v for k, v in articles.items() if re.search(name, v["maintext"])]



def get_start_end(a, b, aname="a", bname="b"):

    assert len(np.intersect1d(a, b)) == 0, f"some elements found in both 'a' and 'b'"

    c = np.sort(np.concatenate([a, b]))
    is_a = np.in1d(c, a)

    res = []
    names = []
    for i in range(len(c)-1):
        if is_a[i] != is_a[i+1]:
            res.append((c[i], c[i+1]))
            if is_a[i]:
                names.append((aname, bname))
            else:
                names.append((bname, aname))

    return res, names


def correct_names_in_main_text(articles):
    """
    there are instances where two companies with overlapping names with listed
    to be in the same article
    this function will return a dict containing the actual names in article
    """

    # get all names
    all_names = np.unique(np.concatenate([v["names_in_text"]
                                          for k, v in articles.items()
                                          if "names_in_text" in v]))
    # add a dash in names to have more explicit matching (?)
    dash_names = {n: re.sub(" ", "-", n) for n in all_names}

    # for each article replace the name with the dashed name
    # - the idea is this will help with tokenizing
    names_in_text_dict = {}
    bad_articles = {}
    for k in keys:
        a = articles[k]
        anames = np.array(a["names_in_text"])
        # dnames = np.array([dash_names[an] for an in anames])
        # replace the longest matching first - to deal with some names
        # being listed because one is a subset of the other
        # i.e. 'Exxon Mobil Corp' and 'Mobil Corp'
        name_len = [len(an) for an in anames]
        name_ord = np.argsort(name_len)[::-1]

        maintext = a["maintext"]
        name_in_article = []
        for an in anames[name_ord]:
            if re.search(an, maintext):
                # get the new-names  in article
                name_in_article.append(an)
                maintext = re.sub(an, dash_names[an], maintext)
            else:
                # print(f"could not find: {an} in {k}")
                if k in bad_articles:
                    bad_articles[k] += [an]
                else:
                    bad_articles[k] = [an]

        names_in_text_dict[k] = name_in_article

    return names_in_text_dict


def articles_containing_company_pair(articles, kb):

    # find all the articles that contain parent nane
    print("searching for articles containing each company name")

    # search for just entity1 first
    pname = kb["entity1"].unique()
    pname = np.unique(pname)
    pdict = {}
    for i, pn in enumerate(pname):
        if i % 200 == 0:
            print(f"{i}/{len(pname)}")
        for k, v in articles.items():
            if pn in v["names_in_text"]:
                if pn in pdict:
                    pdict[pn] += [k]
                else:
                    pdict[pn] = [k]

    print(f"number of entity1 entries found in at least one article: {len(pdict)}")

    # then search for entity2
    # store the articles containing the company pairs in a dict
    cpairs = {}
    for i, _ in enumerate(pdict.items()):
        pn, pv = _

        cnames = kb.loc[kb["entity1"] == pn, "entity2"].unique()

        # for each company, check the articles the parent name is mentioned in
        for cn in cnames:
            # company is a supplier and customer to itself - skip
            if pn == cn:
                print(f"skipping {pn}|{cn}")
                continue

            pair = f"{pn}|{cn}"
            for k in pv:
                # if company is also in article store that
                if cn in articles[k]["names_in_text"]:
                    if pair in cpairs:
                        cpairs[pair] += [k]
                    else:
                        cpairs[pair] = [k]

    return cpairs


def replace_longer_names_with_shortest(text,
                                       replace_names,
                                       short_name_map,
                                       verbose=False):
    """given an article / text and set of names to replace (with shorter ones)"""

    replace_dict = {rn: [rn] + short_name_map.get(rn, [])
                    for rn in replace_names}

    # shortest_name_dict

    # replace the longest names first
    replace_names.sort(key=lambda x: len(x))
    replace_names = replace_names[::-1]

    # for each name to replace in article - will replace with the shortest name
    # replace each name with the shoartest name
    replaced_names_dict = {}
    for rn in replace_names:
        # if the name to replace is no longer in the text - just skip
        # i.e. if Exxon Mobil Corp -> Exxon Mobil then Mobil Corp won't be found
        if re.search(rn, text) is None:
            if verbose:
                print(f"{rn} no longer found in text")
            continue
        # get the names to replace
        names = replace_dict[rn]
        shortest_name = names[-1]
        longer_names = names[:-1]

        replaced_names_dict[rn] = shortest_name
        for ln in longer_names:
            # search for long name in text - it's possible it may no longer exist
            text = re.sub(ln, shortest_name, text)

    # return the text with longer name(s) replaced with short name
    # - along with dictionary for that mapping
    return text, replaced_names_dict


if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)

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

    # ----
    # read the articles in
    # ----

    # read proof of concept data
    assert os.path.exists(get_data_path("articles.json")), \
        f"{get_data_path('articles.json')} does not exist, get from the google drive and save locally"

    with open(get_data_path("articles.json"), "r") as f:
        article_list = json.load(f)

    # store in dict with keys matching json_file - without .json suffix
    # select those that have 'names_in_text'
    # - this should not be needed in the future
    articles = {re.sub("\.json$", "", f["json_file"]): f
                for f in article_list if "names_in_text" in f}

    # get article keys
    keys = list(articles.keys())

    print(f"read in: {len(keys)} articles")

    # ---
    # read in long to short name mapping
    # ---

    ltsn = [i for i in client["news_articles"]["long_to_short_name"].find()]
    ltsn_map = {i["long_name"]: i["short_names"] for i in ltsn if len(i["short_names"]) > 0}
    client.close()

    # ---
    # read in value chain data / knowledge base
    # ---

    print('getting knowledge base')
    # vc = pd.DataFrame(list(client["refinitiv"]["VCHAINS"].find(filter={})))
    # vc.drop("_id", axis=1, inplace=True)
    # vc.to_csv(get_data_path("VCHAINS.csv"), index=False)
    vc = pd.read_csv(get_data_path("VCHAINS.csv"))

    # there are some missing comany names? exclude those
    # vc = vc.loc[~pd.isnull(vc['Company Name'])]

    kb = get_knowledge_base_from_value_chain_data(vc)

    # ---
    # filter: keep only articles that mention two (known) companies
    # ----

    # keep only those articles with two or more names_in_text
    articles = {k: articles[k]
                for k, v in articles.items()
                if len(v['names_in_text']) > 1}

    keys = list(articles.keys())

    print(f"keeping only those with more than one name\nthere are now: {len(articles)} articles")

    # -----
    # replace long names with short ones
    # -----

    # TODO: there could be more than one short name - these should be handled correctly
    #  - namely names should be change to the shortest of all the names

    all_names = np.unique(np.concatenate([v["names_in_text"]
                                          for k, v in articles.items()]))

    short_name_map = niave_long_to_short_name(all_names)

    # get the short name map - giving preference to those read from database (in ltsn_map)
    short_name_map = {k: [v] if k not in ltsn_map else ltsn_map[k]
                      for k, v in short_name_map.items()}

    # put the shortest name last
    for k in short_name_map.keys():
        short_name_map[k].sort(key=lambda x: len(x))
        short_name_map[k] = short_name_map[k][::-1]

    # for each article - replace the longer names with the short names
    # and store the long_to_short_names mapping (dict)
    for i, _ in enumerate(articles.items()):
        k, v = _
        if i % 1000 == 0:
            print(f"{i}/{len(articles)}")

        text, replace_dict = replace_longer_names_with_shortest(text=v['maintext'],
                                                                replace_names=v['names_in_text'],
                                                                short_name_map=short_name_map)
        # add modified text
        articles[k]['mod_maintext'] = text
        # add the long name to short name mapping used
        # - NOTE: some other names may have been mapped
        articles[k]["long_to_short_names"] = replace_dict

    print("taking only articles with more than one (long) to short name")
    articles = {k: articles[k]
                for k, v in articles.items()
                if len(v['long_to_short_names']) > 1}

    keys = list(articles.keys())

    # ---
    # remove carriage returns - to better separate sentences
    # ---

    for k in articles.keys():
        articles[k]["mod_maintext"] = re.sub("\n", " ", articles[k]["mod_maintext"])
        # articles[k]["mod_maintext"] = re.sub("“", "\"", articles[k]["mod_maintext"])

    # ----
    # get sentences
    # ----

    # TODO: perhaps want to get rid of sentences that start with \nFILE PHOTO
    #  - and end with \nFILE PHOTO

    # use SpaCy to get sentences
    # - https://spacy.io/usage/spacy-101#annotations
    # - requires:$ python -m spacy download en_core_web_md
    nlp = spacy.load("en_core_web_sm")
    # nlp = spacy.load("en_core_web_lg")
    # nlp = spacy.load("en_core_web_md")

    # store results (sentences) in a list
    out = []
    # keep track of cases that fall over - investigate later
    investigate = []
    # increment over each of the articles that reference supply
    for ii, _ in enumerate(articles.items()):

        if ii % 1000 == 0:
            print(f"{ii}/{len(articles)}")

        # get article key and details
        k, v = _

        # ---
        # get the relations from the knowledge base
        # ---

        # get all the combinations of the long names
        lnames = [k for k in v["long_to_short_names"].keys()]
        # - store in dict
        combs = [(c[0], c[1]) for c in itertools.combinations(lnames, 2)]

        relations = []
        for cc in combs:
            has_connection = np.in1d(kb['entity1'].values, cc) & np.in1d(kb['entity2'].values, cc)
            # if there is a connection anywhere select it
            # - this should pick up relations that go either way i.e. A supplies B and B supplies A
            if has_connection.any():
                relations.append(kb[has_connection].copy())
            # otherwise, it is a negative case - set relation to NA (not available)
            else:
                relations.append(pd.DataFrame({"entity1": cc[0], "entity2": cc[1], "rel": "NA"}, index=[0]))

        relations = pd.concat(relations)

        # ---
        # for each entity pair get the sentences
        # ---

        # provide text to spacy - just to get the sentences
        text = articles[k]["mod_maintext"]

        # apply spacy to text - just to get sentences
        doc = nlp(text)

        # get start and end positions
        sent_list = list(doc.sents)
        sent_start = np.array([sent.start_char for sent in doc.sents])
        sent_end = np.array([sent.end_char for sent in doc.sents])

        for idx, row in relations.iterrows():

            # get the entities - mapped to their short names
            e1 = row["entity1"]
            e2 = row["entity2"]
            e1_short = v['long_to_short_names'][e1]
            e2_short = v['long_to_short_names'][e2]
            rel = row['rel']

            try:
                # find where entity1 is in text
                a = np.array([(m.start(), m.end()) for m in re.finditer(e1_short, text)])

                # find where entity2 is in text
                b = np.array([(m.start(), m.end()) for m in re.finditer(e2_short, text)])

                # find points in the text to connect the two, via sentence
                start_pts, start_names = get_start_end(a=a[:, 0], b=b[:, 0], aname=e1_short, bname=e2_short)
                end_pts, end_names = get_start_end(a=a[:, 1], b=b[:, 1], aname=e1_short, bname=e2_short)
            except Exception as e:
                print(e)
                investigate.append([k, (e1, e2)])
                print(e1, e2)
                continue

            if len(start_pts) != len(end_pts):
                print("starting points and end points not to the, expect them to be")
                investigate.append([k, (e1, e2)])
                print((e1, e2))
                continue

            # TODO: check start, end names are the same
            # ---
            # get the full sentence
            # ---
            for i, sp in enumerate(start_pts):
                nme = start_names[i]
                sp = start_pts[i]
                ep = end_pts[i]

                # --
                # find the start sentence location
                # --
                # - by taking the start of the first entity, identify the sentences
                # - where that is before the end and take the maximum

                # TODO: this should be double checked/validated
                # start sentence location
                sloc = np.argmax(sp[0] < sent_end)
                # end sentence location
                eloc = np.argmax(sp[1] < sent_end)

                left_start_char = sent_list[sloc].start_char
                right_end_char = sent_list[eloc].end_char
                full_sentence = text[left_start_char: right_end_char]

                #
                date_pub = v["date_publish"]

                # store results in a list
                # - as a dict so can
                # TODO: perhaps include sentence range
                res = {"full_sentence": full_sentence,
                       "entity1": e1_short,
                       "entity2": e2_short,
                       "relation": rel,
                       "entity1_full": e1,
                       "entity2_full": e2,
                       "start_sent": int(sloc),
                       "end_sent": int(eloc),
                       "num_sentence": int((eloc - sloc) + 1),
                       "article": k,
                       "date_publish": v["date_publish"],
                       "source_domain": v["source_domain"]}

                out.append(res)

                # # if the sentences are not too far apart
                # if eloc - sloc <= max_num_sentences_to_combine:
                #
                # else:
                #     too_long_count += 1
                #     # print("sentence too long")


    # ----
    # write to file
    # -----

    with open(get_data_path("full_sentences.json"), "w") as f:
        json.dump(out, f, indent=4)

