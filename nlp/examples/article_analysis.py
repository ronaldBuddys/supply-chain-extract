# look at names in articles, how often pairs occur, where supply is mention
# number of unique titles


import json
import os
import re
import sys
import numpy as np
import pandas as pd

# from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy


try:
    # python package (nlp) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None






from nlp.utils import get_database
from nlp import get_configs_path, get_data_path


def find_all_articles_with_name(articles, name):
    return [v for k, v in articles.items() if re.search(name, v["maintext"])]


def remove_suffix(name, suffixes):
    for s in suffixes:
        # regex: space, word, space then any character to end
        # or
        name = re.sub(f" {s} .*$| {s}$", "", name)
    return name


if __name__ == "__main__":

    # TODO: confirm using the long name -> short name replacement
    # TODO: review long name -> short name relationship, try to avoid using rigit rules
    # TODO: add entity rule for short names using spacy?
    # TODO: attempt relationship mapping

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

    # TODO: could read from database, put that here

    # read proof of concept data
    with open(get_data_path("articles_with_names.json"), "r") as f:
        articles = json.load(f)

    keys = list(articles.keys())

    # ---
    # read in value chain data
    # ---

    vc = pd.DataFrame(list(client["refinitiv"]["VCHAINS"].find(filter={})))

    # ---
    # search for articles containing pair
    # ---

    # TODO: refactor this - search for parents first

    # find all the articles that contain parent nane
    pname = vc["Parent Name"].unique()
    pdict = {}
    for i, pn in enumerate(pname):
        if i % 50 == 0:
            print(f"{i}/{len(pname)}")
        for k, v in articles.items():
            if pn in v["names_in_text"]:
                if pn in pdict:
                    pdict[pn] += [k]
                else:
                    pdict[pn] = [k]

    print(f"number of Parent Company's found in articles: {len(pdict)}")

    # store the articles containing the company pairs in a dict
    cpairs = {}
    for i, _ in enumerate(pdict.items()):
        pn, pv = _
        # for each parent get the Suppliers
        # TODO: loosen this to get all company (i.e. include Customers)
        cnames = vc.loc[(vc["Parent Name"] == pn) & (vc["Relationship"] == "Supplier"),
                        "Company Name"].unique()
        # for each company, check the articles the parent name is mentioned in
        for cn in cnames:
            pair = f"{pn}|{cn}"
            for k in pv:
                # if company is also in article store that
                if cn in articles[k]["names_in_text"]:
                    if pair in cpairs:
                        cpairs[pair] += [k]
                    else:
                        cpairs[pair] = [k]

    # find the pair that is mentioned the most
    cpair_count = [(k, len(v)) for k, v in cpairs.items()]
    cpair_count = pd.DataFrame(cpair_count, columns=["pair", "count"])
    cpair_count.sort_values("count", ascending=False, inplace=True)

    # ----
    # find all the articles with supply|supplies|supplier in maintext
    # ----

    supply_articles = []
    for k, v in articles.items():
        if re.search("supply|supplier|supplies", v["maintext"]):
            supply_articles.append(k)

    print(f"there are: {len(supply_articles)} that mention {'supply|supplier|supplies'}")

    # ----
    # title count
    # ----

    all_titles = {"titles": [a['title'] for a in articles.values()]}
    all_titles = pd.DataFrame(all_titles)
    all_titles["count"] = 1

    title_count = pd.pivot_table(all_titles, index="titles", values="count", aggfunc="count")

    title_count.sort_values("count", ascending=False, inplace=True)

    print(f"there are: {len(title_count)} unique titles, {len(title_count.loc[title_count['count']> 1])} occur more than once")

    # ---
    # get name suffixes - in an attempt to make a 'short' name that often gets referenced in article
    # ---

    # TODO: short_name_map needs to be reviewed!, preferable to use some NLP package (spacy?)
    # This is pretty hard coded list of company name 'suffixes'
    # - some of these were taken by counting suffixes occrances, removing those and repeating
    # - others were just a gues
    suffixes = ['Inc', 'Corp', 'Ltd', 'Co', 'PLC', 'SA', 'AG', 'LLC', 'NV', 'SE',
                'ASA', 'Bhd', 'SpA', 'Association', 'Aerospace', 'AB', 'Oyj'] + \
               ['Co', 'Holdings', 'Group', 'Technologies', 'International',
                'Systems', 'Energy', 'Communications', 'Airlines', 'Motor',
                'Technology', 'Oil', 'Motors', 'Corp', 'Industries', 'Steel',
                'Holding', 'Airways', 'Aviation', 'Automotive', 'Networks',
                'Electronics', 'Digital', 'BP', 'Electric', 'Aircraft',
                'US', 'Mobile', 'Software', 'Broadcom', 'Brands',
                'Service', 'Semiconductor', 'Petroleum'] + \
               ['Platforms', 'Precision', 'Industry', 'AeroSystems', 'Media', 'Petrochemical']



    #  'International Business Machines Corp' -> 'IBM'
    # 'News Corp' -> 'News Corp'
    # NOTE: if it starts with air it needs two words

    all_names = np.unique(np.concatenate([v["names_in_text"] for k, v in articles.items()]))

    short_name = pd.DataFrame([(n, remove_suffix(n, suffixes)) for n in all_names],
                              columns=["name", "short"])
    # look at longer names
    short_name["len"] = [len(n) for n in short_name["short"]]
    short_name.sort_values("len", ascending=False, inplace=True)

    # making a mapping dictionary
    short_name_map = {i[0]: i[1] for i in zip(short_name["name"], short_name["short"])}

    # HARDCODED!
    short_name_map['International Business Machines Corp'] = "IBM"
    short_name_map['News Corp'] = 'News Corp'

    # TODO: should check content to determine the which (short names?) are duplicated

    # -----
    # replace long names with short ones
    # -----

    # replace the long names with short names
    # - this is done because the short names are often used more
    keys = list(articles.keys())
    for k in keys:
        a = articles[k]
        for n in a['names_in_text']:
            a["maintext"] = re.sub(n, short_name_map[n], a["maintext"])
        a["short_names_in_text"] = [short_name_map[n] for n in a["names_in_text"]]
        articles[k] = a

    # TODO: remove "F2 percent" ? LMT;-PCTCHNG:2} ?

    # ----
    # spacy
    # ----

    # https://spacy.io/usage/spacy-101#annotations

    # requires:$ python -m spacy download en_core_web_md
    # nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_md")

    # https://spacy.io/usage/rule-based-matching#entityruler
    ruler = nlp.add_pipe('entity_ruler')

    patterns = [{"label": "ORG", "pattern": v} for v in short_name_map.values()]
    ruler.add_patterns(patterns)
    # print(json.dumps(patterns, indent=4))

    a = articles[keys[1]]

    doc = nlp(a["maintext"])

    print(a["short_names_in_text"])
    print(json.dumps([(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "ORG"], indent=4))
    # for token in doc:
    #     print(token.text)
    print(a["names_in_text"])

    for ent in doc.ents:
        if ent.label_ == "ORG":
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
        # if ent.label_ == "GPE":
        # if re.search("Alphabet", ent.text):
        #     print(ent.text, ent.start_char, ent.end_char, ent.label_)

    # from spacy import displacy
    # displacy.serve(doc, style="dep")


    # # ---
    # # replace names with dash separated - in an attempt to deal with short names later
    # # ---
    #
    # # TODO: drop this - just use spacy?
    #
    # all_names = np.unique(np.concatenate([v["names_in_text"]
    #                                       for k, v in articles.items()]))
    # dash_names = {n: re.sub(" ", "-", n) for n in all_names}
    #
    # # for each article replace the name with the dashed name
    # # - the idea is this will help with tokenizing
    #
    #
    # for k in keys:
    #     a = articles[k]
    #     anames = np.array(a["names_in_text"])
    #     dnames = np.array([dash_names[an] for an in anames])
    #     # replace the longest matching first - to deal with
    #     name_len = [len(an) for an in anames]
    #     name_ord = np.argsort(name_len)[::-1]
    #
    #     maintext = a["maintext"]
    #     name_in_article = []
    #     for an in anames[name_ord]:
    #         if re.search(an, maintext):
    #
    #             name_in_article.append(an)
    #             maintext = re.sub(an, dash_names[an], maintext)
    #         else:
    #             # print(f"could not find: {an} in {k}")
    #             pass
    #
    #     a["names_in_text"] = name_in_article
    #
    # # keep only those articles with two or more names_in_text
    # articles = {k: v for k, v in articles.items() if len(v["names_in_text"]) > 1}


    # tmp = [a for a in articles.values() if 'News Corp' in a["names_in_text"]]
    #
    #
    # all_sufs = pd.DataFrame([ (n, n.split(" ")[-1])for n in all_names],
    #                         columns=["name", "suf"])
    # suf_count = pd.pivot_table(all_sufs,
    #                            index="suf",
    #                            values="name",
    #                            aggfunc="count").reset_index()
    #
    # suf_count.sort_values("name", ascending=False, inplace=True)
    #
    # # drop the more common suffixes - in an attempt to make short names
    # suf_vals = suf_count.loc[suf_count["name"] >= 2, "suf"].values
    # # regular expression for matching suffixes
    # suf_regex = "".join([f" {sv}$|" for sv in suf_vals])
    #
    # # get the second suffix
    # suf2 = pd.DataFrame([(n, re.sub(suf_regex, "", n).split(" ")[-1])
    #                      for n in all_names if bool(re.search(suf_regex, n))],
    #                     columns=["name", "suf"])
    #
    # suf_count2 = pd.pivot_table(suf2,
    #                            index="suf",
    #                            values="name",
    #                            aggfunc="count").reset_index()
    # suf_count2.sort_values("name", ascending=False, inplace=True)
    #
    # # do another suffix count
    # suf_vals2 = suf_count2.loc[suf_count2["name"] >= 2, "suf"].values
    # suf_regex2 = "".join([f" {sv}$|" for sv in suf_vals2])
    #
    # short_names = {n: re.sub(suf_regex2, "", re.sub(suf_regex, "", n))
    #                for n in all_names}
    #
    # tmp = [a for a in articles.values() if 'Meta Platforms Inc' in a["names_in_text"]]
    # tmp = [a for a in articles.values() if 'Electronic Arts Inc' in a["names_in_text"]]


    # HARDCODED: ['Exxon Mobil Corp', 'Mobil Corp'] were erronously mathced?
    # drop_exxon_mobil = []
    # for k in articles.keys():
    #     if np.in1d(articles[k]["names_in_text"], ['Exxon Mobil Corp', 'Mobil Corp']).all():
    #         drop_exxon_mobil.append(k)
    #     # just remove Mobil Corp - this isn't robust
    #     if np.in1d(['Mobil Corp'], articles[k]["names_in_text"]).any():
    #         articles[k]["names_in_text"] = [n for n in articles[k]["names_in_text"] if n != "Mobil Corp"]
    #
    # for k in drop_exxon_mobil:
    #     articles.pop(k)

    # # ---
    # # rules for getting short names
    # # ---
    #
    # all_names = np.unique(np.concatenate([v["names_in_text"]
    #                                       for k, v in articles.items()]))
    #
    # tmp = find_all_articles_with_name(articles, 'Ryanair Holdings PLC')
    #
    # # Rule 1: first name
    # # short_names = [n.split(" ")[0] for n in all_names]
    #
    # # keep track of articles where did not find all names
    # not_found_all_names = []
    #
    # #
    # nlp = English()
    # tokenizer = nlp.tokenizer
    #
    # # record all the short names
    # short_name_dict = {n: np.array([]) for n in all_names}
    #
    # investigate = []
    # for i, _ in enumerate(articles.items()):
    #
    #     stop = False
    #     k, v = _
    #     names = v["names_in_text"]
    #     # Rule 1: first name
    #     short_names = {n: n.split(" ")[0] for n in names}
    #
    #     # tokenize the data
    #     # - is there a better, more correct way of doing this?
    #     text = v['maintext']
    #
    #     # confirm the long names are in text
    #     # - then remove it
    #     for n in names:
    #
    #         if not bool(re.search(n, text)):
    #             print(f"{n} not found in text, expected to!")
    #             print(f"names list: {names}")
    #             stop = True
    #             break
    #
    #         # remove the long names
    #         text = re.sub(n, "", text)
    #
    #     if stop:
    #         print("skipping")
    #         investigate += [k]
    #         continue
    #
    #     # now find the short names
    #     for n, sn in short_names.items():
    #         if bool(re.search(sn, text)):
    #             short_name_dict[n] = np.union1d(short_name_dict[n], sn)
    #     # replace the long names
    #
    #
    # len([n for n, v in short_name_dict.items() if len(v) == 0])
    #
    #
