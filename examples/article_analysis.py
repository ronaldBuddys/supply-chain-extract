# look at names in articles, how often pairs occur, where supply is mention
# number of unique titles


import json
import os
import re
import time
import gzip
import itertools


import sys
import numpy as np
import pandas as pd

# from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy
from spacy.matcher import PhraseMatcher


try:
    # python package (nlp) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None
    
from nlp.utils import get_database
from nlp import get_configs_path, get_data_path

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def find_all_articles_with_name(articles, name):
    return [v for k, v in articles.items() if re.search(name, v["maintext"])]


def remove_suffix(name, suffixes):
    for s in suffixes:
        # regex: space, word, space then any character to end
        # or
        name = re.sub(f" {s} .*$| {s}$", "", name)
    return name


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

if __name__ == "__main__":

    #TODO: Cleaning of news data
    #Extra cleaning on the news data? 
    #Clean out bad articles

    #TODO: Long.Short Name Conversion
    #Long name to many short names

    #TODO: Article analysis on number of false positives
    #Look at how articles mention each c
    # ompany 
    #Eyeball false positive to true positives 
    #False positives -> Create manual labels 

    #TODO: NEgative Examples
    #Negative Examples
    #Negative example from articles that mentions company A from KB and any entity recognised by spaCy
    #Use spacy tags to create negative examples?

    #TODO: More Spacy Integration and Language Model Integration
    #Use cusutom tokenizier instead of regex to match entities
    #E.g. Spacy POS Tags to be used as features
    #Figure out where inthe process a large language model would fit

    # ----
    # parameters
    # ----

    # if True will read articles from local json file: data/articles.json
    # - this can reduce burden on the remote database (network usage)
    read_local_articles = True

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
    if read_local_articles:
        # read proof of concept data
        with open(get_data_path("articles.json"), "r") as f:
            article_list = json.load(f)

        #
        articles = {re.sub("\.json$", "", f["json_file"]): f
                    for f in article_list if "names_in_text" in f}

    else:
        # instead, read (a larger) set of articles from mongo
        # - this can be quite large
        # because taking 'names_in_text' this might not be everything
        articles = {re.sub("\.json$", "", f["json_file"]): f
                    for f in client["news_articles"]["articles"].find({'names_in_text': {"$exists": True}})}

    keys = list(articles.keys())

    print(f"read in: {len(keys)} articles")

    # ---
    # read in value chain data
    # ---

    vc = pd.DataFrame(list(client["refinitiv"]["VCHAINS"].find(filter={})))

    #TODO: Connect to DB to load latest mapping of long to short name
    #print(f"database names: {client.list_database_names()}")

    # ---
    # remove names pairs that have been erroneously found in text, because of similarities in names
    # ----

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
        dnames = np.array([dash_names[an] for an in anames])
        # replace the longest matching first - to deal with
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

    # keep only those articles with two or more names_in_text
    articles = {k: articles[k]
                for k, v in names_in_text_dict.items()
                if len(v) > 1}

    print(f"after removing erroneous articles, there are now: {len(articles)} articles")

    # ---
    # search for articles containing pair
    # ---

    # TODO: refactor this - search for parents first

    # find all the articles that contain parent nane
    print("searching for parent names")
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
    #####This is optional exploratory analysis##### 
    all_titles = {"titles": [a['title'] for a in articles.values()]}
    all_titles = pd.DataFrame(all_titles)
    all_titles["count"] = 1

    title_count = pd.pivot_table(all_titles, index="titles", values="count", aggfunc="count")

    title_count.sort_values("count", ascending=False, inplace=True)

    print(f"there are: {len(title_count)} unique titles, {len(title_count.loc[title_count['count']> 1])} occur more than once")

    # ----
    # for each company in value chain data - count of the number of articles in
    # ----

    article_count = {}
    # NOTE: maybe reverse this operation and just cycle through articles
    # - rather than names and articles ~ O(m*n)
    for n in all_names:
        for k, v in articles.items():
            if n in v["names_in_text"]:
                if n in article_count:
                    article_count[n] += 1
                else:
                    article_count[n] = 1

    print(f"there were {len(article_count)} companies found in articles")

    c_count = pd.DataFrame([(k, v) for k, v in article_count.items()],
                            columns=["name", "in_articles"])
    c_count.sort_values("in_articles", ascending=False, inplace=True)

    # ---
    # get name suffixes - in an attempt to make a 'short' name that often gets referenced in article
    # ---
    # TODO: Replace this with MongoDB Mapping

    # TODO: short_name_map needs to be reviewed!, preferable to use some NLP package (spacy?)
    # This is pretty hard coded list of company name 'suffixes'
    # - some of these were taken by counting suffixes occrances, removing those and repeating
    # - others were just a gues
    suffixes = ['Inc', 'Corp', 'Ltd', 'Co', 'PLC', 'SA', 'AG', 'LLC', 'NV', 'SE',
                'ASA', 'Bhd', 'SpA', 'Association', 'Aerospace', 'AB', 'Oyj', "Plc"] + \
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
    short_name_map["Amazon.com Inc"] = "Amazon"
    short_name_map["General Electric Co"] = "GE"
    short_name_map["Lockheed Martin Corp"] = "Lockheed"

    # add to database - this should be one off
    # - it is not efficient to search names one at a time but with small number shouldn't be a problem
    # for k, v in short_name_map.items():
    #     filter = {"long_name": k}
    #     lsname = client["news_articles"]["long_to_short_name"].find_one(filter)
    #
    #     # if name has not be checked add guess from above
    #     if not lsname["checked"]:
    #         client["news_articles"]["long_to_short_name"].update_one(filter,
    #                                                                  update={"$addToSet": {"short_names": v}})

    c_count["short_name"] = c_count["name"].map(short_name_map)

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
            # HARDCODE: replace ’ with ' - just a guess to try to deal with them
            a["maintext"] = re.sub("’", "'", a["maintext"])

        a["short_names_in_text"] = [short_name_map[n] for n in a["names_in_text"]]
        articles[k] = a

    # TODO: remove "F2 percent" ? LMT;-PCTCHNG:2} ?

    ####Section on Creating examples
    
    # ----
    # spacy
    # ----

    # TODO: perhaps want to get rid of sentences that start with \nFILE PHOTO
    # - and end with \nFILE PHOTO


    # https://spacy.io/usage/spacy-101#annotations

    # requires:$ python -m spacy download en_core_web_md
    # nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_sm")
    # nlp = spacy.load("en_core_web_md")

    # ----
    # Extending entity ruler -> NOT USED AT THIS POINT ANYMORE (Use direct matching of company short name instead ) TODO: Use custom tokenizer instead
    # ----
    # https://spacy.io/usage/rule-based-matching#entityruler

    ruler = nlp.add_pipe('entity_ruler')
    patterns = [{"label": "ORG", "pattern": v} for v in short_name_map.values()]
    ruler.add_patterns(patterns)

    # ----
    # Start of loop to create examples

    # store results in a list
    out = []
    #Create an error log
    errorlog = {}
    # increment over each of the articles that reference supply
    for ii, k in enumerate(supply_articles):
    #for ii, k in enumerate(["d70b8e93655d2bcfe4d17991e8d6a3a6ef5b8bd6b86d55e6a7ebd5ae5b1b637e"]):
        if ii % 100 == 0:
            print(f"{ii} / {len(supply_articles)}")
        
        #This create all possible pairs that are included in the article 
        combinations = itertools.combinations(articles[k]["names_in_text"],2)   
        pairs_list = [f"{combination[0]}|{combination[1]}" for combination in combinations]
        
        #cps lists all the pairs that are included in the article that are also in the knowledge base -> Will use this to create a label for the article
        cps = [cp for cp, v in cpairs.items() if k in v]
        #cps includes all paris from the KB mentioned in the article. cps_mirror includes all the reversed pairs
        #This is done so that the order in which all possible pairs were created does not matter when we generate the label
        cps_mirror = cps + [f"{cp.split('|')[::-1][0]}|{cp.split('|')[::-1][1]}" for cp in cps]

        text = articles[k]["maintext"]
        doc = nlp(text)

        #Split article into sentences
        sent_list = list(doc.sents)
        sent_start = np.array([sent.start_char for sent in doc.sents])
        sent_end = np.array([sent.end_char for sent in doc.sents])
        
        # for each pair, get the short name
        for cp in pairs_list:
            #TODO: Check if better to use custom tokenizier
            # long names
            ln = cp.split("|")
            # find relevant short name using the mapping to match names in text
            sn = [short_name_map[l] for l in ln]

            # Generate a label for the pair that is used forall senteces
            label = 1 if cp in cps_mirror else 0

            # find the locations where entity 'a' is mentioned
            # a = [(m.start(), m.end()) for m in re.finditer(sn[0], articles[k]["maintext"])]

            try:
                #Finding short names using regex over full doc
                #First find all mentions of a and b
                a_unf = []
                expression_a = sn[0]
                for match in re.finditer(expression_a, doc.text):
                    a_unf.append(match)
                
                b_unf = []
                expression_b = sn[1]
                for match in re.finditer(expression_b, doc.text):
                    b_unf.append(match)

                #Filter out cases where one name encompasses another
                a = []
                for span_a in a_unf:
                    count = 0
                    for span_b in b_unf:
                        #Check if the matches for a are included in the matches found in b, if yes ignore those -> Example: comp a Suzuki's match is included in comp b Maruti Suzuki India's match
                        if span_a.span()[0] >= span_b.span()[0] and span_a.span()[1] <= span_b.span()[1]:
                            count +=1
                    if count==0:
                        a.append(span_a.span()) 
                a = np.unique(np.array(a)).reshape(-1,2)

                b = []
                for span_b in b_unf:
                    count = 0
                    for span_a in a_unf:
                        #Check if the matches for b are included in the matches found in a, if yes ignore those
                        if span_b.span()[0] >= span_a.span()[0] and span_b.span()[1] <= span_a.span()[1]:
                            count +=1
                    if count==0:
                        b.append(span_b.span())
                b = np.unique(np.array(b)).reshape(-1,2)

                #Check if either a or b is empty
                if len(np.squeeze(a)) == 0:
                    raise Exception(sn[0]," not found, most likely it was included in the other company name")
                if len(np.squeeze(b)) == 0:
                    raise Exception(sn[1]," not found, most likely it was included in the other company name")

                # find points in the text to connect the two, via sentence
                # storing long names instead of short names
                start_pts, names = get_start_end(a=a[:,0], b=b[:,0], aname=ln[0], bname=ln[1])
                end_pts, names = get_start_end(a[:,1], b[:,1], aname=ln[0], bname=ln[1])

            except Exception as e:
                print(e)
                if k in errorlog.keys():
                    errorlog[k] += [cp]
                else:
                    errorlog[k] = [cp]

            assert len(start_pts) == len(end_pts), \
                "starting points and end points not to the, expect them to be"

            # get the full sentence
            for i, sp in enumerate(start_pts):
                nme = names[i]
                ep = end_pts[i]
                # find the start sentence location
                # - by taking the start of the first entity, identify the sentences
                # - where that is before the end and take the maximum
                # TODO: this should be double checked/validated
                sloc = np.argmax(sp[0] < sent_end)
                # end sentence location
                eloc = np.argmax(sp[1] < sent_end)

                # s = sent_list[0]
                # s.char_span(start_idx=0, end_idx=200)

                # TODO: should there be a limit in the distance / number
                #  of sentences inbetween?
                # for now require difference to be less than equal to two
                # NOTE: some sentence could just be \n
                if eloc - sloc <= 3:
                    # 'left' sentence starts
                    left_sent_start = sent_list[sloc].start_char
                    # take to where the first entity start
                    left = text[left_sent_start: sp[0]]
                    # middle is from end of first entity to start of next
                    middle = text[ep[0]:sp[1]]
                    # right is from end of second entity to the end of that sentence
                    right = text[ep[1]: sent_list[eloc].end_char]

                    # full_sentence = left + nme[0] + middle + nme[1] + right

                    # store results in a list
                    out.append([nme[0], nme[1], left, middle, right, k, label])
    # End of loop to create examples
    # ----

    #Store Dataframe
    df = pd.DataFrame(out, columns=["entity1", "entity2", "left", "middle", "right", "article","label"])
    df.to_csv(get_data_path("example_inputs_pos_and_neg.tsv"), sep="\t", index=False)

    # ---
    # Store in Format to be used in Stanford Notebooks
    # ---

    #Creating a dataframe in the form required by the Coprus class
    articles_final = pd.DataFrame()
    #Bring DF in shape expected by Corpus class
    articles_final.loc[:,["entity1","entity2","left","entity1",'middle','entity2','right',"left","entity1",'middle','entity2','right']] = df.loc[:,["entity1","entity2","left","entity1",'middle','entity2','right',"left","entity1",'middle','entity2','right']]
    #Remove \n as this trips up the Corpus class
    articles_final.replace("\n","",inplace=True)
    articles_final.replace('(\n)','',regex=True,inplace=True)

    #Store as TSV
    articles_final.to_csv(get_data_path("example_inputs_long_names_with_neg.tsv"), sep="\t", index=False,header=False)

    #Convert to GZ file
    with open(get_data_path("example_inputs_long_names_with_neg.tsv"), 'rb') as src, gzip.open(get_data_path("example_inputs_long_names_with_neg.tsv.gz"), 'wb') as dst:
        dst.writelines(src)
    

    ####Previous stuff

    #doc = nlp("A complex-example,!")
    # print([token.text for token in doc])
    #
    # a = articles[keys[1]]
    # t0 = time.perf_counter()
    # doc = nlp(a["maintext"])
    # t1 = time.perf_counter()
    # # will take about ~ 11min to run fo 5500 articles
    #
    # print(a["short_names_in_text"])
    # print(json.dumps([(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "ORG"], indent=4))
    # # for token in doc:
    # #     print(token.text)
    # print(a["names_in_text"])
    #
    # for ent in doc.ents:
    #     if ent.label_ == "ORG":
    #         print(ent.text, ent.start_char, ent.end_char, ent.label_)
    #     # if ent.label_ == "GPE":
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

    # get the articles that have no names in text
    # no_names_in_text = [f for f in client["news_articles"]["articles"].find({'names_in_text': {"$exists": False}})]
    # # save them local - must convery _id
    # tmp = []
    # for n in no_names_in_text:
    #     n["_id"] = str("_id")
    #     tmp.append(n)
    # with open(get_data_path("articles_with_no_names_in_text.json"), "w") as f:
    #     json.dump(tmp, f)
    # # drop those articles
    # client["news_articles"]["articles"].delete_many({'names_in_text': {"$exists": False}})