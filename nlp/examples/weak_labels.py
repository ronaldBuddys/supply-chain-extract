# use snorkel to establish rules for making weak labels

import json
import os
import re
import time
import gzip
import itertools
from collections import OrderedDict


from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.augmentation import PandasTFApplier
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelingFunction
from snorkel.augmentation import MeanFieldPolicy
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.augmentation import RandomPolicy

import nltk
from nltk.corpus import wordnet as wn


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

    # TODO: remove all content inbetween () and {}
    # TODO: replace entities to be wrapped with {} and ()
    # TODO: create labelling function to use KB -
    # TODO: consider hardcoding no relation labels to be 0

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_colwidth", 70)

    # ----
    # parameters
    # ----

    max_sentence_number = 3

    max_characters = 500

    min_conf_score_for_supplier = 0.9

    # if True will read articles from local json file: data/articles.json
    # - this can reduce burden on the remote database (network usage)
    # read_local_articles = True

    # ----
    # connect to database
    # ----

    # get credentials
    # with open(get_configs_path("mongo.json"), "r+") as f:
    #     mdb_cred = json.load(f)
    #
    # # get mongodb client - for connections
    # client = get_database(username=mdb_cred["username"],
    #                       password=mdb_cred["password"],
    #                       clustername=mdb_cred["cluster_name"])
    # ---
    # read in value chain data / knowledge base
    # ---

    # t0 = time.time()
    # vc = pd.DataFrame(list(client["refinitiv"]["VCHAINS"].find(filter={})))
    # t1 = time.time()

    # there are some missing company names? exclude those
    # vc = vc.loc[~pd.isnull(vc['Company Name'])]

    # read in locally stored valued chains
    vc = pd.read_csv(get_data_path("VCHAINS.csv"))

    kb = get_knowledge_base_from_value_chain_data(vc)

    # for now no longer need connection
    # client.close()

    # ---
    # read in full_sentences store locally
    # ---
    # sent_file = get_data_path('full_sentences.json')
    sent_file = get_data_path("processed_sentences.json")

    assert os.path.exists(sent_file), \
        f"looks like: {sent_file}, copy from google drive data/{os.path.basename(sent_file)}"

    with open(sent_file, "r") as f:
        full_sents = json.load(f)

    df = pd.DataFrame(full_sents)

    # ---
    # store data in dataframe - to allow for high level summary
    # ---

    # TODO: this could be removed, it looks like secondary analysis
    #
    #
    #
    # # short to long name map for entities in sentences
    # s2l_entity1 = df[["entity1", "entity1_full"]].drop_duplicates()
    # s2l_entity2 = df[["entity2", "entity2_full"]].drop_duplicates()
    #
    # s2l_entity1.rename(columns={"entity1": "entity", "entity1_full": "entity_full"}, inplace=True)
    # s2l_entity2.rename(columns={"entity2": "entity", "entity2_full": "entity_full"}, inplace=True)
    #
    # s2l_entity = pd.concat([s2l_entity1, s2l_entity2])
    #
    # s2l_entity.drop_duplicates(inplace=True)
    #
    # # TODO: check how many relations are 1 to 1
    # #  - want to be strict on there only being a 1-1 mapping?
    # ent_count = pd.pivot_table(s2l_entity,
    #                            index='entity',
    #                            values='entity_full',
    #                            aggfunc="count").reset_index()
    #
    # multi_map_ents = ent_count.loc[ent_count['entity_full'] > 1, "entity"]
    # print("short to long names with 1 to many mapping")
    # print(s2l_entity.loc[s2l_entity['entity'].isin(multi_map_ents)].sort_values("entity"))

    # ----
    # Filtering
    # ----

    # TODO: consider if this should be done somewhere else - i.e. seperate to this
    # filtering
    # df = df.loc[df['num_sentence'] <= max_sentence_number]
    #
    # # add sentence length
    # df["num_chars"] = [len(i) for i in df['full_sentence']]
    # df = df.loc[df['num_chars'] <= max_characters]

    # rename full sentence to text
    df.rename(columns={"full_sentence": "text"}, inplace=True)

    # ---
    # process sentences / text
    # ---

    # include the confidence from the knowledge base
    df = df.merge(kb[["entity1", "entity2", "Confidence Score (%)"]],
                  left_on=["entity1_full", "entity2_full"],
                  right_on=["entity1", "entity2"],
                  how="left",
                  suffixes=["", "_drop"])

    df.drop(['entity1_drop', 'entity2_drop'], axis=1, inplace=True)

    # if confidence score is missing i.e. the relation was not in the KB
    # set it to zero
    df.loc[pd.isnull(df["Confidence Score (%)"]), "Confidence Score (%)"] = 0

    # HACK: review - if confidence is not high enough set relation ship to NA
    # - so we only considered most 'likely' supplier relationships
    # df.loc[df['Confidence Score (%)'] < min_conf_score_for_supplier, "relation"] = "NA"

    # NOTE: this is not really needed
    # - this should be done else where?
    # add {} around entity 1
    # df["text"] = [re.sub(row['entity1'], "{%s}" % row['entity1'], row['text'])
    #                        for idx, row in df.iterrows()]
    # # and [] around entity 2
    # df["text"] = [re.sub(row['entity2'], "[%s]" % row['entity2'], row['text'])
    #                        for idx, row in df.iterrows()]

    # --
    # add some more metrics
    # --

    plt.plot(np.sort(df["num_chars"].values)[::-1])
    plt.title("number of characters in text")
    plt.show()

    # approximate number of tokens
    df["num_tokens"] = [len(i.split(" ")) for i in df['text']]

    plt.plot(np.sort(df["num_tokens"].values)[::-1])
    plt.title("approximate number of tokens in text")
    plt.show()

    # --
    # high level analysis
    # --

    # sentence per article
    # sent_per_art = pd.pivot_table(df, index='article', values="text", aggfunc='count')
    # sent_per_art = sent_per_art.reset_index()
    # sent_per_art.sort_values("text", ascending=False, inplace=True)
    #
    # # determine a cutoff level
    # q_art = np.quantile(sent_per_art['text'].values, q=0.99)
    #
    # # select articles that are at or below above threshold
    # use_articles = sent_per_art.loc[sent_per_art['text'] <= q_art, "article"].values

    # take subset of sentences
    # df = df.loc[df['article'].isin(use_articles)]

    # ----
    # sentence 'size'
    # ----

    num_sent = pd.pivot_table(df,
                              index='num_sentence',
                              values="text",
                              aggfunc="count").reset_index()

    # --
    # relation count
    # --

    relation_count = pd.pivot_table(df,
                                    index="relation",
                                    values="text",
                                    aggfunc="count")
    print("count of 'relation'")
    print(relation_count)

    # ---
    # get a count of entity2 - the company that does the supplying
    # ---

    e2_count = pd.pivot_table(df.loc[df['relation'] == "Supplier"],
                               index="entity2",
                               values="text",
                               aggfunc="count").reset_index()
    e2_count.rename(columns={"text": "count"}, inplace=True)
    e2_count.sort_values("count", ascending=False, inplace=True)

    e2_count_quantile = np.quantile(e2_count['count'].values, q=0.6)


    plt.plot(np.cumsum(e2_count['count'].values)/e2_count['count'].values.sum())
    plt.title("sentences per each entity2, cumulative ")
    plt.show()

    df = df.merge(e2_count,
                  on="entity2",
                  how="left")
    df.rename(columns={"count": "e2_count"}, inplace=True)

    # ---
    # get entity pair count
    # ---

    e_pair = pd.pivot_table(df,
                            index=["entity1", "entity2", "relation"],
                            values="text",
                            aggfunc="count").reset_index()
    e_pair.sort_values("text", ascending=False, inplace=True)
    e_pair.rename(columns={'text': "epair_count"}, inplace=True)

    # entity pair quantile - for suppliers
    # - to help identify lesser mentioned pairs, which the assumption
    epair_q = np.quantile(e_pair.loc[e_pair["relation"] == "Supplier", 'epair_count'].values, q=0.6)

    # merge on the metric
    df = df.merge(e_pair,
                  on=["entity1", "entity2", "relation"],
                  how="left")

    # ----
    # company count per sentence - considering all companies in same sentence
    # ----

    # for each article - start_sent - end_sent triple get a all the (unique) companies that
    # exists for that sentence. dividing the number of company by num_tokens (or num_chars)
    # can give a measure of density
    # - the idea higher density implies less likely of describing a supplier relation
    # - at least for all combinations

    idx_col = ['date_publish', 'source_domain', 'article', "start_sent", "end_sent"]
    art_start_end = df[idx_col + ["entity1_full", "entity2_full"]].copy(True)
    # make a pair column, and then combine across idx_col, then count the unique
    art_start_end["pair"] = art_start_end[["entity1_full", "entity2_full"]].apply(lambda x: "|".join(x), axis=1)

    # combine across idx_col
    # TODO: double check the lambda function below
    # - lambda function: join all pairs with |, split the result with |, take unique, get length
    comb_ase = pd.pivot_table(art_start_end,
                              index=idx_col,
                              values="pair",
                              aggfunc=lambda x: len(np.unique(("|".join(x)).split("|"))))

    comb_ase.rename(columns={"pair": "companies_in_text"}, inplace=True)
    comb_ase.reset_index(inplace=True)

    df = df.merge(comb_ase,
                  on=idx_col,
                  how="left")

    # skip the measure of density, as it's harder to interpret
    # tmp["company_density"] = tmp["companies_in_text"] / tmp["num_tokens"]

    # ---
    # get total sentence count - by counting entity1 as well
    # ---
    #
    # e1_count = pd.pivot_table(df.loc[df['relation'] == "Supplier"],
    #                           index="entity1",
    #                           values="text",
    #                           aggfunc="count").reset_index()
    # e1_count.rename(columns={"text": "count"}, inplace=True)
    # e1_count.sort_values("count", ascending=False, inplace=True)
    #
    # df = df.merge(e1_count,
    #               on="entity1",
    #               how="left")
    # df.rename(columns={"count": "e1_count"}, inplace=True)
    #
    # e1_count_q = np.quantile(e1_count['count'].values, q=0.8)
    # e2_count_q = np.quantile(e2_count['count'].values, q=0.8)

    # --
    # combine
    # ---

    # e2_count.rename(columns={"entity2": "entity"}, inplace=True)
    # e_tot = pd.pivot_table(pd.concat([e1_count, e2_count]),
    #                        index=['entity'],
    #                        values="count",
    #                        aggfunc="sum").reset_index()
    #
    # e_tot.sort_values("count", ascending=False, inplace=True)

    # np.quantile(e_tot['count'].values, q=0.95)

    # -----
    # Labelling functions
    # -----

    # ref: https://www.snorkel.org/use-cases/01-spam-tutorial
    SUPPLIER = 1
    NO_REL = 0
    ABSTAIN = -1

    # TODO: revise these by doing some analysis on gold_labels (with sentences)
    # TODO: review if it matters having some of these separate
    # TODO: consider using

    # using a leading ' ' is to avoid matching in the middle of words

    @labeling_function()
    def regex_supply(x):
        return SUPPLIER if re.search(r" supply", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_supplier(x):
        return SUPPLIER if re.search(r" supplier", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_supplies(x):
        return SUPPLIER if re.search(r" supplies", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_buys(x):
        return SUPPLIER if re.search(r" buy | buys | buyer ", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_customer(x):
        return SUPPLIER if re.search(r" customer| client", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_make(x):
        return SUPPLIER if re.search(r" make | makes | maker", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_made(x):
        return SUPPLIER if re.search(r" made by| made for", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_sells(x):
        return SUPPLIER if re.search(r" sell | sells | seller ", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_sales(x):
        return SUPPLIER if re.search(r" sales", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_provides(x):
        return SUPPLIER if re.search(r" provide| provides", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_produces(x):
        return SUPPLIER if re.search(r" produce| produces", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_contract(x):
        return SUPPLIER if re.search(r" contract", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_shipments(x):
        return SUPPLIER if re.search(r" shipment", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_order(x):
        return SUPPLIER if re.search(r" order| ordered", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_agreement(x):
        return SUPPLIER if re.search(r" agreement", x.text, flags=re.I) else ABSTAIN


    @labeling_function()
    def regex_offer(x):
        return SUPPLIER if re.search(r" offer", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_serves(x):
        return SUPPLIER if re.search(r" serve| serves", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_deliver(x):
        return SUPPLIER if re.search(r" delivered| delivers", x.text, flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_used_by(x):
        return SUPPLIER if re.search(r" used by", x.text, flags=re.I) else ABSTAIN


    @labeling_function()
    def relation_na(x):
        """if there is not relation - that's probably the case so use it"""
        return NO_REL if x['Confidence Score (%)'] == 0 else ABSTAIN

    @labeling_function()
    def astrix_count(x):
        """reuters specific - if there are many *'s (more than 1) assume
        they represent bullet points - which are often unrelated (new bulletins)"""
        return NO_REL if len(re.findall("\* ", x.text)) >= 2 else ABSTAIN

    @labeling_function()
    def dash_count(x):
        """reuters specific - if there are many *'s (more than 1) assume
        they represent bullet points - which are often unrelated (new bulletins)"""
        return NO_REL if len(re.findall(" - ", x.text)) >= 3 else ABSTAIN

    @labeling_function()
    def dollar_sign_count(x):
        """reuters specific - if there are many *'s (more than 1) assume
        they represent bullet points - which are often unrelated (new bulletins)"""
        return NO_REL if len(re.findall("\$ ", x.text)) >= 4 else ABSTAIN

    @labeling_function()
    def arrow_count(x):
        """sometimes > are used for bullets, which are short, unrelated market comments"""
        return NO_REL if len(re.findall(" >", x.text)) >= 2 else ABSTAIN

    @labeling_function()
    def cap_q_count(x):
        """if there are many capital Q's in article then it's probably an earnings report"""
        return NO_REL if len(re.findall(" Q", x.text)) >= 10 else ABSTAIN

    @labeling_function()
    def percent_symbol_count(x):
        """articles with many % symbols are often market commentary"""
        return NO_REL if len(re.findall(" \%", x.text)) >= 2 else ABSTAIN

    @labeling_function()
    def percent_word_count(x):
        """articles with the word percent many time are often market commentary"""
        return NO_REL if len(re.findall(" percent", x.text, re.IGNORECASE)) >= 3 else ABSTAIN


    @labeling_function()
    def companies_in_text(x):
        """consider only the number of sentence e2 is in, with the idea the fewer the better
        - less popular suggests a relationship will be mentioned?
        NOTE: often all companies are not identified (using our KB) """
        return NO_REL if x.companies_in_text >= 5 else ABSTAIN


    @labeling_function()
    def epair_count(x):
        """if the (supplier) entity pair does not come up often then give it supplier label"""
        if x.relation == "Supplier":
            if x.epair_count > epair_q:
                return ABSTAIN
            else:
                return SUPPLIER
        else:
            return ABSTAIN


    @labeling_function()
    def e2_sentence_count(x):
        """consider only the number of sentence e2 is in, with the idea the fewer the better
        - less popular suggests a relationship will be mentioned?"""
        return SUPPLIER if (x.e2_count < e2_count_quantile) & (x.relation == "Supplier") else ABSTAIN


    # @labeling_function()
    # def e1_and_e2_sentence_count(x):
    #     """if supplier and e1 and e2 occur a lot then abstain """
    #     if x.relation == "Supplier":
    #         if (x.e2_count > e2_count_q) & (x.e1_count > e1_count_q):
    #             return ABSTAIN
    #         else:
    #             return SUPPLIER
    #     else:
    #         return ABSTAIN


    # ---
    # combine label functions
    # ----

    lfs = [
        regex_supply,
        regex_supplier,
        regex_supplies,
        regex_buys,
        regex_customer,
        regex_make,
        regex_made,
        regex_sells,
        regex_provides,
        regex_produces,
        regex_contract,
        regex_order,
        regex_deliver,
        regex_used_by,
        regex_agreement,
        regex_offer,
        regex_shipments,
        relation_na,
        astrix_count,
        arrow_count,
        dash_count,
        dollar_sign_count,
        cap_q_count,
        percent_symbol_count,
        percent_word_count,
        companies_in_text,
        epair_count,
        e2_sentence_count
    ]

    # ---
    # apply label functions
    # ---

    # select a subset
    # df_train = df.sample(20000, random_state=2)
    df_train = df.copy(True)

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)

    print((L_train != ABSTAIN).mean(axis=0))

    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    # ---
    # generate weak labels
    # ---

    majority_model = MajorityLabelVoter()
    preds_train = majority_model.predict(L=L_train)

    # using LabelModel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=550, log_freq=100, seed=123)

    preds_train = label_model.predict(L=L_train)

    df_train["weak_label"] = preds_train

    print(f"weak label abstained from: {100 * (preds_train < 0).mean(): .2f}%")
    print(f"no relation: {100 * (preds_train == 0).mean(): .2f}%")
    print(f"supplier: {100 * (preds_train == 1).mean(): .2f}%")

    # write to file
    # TODO: consider adding transformations
    # out["num_sentence"] += 1


    # # REMOVE THIS
    # probs_train = label_model.predict_proba(L_train)
    #
    # # include the prob
    # df_train["prob_label"] = probs_train[:,1]
    # res = df_train.to_dict("records")
    #
    # from nlp import get_data_path
    # with open(get_data_path("weak_label_temp.json"), "w") as f:
    #     json.dump(res, f, indent=4)
    #



    # --
    # probabilistic labelling and filtering out ABSTAIN
    # --

    probs_train = label_model.predict_proba(L_train)

    # include the prob
    df_train["prob_label"] = probs_train[:,1]
    res = df_train.to_dict("records")

    from nlp import get_data_path
    with open(get_data_path("text_with_weak_labels.json"), "w") as f:
        json.dump(res, f, indent=4)

    # select only the labelled (ignore abstained - should double check validate this)
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=probs_train, L=L_train
    )
    df_train_filtered = df_train_filtered.copy(True)


    # ----
    # investigate
    # -----

    #8837bdb10360390047271ad0872e79b6a9e7e040d729daf2361fb0418917a772_20170928142741_www.reuters.com_FordMotoCo_LyftInc_6|7

    # # --
    # # check weak labels
    # # --
    #
    # # negative cases
    # rel_na = df_train_filtered.loc[df_train_filtered["relation"] == "NA"]
    #
    # # true negative rate: weak label is 0 and relation is NA
    # # false positive rate: weak label is 1 and relation is NA
    # tn, fp = np.unique(rel_na["weak_label"].values, return_counts=True)[1] / len(rel_na)
    #
    # # 'positive' cases
    # # - note in here expect there to be many false positives
    # # i.e. when there is a supplier relation but the text does not provide that information
    # rel_sup = df_train_filtered.loc[df_train_filtered["relation"] == "Supplier"]
    #
    # # here assume if the weak label is 1 then it's a True positive (might not be in
    # fp, tp = np.unique(rel_sup["weak_label"].values, return_counts=True)[1] / len(rel_sup)
    #
    # # rel_sup.loc[rel_sup["weak_label"] == 1, "text"].values[100]
    #
    # # ---
    # # make transformations
    # # ---
    #
    # # https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial
    # # import names
    # from snorkel.augmentation import transformation_function
    # from snorkel.preprocess.nlp import SpacyPreprocessor
    #
    # spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)
    #
    # # Swap two adjectives at random.
    # @transformation_function(pre=[spacy])
    # def swap_adjectives(x):
    #     adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    #     # Check that there are at least two adjectives to swap.
    #     if len(adjective_idxs) >= 2:
    #         idx1, idx2 = sorted(np.random.choice(adjective_idxs, 2, replace=False))
    #         # Swap tokens in positions idx1 and idx2.
    #         x.text = " ".join(
    #             [
    #                 x.doc[:idx1].text,
    #                 x.doc[idx2].text,
    #                 x.doc[1 + idx1 : idx2].text,
    #                 x.doc[idx1].text,
    #                 x.doc[1 + idx2 :].text,
    #             ]
    #         )
    #         return x
    #
    #
    # # --
    # # using nltk
    # # ---
    #
    #
    #
    # nltk.download("wordnet")
    #
    #
    # def get_synonym(word, pos=None):
    #     """Get synonym for word given its part-of-speech (pos)."""
    #     synsets = wn.synsets(word, pos=pos)
    #     # Return None if wordnet has no synsets (synonym sets) for this word and pos.
    #     if synsets:
    #         words = [lemma.name() for lemma in synsets[0].lemmas()]
    #         if words[0].lower() != word.lower():  # Skip if synonym is same as word.
    #             # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
    #             return words[0].replace("_", " ")
    #
    #
    # def replace_token(spacy_doc, idx, replacement):
    #     """Replace token in position idx with replacement."""
    #     return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])
    #
    #
    # @transformation_function(pre=[spacy])
    # def replace_verb_with_synonym(x):
    #     # Get indices of verb tokens in sentence.
    #     verb_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "VERB"]
    #     if verb_idxs:
    #         # Pick random verb idx to replace.
    #         idx = np.random.choice(verb_idxs)
    #         synonym = get_synonym(x.doc[idx].text, pos="v")
    #         # If there's a valid verb synonym, replace it. Otherwise, return None.
    #         if synonym:
    #             x.text = replace_token(x.doc, idx, synonym)
    #             return x
    #
    #
    # @transformation_function(pre=[spacy])
    # def replace_noun_with_synonym(x):
    #     # Get indices of noun tokens in sentence.
    #     noun_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
    #     if noun_idxs:
    #         # Pick random noun idx to replace.
    #         idx = np.random.choice(noun_idxs)
    #         synonym = get_synonym(x.doc[idx].text, pos="n")
    #         # If there's a valid noun synonym, replace it. Otherwise, return None.
    #         if synonym:
    #             x.text = replace_token(x.doc, idx, synonym)
    #             return x
    #
    #
    # @transformation_function(pre=[spacy])
    # def replace_adjective_with_synonym(x):
    #     # Get indices of adjective tokens in sentence.
    #     adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    #     if adjective_idxs:
    #         # Pick random adjective idx to replace.
    #         idx = np.random.choice(adjective_idxs)
    #         synonym = get_synonym(x.doc[idx].text, pos="a")
    #         # If there's a valid adjective synonym, replace it. Otherwise, return None.
    #         if synonym:
    #             x.text = replace_token(x.doc, idx, synonym)
    #             return x
    #
    #
    # tfs = [
    #     # change_person,
    #     swap_adjectives,
    #     replace_verb_with_synonym,
    #     replace_noun_with_synonym,
    #     replace_adjective_with_synonym,
    # ]
    #
    # # from utils import preview_tfs
    # # copied from: https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/utils.py
    #
    # def preview_tfs(df, tfs):
    #     transformed_examples = []
    #     for f in tfs:
    #         for i, row in df.sample(frac=1, random_state=2).iterrows():
    #             transformed_or_none = f(row)
    #             # If TF returned a transformed example, record it in dict and move to next TF.
    #             if transformed_or_none is not None:
    #                 transformed_examples.append(
    #                     OrderedDict(
    #                         {
    #                             "TF Name": f.name,
    #                             "Original Text": row.text,
    #                             "Transformed Text": transformed_or_none.text,
    #                         }
    #                     )
    #                 )
    #                 break
    #     return pd.DataFrame(transformed_examples)
    #
    # preview_tfs(rel_sup, tfs)
    #
    # # # Random policy
    # # random_policy = RandomPolicy(
    # #     len(tfs), sequence_length=2, n_per_original=2, keep_original=True
    # # )
    #
    # mean_field_policy = MeanFieldPolicy(
    #     len(tfs),
    #     sequence_length=2,
    #     n_per_original=2,
    #     keep_original=True,
    #     p=[0.1, 0.3, 0.3, 0.3],
    # )
    #
    # tf_applier = PandasTFApplier(tfs, mean_field_policy)
    # rel_sup_aug = tf_applier.apply( rel_sup.loc[rel_sup["weak_label"] == 1])
    # Y_train_augmented = rel_sup_aug["weak_label"].values
    #
    # # ---
    # # write to file
    # # ----
    #
    # df_train.drop(["e2_count", "e1_count"], axis=1, inplace=True)
    # df_train["augmented"] = False
    # rel_sup_aug.drop(["e2_count", "e1_count"], axis=1, inplace=True)
    # rel_sup_aug["augmented"] = True
    #
    # out = pd.concat([df_train, rel_sup_aug], axis=0)
    #
    # # out["num_sentence"] += 1
    # res = out.to_dict("records")
    #
    # from nlp import get_data_path
    # with open(get_data_path("text_with_weak_labels.json"), "w") as f:
    #     json.dump(res, f, indent=4)


    # - this is slow and not really needed
    # # use the knowledge base to give label
    # def kb_label(x, kb, s2l_entity):
    #     # this causes issue when tring to compile
    #     # try:
    #     #     e1 = re.search('\{(.*?)\}', x)[1]
    #     #     e2 = re.search('\((.*?)\)', x)[1]
    #     # except Exception as e:
    #     #     return ABSTAIN
    #
    #     # HACK: to hand if if a series is provided.. this is not done well
    #     if not isinstance(x, str):
    #         x = x[0]
    #     e1 = x[x.find("{")+1:x.find("}")]
    #     e2 = x[x.find("(")+1:x.find(")")]
    #
    #     # get the long names
    #     # NOTE: here there can be multiple matches - what follows will be
    #     # more liberal in saying there is a supplier relation ship
    #     # i.e. it could be Rolls-Royce Holdings PLC and Rolls-Royce PLC
    #     # will be considered the same for short name Rolls-Royce
    #     e1_long = s2l_entity.loc[s2l_entity['entity'] == e1, 'entity_full'].values
    #     e2_long = s2l_entity.loc[s2l_entity['entity'] == e2, 'entity_full'].values
    #
    #     # TODO: could allow for ABSTAINING if unsure
    #     # NOTE: the following assumes only one relation
    #     if np.any(kb['entity1'].isin(e1_long) & kb['entity2'].isin(e2_long)):
    #         return SUPPLIER
    #     else:
    #         return NO_REL
    #
    # kb_label_lf = LabelingFunction(name="kb_label",
    #                                f=kb_label,
    #                                resources={"kb": kb,
    #                                           "s2l_entity": s2l_entity})
