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

    # TODO: clean this file up
    # TODO: remove all content inbetween () and {}
    # TODO: replace entities to be wrapped with {} and ()
    # TODO: create labelling function to use KB -
    # TODO: consider hardcoding no relation labels to be 0

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_colwidth", 70)

    # ----
    # parameters
    # ----

    # TODO: should really store parameters (and label function names) used to generate weak labels

    # input sentence file
    sent_file = get_data_path("processed_sentences.json")

    # this can be None
    # class_balance = [0.9, 0.1]
    class_balance = None

    # output file
    output_file = get_data_path(f"weak_labels{'' if class_balance is None else '_w_class_balance' }.csv")

    # ---
    # read in value chain data / knowledge base
    # ---

    # read in locally stored valued chains
    vc = pd.read_csv(get_data_path("VCHAINS.csv"))

    kb = get_knowledge_base_from_value_chain_data(vc)

    # ---
    # read in full_sentences store locally
    # ---

    assert os.path.exists(sent_file), \
        f"looks like: {sent_file}, copy from google drive data/{os.path.basename(sent_file)}"

    with open(sent_file, "r") as f:
        full_sents = json.load(f)

    df = pd.DataFrame(full_sents)

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

    # ----
    # sentence 'size'
    # ----

    # num_sent = pd.pivot_table(df,
    #                           index='num_sentence',
    #                           values="text",
    #                           aggfunc="count").reset_index()

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

    # e2_count = pd.pivot_table(df.loc[df['relation'] == "Supplier"],
    #                            index="entity2",
    #                            values="text",
    #                            aggfunc="count").reset_index()
    # e2_count.rename(columns={"text": "count"}, inplace=True)
    # e2_count.sort_values("count", ascending=False, inplace=True)
    #
    # e2_count_quantile = np.quantile(e2_count['count'].values, q=0.6)
    #
    #
    # plt.plot(np.cumsum(e2_count['count'].values)/e2_count['count'].values.sum())
    # plt.title("sentences per each entity2, cumulative ")
    # plt.show()
    #
    # df = df.merge(e2_count,
    #               on="entity2",
    #               how="left")
    # df.rename(columns={"count": "e2_count"}, inplace=True)

    # ---
    # get entity pair count
    # ---

    # e_pair = pd.pivot_table(df,
    #                         index=["entity1", "entity2", "relation"],
    #                         values="text",
    #                         aggfunc="count").reset_index()
    # e_pair.sort_values("text", ascending=False, inplace=True)
    # e_pair.rename(columns={'text': "epair_count"}, inplace=True)
    #
    # # entity pair quantile - for suppliers
    # # - to help identify lesser mentioned pairs, which the assumption
    # epair_q = np.quantile(e_pair.loc[e_pair["relation"] == "Supplier", 'epair_count'].values, q=0.6)
    #
    # # merge on the metric
    # df = df.merge(e_pair,
    #               on=["entity1", "entity2", "relation"],
    #               how="left")

    # ----
    # companies_in_text - considering all companies (from KB) in same sentence
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
    # comb_ase.reset_index(inplace=True)

    df = df.merge(comb_ase,
                  on=idx_col,
                  how="left")

    # skip the measure of density, as it's harder to interpret
    # tmp["company_density"] = tmp["companies_in_text"] / tmp["num_tokens"]


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
        return SUPPLIER if re.search(r" make| makes| maker", x.text, flags=re.I) else ABSTAIN

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
    def relation_pos(x):
        """if there is not relation - that's probably the case so use it"""
        return SUPPLIER if x['Confidence Score (%)'] >= 0.99 else ABSTAIN

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


    # drop these ?
    # @labeling_function()
    # def epair_count(x):
    #     """if the (supplier) entity pair does not come up often then give it supplier label"""
    #     if x.relation == "Supplier":
    #         if x.epair_count > epair_q:
    #             return ABSTAIN
    #         else:
    #             return SUPPLIER
    #     else:
    #         return ABSTAIN
    #
    #
    # @labeling_function()
    # def e2_sentence_count(x):
    #     """consider only the number of sentence e2 is in, with the idea the fewer the better
    #     - less popular suggests a relationship will be mentioned?"""
    #     return SUPPLIER if (x.e2_count < e2_count_quantile) & (x.relation == "Supplier") else ABSTAIN


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

    # uncomment below for slightly worse performance
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
        # regex_contract,
        regex_order,
        regex_deliver,
        regex_used_by,
        regex_agreement,
        # regex_offer,
        # regex_shipments,
        relation_na,
        relation_pos,
        astrix_count,
        arrow_count,
        dash_count,
        dollar_sign_count,
        cap_q_count,
        percent_symbol_count,
        percent_word_count,
        companies_in_text,
        # don't include these
        # epair_count,
        # e2_sentence_count
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

    # # using LabelModel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train,
                    # Y_dev= Y_dev,
                    class_balance=class_balance,
                    # default parameters
                    n_epochs=500,
                    log_freq=100,
                    seed=123)

    preds_train = label_model.predict(L=L_train)

    # print((L_train != ABSTAIN).mean(axis=0))

    # print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    df_train["weak_label"] = preds_train

    print(f"weak label abstained from: {100 * (preds_train < 0).mean(): .2f}%")
    print(f"no relation: {100 * (preds_train == 0).mean(): .2f}%")
    print(f"supplier: {100 * (preds_train == 1).mean(): .2f}%")

    # --
    # probabilistic labelling and filtering out ABSTAIN
    # --

    probs_train = label_model.predict_proba(L_train)

    # include the prob
    df_train["prob_label"] = probs_train[:, 1]
    # res = df_train.to_dict("records")

    # ---
    # write to file
    # ---

    label_cols = ["id", "weak_label", "prob_label"]

    out = df_train[label_cols].copy(True)

    out.rename(columns={"id": "label_id"}, inplace=True)

    out.to_csv(output_file, index=False)

    # from nlp import get_data_path
    # with open(get_data_path("text_with_weak_labels.json"), "w") as f:
    #     json.dump(res, f, indent=4)
    #
    # # select only the labelled (ignore abstained - should double check validate this)
    # df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    #     X=df_train, y=probs_train, L=L_train
    # )
    # df_train_filtered = df_train_filtered.copy(True)

    # ----
    # Analysis on gold labels - MOVE THIS ELSEWHERE - shouldn't be part of this script
    # ---

    print("peaking ahead")

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

    client.close()

    # merge on gold labels
    sd = df_train.merge(gl, left_on="id", right_on="label_id", how="inner")

    # sd['id'].unique().shape

    # map gold labels to 0 or 1
    neg_vals = ["NA", "competitor", "reverse"]
    # - some of these might be generous
    pos_vals = ["Supplier"]#, "partnership", "owner"]
    label_map = {i: 1 if i in pos_vals else 0 for i in sd['gold_label'].unique()}

    sd['gl'] = sd['gold_label'].map(label_map)

    _ = sd.loc[sd["weak_label"] != -1]
    print(f'weak = gold: {(_["weak_label"] == _["gl"]).mean()}')

    _ = sd.loc[sd["weak_label"] == 1]

    print(f'weak = gold: {(_["weak_label"] == _["gl"]).mean()}')

    # LFAnalysis(L_train, lfs).lf_summary()

    applier = PandasLFApplier(lfs=lfs)
    L_sd = applier.apply(df=sd)

    print((L_sd != ABSTAIN).mean(axis=0))

    print(LFAnalysis(L=L_sd, lfs=lfs).lf_summary())

    print(LFAnalysis(L_sd, lfs).lf_summary(sd['gl'].values))


    # ---------------





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