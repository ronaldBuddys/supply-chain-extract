
import json
import os
from collections import OrderedDict

from snorkel.augmentation import transformation_function
from snorkel.preprocess.nlp import SpacyPreprocessor

from snorkel.augmentation import PandasTFApplier
from snorkel.augmentation import MeanFieldPolicy
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


spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

@transformation_function(pre=[spacy])
def swap_adjectives(x):
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    # Check that there are at least two adjectives to swap.
    if len(adjective_idxs) >= 2:
        idx1, idx2 = sorted(np.random.choice(adjective_idxs, 2, replace=False))
        # Swap tokens in positions idx1 and idx2.
        x.text = " ".join(
            [
                x.doc[:idx1].text,
                x.doc[idx2].text,
                x.doc[1 + idx1 : idx2].text,
                x.doc[idx1].text,
                x.doc[1 + idx2 :].text,
            ]
        )
        return x


def get_synonym(word, pos=None):
    """Get synonym for word given its part-of-speech (pos)."""
    synsets = wn.synsets(word, pos=pos)
    # Return None if wordnet has no synsets (synonym sets) for this word and pos.
    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        if words[0].lower() != word.lower():  # Skip if synonym is same as word.
            # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
            return words[0].replace("_", " ")


def replace_token(spacy_doc, idx, replacement):
    """Replace token in position idx with replacement."""
    return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])


@transformation_function(pre=[spacy])
def replace_verb_with_synonym(x):
    # Get indices of verb tokens in sentence.
    verb_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "VERB"]
    if verb_idxs:
        # Pick random verb idx to replace.
        idx = np.random.choice(verb_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="v")
        # If there's a valid verb synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x


@transformation_function(pre=[spacy])
def replace_noun_with_synonym(x):
    # Get indices of noun tokens in sentence.
    noun_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
    if noun_idxs:
        # Pick random noun idx to replace.
        idx = np.random.choice(noun_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="n")
        # If there's a valid noun synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x


@transformation_function(pre=[spacy])
def replace_adjective_with_synonym(x):
    # Get indices of adjective tokens in sentence.
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    if adjective_idxs:
        # Pick random adjective idx to replace.
        idx = np.random.choice(adjective_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="a")
        # If there's a valid adjective synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x


# from utils import preview_tfs
# copied from: https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/utils.py

def preview_tfs(df, tfs):
    transformed_examples = []
    for f in tfs:
        for i, row in df.sample(frac=1, random_state=2).iterrows():
            transformed_or_none = f(row)
            # If TF returned a transformed example, record it in dict and move to next TF.
            if transformed_or_none is not None:
                transformed_examples.append(
                    OrderedDict(
                        {
                            "TF Name": f.name,
                            "Original Text": row.text,
                            "Transformed Text": transformed_or_none.text,
                        }
                    )
                )
                break
    return pd.DataFrame(transformed_examples)



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

    # data splits to add augmentation to
    use_splits = [0, 1]

    # number of augmentations to apply per instance (?)
    sequence_length = 2

    # number of augmented instances per original (targeted, may not produce this many)
    n_per_original = 2

    # split data
    split_file = get_data_path("df_full.tsv")

    # weak label file - generated from weak_labels.py
    weak_label_file = get_data_path("weak_labels.csv")

    # output file
    out_file = get_data_path("df_full_aug_wl.tsv")

    # ---
    # read in the split data
    # ---

    split_data = pd.read_csv(split_file, sep="\t")

    df = split_data.loc[split_data["split"].isin(use_splits)].copy(True)

    # ---
    # weak label data
    # ---

    wl = pd.read_csv(weak_label_file)

    # ---
    # merge on weak label
    # ---

    df.drop(["weak_label", "prob_label"], axis=1, inplace=True)

    df = df.merge(wl,
                  left_on="id",
                  right_on="label_id",
                  how="left")

    df.drop("label_id", axis=1, inplace=True)

    # ---
    # rename columns - set text -> text_org and text_with_markers -> text
    # ---

    df.rename(columns={"text": "text_org", "text_with_marker": "text"}, inplace=True)

    # ---
    # make transformations
    # ---

    # apply transformations only where weak_label = 1
    # - because those are the more rare cases we want to increase
    df1 = df.loc[df["weak_label"] == 1].copy(True)

    # https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial
    # import names

    # --
    # using nltk
    # ---

    nltk.download("wordnet")

    # replace verb, noun, and adjective with synonym with goal of not changing meaning
    # of sentence
    tfs = [
        # change_person,
        # swap_adjectives,
        replace_verb_with_synonym,
        replace_noun_with_synonym,
        replace_adjective_with_synonym,
    ]


    # preview_tfs(rel_sup, tfs)

    # # Random policy
    random_policy = RandomPolicy(
        len(tfs), sequence_length=2, n_per_original=2, keep_original=False
    )

    # mean_field_policy = MeanFieldPolicy(
    #     len(tfs),
    #     sequence_length=2,
    #     n_per_original=2,
    #     keep_original=True,
    #     p=[0.1,0.3, 0.3, 0.3],
    # )

    tf_applier = PandasTFApplier(tfs, random_policy)
    df1_aug = tf_applier.apply( df1)
    Y_train_augmented = df1_aug["weak_label"].values

    # ---
    # rename columns back to original
    # ---

    df1_aug.rename(columns={"text_org": "text", "text": "text_with_marker"}, inplace=True)

    # contact augmented data with original split data
    out = pd.concat([split_data, df1_aug])


    # ---
    # write to file
    # ----

    out.to_csv(out_file, sep='\t', index=False)


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
    #
    #
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
