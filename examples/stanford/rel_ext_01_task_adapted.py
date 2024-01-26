#!/usr/bin/env python
# coding: utf-8

# # Relation extraction using distant supervision: task definition

# In[1]:


__author__ = "Bill MacCartney and Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


# ## Contents
# 
# 1. [Overview](#Overview)
#   1. [The task of relation extraction](#The-task-of-relation-extraction)
#   1. [Hand-built patterns](#Hand-built-patterns)
#   1. [Supervised learning](#Supervised-learning)
#   1. [Distant supervision](#Distant-supervision)
# 1. [Set-up](#Set-up)
# 1. [The corpus](#The-corpus)
# 1. [The knowledge base](#The-knowledge-base)
# 1. [Problem formulation](#Problem-formulation)
#   1. [Joining the corpus and the KB](#Joining-the-corpus-and-the-KB)
#   1. [Negative instances](#Negative-instances)
#   1. [Multi-label classification](#Multi-label-classification)
#   1. [Building datasets](#Building-datasets)
# 1. [Evaluation](#Evaluation)
#   1. [Splitting the data](#Splitting-the-data)
#   1. [Choosing evaluation metrics](#Choosing-evaluation-metrics)
#   1. [Running evaluations](#Running-evaluations)
#   1. [Evaluating a random-guessing strategy](#Evaluating-a-random-guessing-strategy)
# 1. [A simple baseline model](#A-simple-baseline-model)

# ## Overview
# 
# This notebook illustrates an approach to [relation extraction](http://deepdive.stanford.edu/relation_extraction) using [distant supervision](http://deepdive.stanford.edu/distant_supervision). It uses a simplified version of the approach taken by Mintz et al. in their 2009 paper, [Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113). If you haven't yet read that paper, read it now! The rest of the notebook will make a lot more sense after you're familiar with it.
# 
# ### The task of relation extraction
# 
# Relation extraction is the task of extracting from natural language text relational triples such as:
# 
# ```
# (founders, SpaceX, Elon_Musk)
# (has_spouse, Elon_Musk, Talulah_Riley)
# (worked_at, Elon_Musk, Tesla_Motors)
# ```
# 
# If we can accumulate a large knowledge base (KB) of relational triples, we can use it to power question answering and other applications. Building a KB manually is slow and expensive, but much of the knowledge we'd like to capture is already expressed in abundant text on the web. The aim of relation extraction, therefore, is to accelerate the construction of new KBs — and facilitate the ongoing curation of existing KBs — by extracting relational triples from natural language text.
# 
# ### Hand-built patterns
# 
# An obvious way to start is to write down a few patterns which express each relation. For example, we can use the pattern "X is the founder of Y" to find new instances of the `founders` relation. If we search a large corpus, we may find the phrase "Elon Musk is the founder of SpaceX", which we can use as evidence for the relational triple `(founders, SpaceX, Elon_Musk)`.
# 
# Unfortunately, this approach doesn't get us very far. The central challenge of relation extraction is the fantastic diversity of language, the multitude of possible ways to express a given relation. For example, each of the following sentences expresses the relational triple `(founders, SpaceX, Elon_Musk)`:
# 
# - "You may also be thinking of *Elon Musk* (founder of *SpaceX*), who started PayPal."
# - "Interesting Fact: *Elon Musk*, co-founder of PayPal, went on to establish *SpaceX*, one of the most promising space travel startups in the world."
# - "If Space Exploration (*SpaceX*), founded by Paypal pioneer *Elon Musk* succeeds, commercial advocates will gain credibility and more support in Congress."
# 
# The patterns which connect "Elon Musk" with "SpaceX" in these examples are not ones we could have easily anticipated. To do relation extraction effectively, we need to go beyond hand-built patterns.
# 
# ### Supervised learning
# 
# Effective relation extraction will require applying machine learning methods. The natural place to start is with supervised learning. This means training an extraction model from a dataset of examples which have been labeled with the target output. Sentences like the three examples above would be annotated with the `founders` relation, but we'd also have sentences which include "Elon Musk" and "SpaceX" but do not express the `founders` relation, such as:
# 
# - "Billionaire entrepreneur *Elon Musk* announced the latest addition to the *SpaceX* arsenal: the 'Big F---ing Rocket' (BFR)".
# 
# Such "negative examples" would be labeled as such, and the fully-supervised model would then be able to learn from both positive and negative examples the linguistic patterns that indicate each relation.
# 
# The difficulty with the fully-supervised approach is the cost of generating training data. Because of the great diversity of linguistic expression, our model will need lots and lots of training data: at least tens of thousands of examples, although hundreds of thousands or millions would be much better. But labeling the examples is just as slow and expensive as building the KB by hand would be.
# 
# ### Distant supervision
# 
# The goal of distant supervision is to capture the benefits of supervised learning without paying the cost of labeling training data. Instead of labeling extraction examples by hand, we use existing relational triples to automatically identify extraction examples in a large corpus. For example, if we already have in our KB the relational triple `(founders, SpaceX, Elon_Musk)`, we can search a large corpus for sentences in which "SpaceX" and "Elon Musk" co-occur, make the (unreliable!) assumption that all the sentences express the `founder` relation, and then use them as training data for a learned model to identify new instances of the `founder` relation — all without doing any manual labeling.
# 
# This is a powerful idea, but it has two limitations. The first is that, inevitably, some of the sentences in which "SpaceX" and "Elon Musk" co-occur will not express the `founder` relation — like the BFR example above. By making the blind assumption that all such sentences do express the `founder` relation, we are essentially injecting noise into our training data, and making it harder for our learning algorithms to learn good models. Distant supervision is effective in spite of this problem because it makes it possible to leverage vastly greater quantities of training data, and the benefit of more data outweighs the harm of noisier data.
# 
# The second limitation is that we need an existing KB to start from. We can only train a model to extract new instances of the `founders` relation if we already have many instances of the `founders` relation. Thus, while distant supervision is a great way to extend an existing KB, it's not useful for creating a KB containing new relations from scratch.
# 
# \[ [top](#Relation-extraction-using-distant-supervision:-task-definition) \]

# ## Set-up
# 
# *  Make sure your environment includes all the requirements for [the cs224u repository](https://github.com/cgpotts/cs224u).
# 
# * If you haven't already, download [the course data](http://web.stanford.edu/class/cs224u/data/data.tgz), unpack it, and place it in the directory containing the course repository – the same directory as this notebook. (If you want to put it somewhere else, change `rel_ext_data_home` below.)

# # Make sure the files 'example_inputs_long_names_pos_and_neg.tsv.gz' and 'example_kb.tsv.gz' are in the data folder. These are created using the value_chain_data_exploration notebook in the examples folder

# In[13]:


import random
import os
from collections import Counter, defaultdict
from nlp.examples.
import utils
import sys


# In[14]:


# Set all the random seeds for reproducibility. Only the
# system seed is relevant for this notebook.

utils.fix_random_seeds()


# In[15]:


try:
    # python package (nlp) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


# In[16]:


rel_ext_data_home = os.path.join(src_path,'data')
rel_ext_data_home


# ## The corpus

# As usual when we're doing NLP, we need to start with a _corpus_ — a large sample of natural language text. And because our goal is to do relation extraction with distant supervision, we need to be able to identify entities in the text and connect them to a knowledge base of relations between entities. So, we need a corpus in which entity mentions are annotated with _entity resolutions_ which map them to unique, unambiguous identifiers. Entity resolution serves two purposes:
# 
# 1. It ensures that if an entity mention could refer to two different entities, it is properly disambiguated. For example, "New York" could refer to the city or the state.
# 1. It ensures that if two different entity mentions refer to the same entity, they are properly identified. For example, both "New York City" and "The Big Apple" refer to New York City.
# 
# The corpus we'll use for this project is derived from the [Wikilinks dataset](https://code.google.com/archive/p/wiki-links/) [announced by Google in 2013](https://research.googleblog.com/2013/03/learning-from-big-data-40-million.html). This dataset contains over 40M mentions of 3M distinct entities spanning 10M webpages. It provides entity resolutions by mapping each entity mention to a Wikipedia URL.
# 
# Now, in order to do relation extraction, we actually need _pairs_ of entity mentions, and it's important to have the context around and between the two mentions. Fortunately, UMass has provided an [expanded version of Wikilinks](http://www.iesl.cs.umass.edu/data/data-wiki-links) which includes the context around each entity mention. We've written code to stitch together pairs of entity mentions along with their contexts, and we've filtered the examples extensively. The result is a compact corpus suitable for our purposes.
# 
# Because we're frequently going to want to retrieve corpus examples containing specific entities, we've created a `Corpus` class which holds not only the examples themselves, but also a precomputed index. Let's take a closer look.

# In[17]:


corpus = rel_ext.Corpus(os.path.join(rel_ext_data_home, 'example_inputs_long_names_pos_and_neg.tsv.gz'))

print('Read {0:,} examples'.format(len(corpus)))


# Great, that's a lot of examples! Let's take a closer look at one.

# In[18]:


print(corpus.examples[1])


# In[19]:


for element in corpus.examples[1]:
    print(element)


# Every example represents a fragment of webpage text containing two entity mentions. The first two fields, `entity_1` and `entity_2`, contain unique identifiers for the two entities mentioned. We name entities using Wiki IDs, which you can think of as the last portion of a Wikipedia URL. Thus the Wiki ID `Barack_Obama` designates the entity described by [https://en.wikipedia.org/wiki/Barack_Obama](https://en.wikipedia.org/wiki/Barack_Obama).
# 
# The next five fields represent the text surrounding the two mentions, divided into five chunks: `left` contains the text before the first mention, `mention_1` is the first mention itself, `middle` contains the text between the two mentions, `mention_2` is the second mention, and the field `right` contains the text after the second mention. Thus, we can reconstruct the context as a single string like this:

# In[20]:


ex = corpus.examples[1]

' '.join((ex.left, ex.mention_1, ex.middle, ex.mention_2, ex.right))


# The last five fields contain the same five chunks of text, but this time annotated with part-of-speech (POS) tags, which may turn out to be useful when we start building models for relation extraction.
# 
# Let's look at the distribution of entities over the corpus. How many entities are there, and what are the most common ones?

# In[21]:


counter = Counter()
for example in corpus.examples:
    counter[example.entity_1] += 1
    counter[example.entity_2] += 1
print('The corpus contains {} entities'.format(len(counter)))
counts = sorted([(count, key) for key, count in counter.items()], reverse=True)
print('The most common entities are:')
for count, key in counts[:20]:
    print('{:10d} {}'.format(count, key))


# The main benefit we gain from the `Corpus` class is the ability to retrieve examples containing specific entities. Let's find examples containing `Elon_Musk` and `Tesla_Motors`.

# In[22]:


corpus.show_examples_for_pair('Apple Inc', 'Corning Inc')


# Actually, this might not be all of the examples containing `Elon_Musk` and `Tesla_Motors`. It's only the examples where `Elon_Musk` was mentioned first and `Tesla_Motors` second. There may be additional examples that have them in the reverse order. Let's check.

# In[23]:


corpus.show_examples_for_pair('Corning Inc', 'Apple Inc')


# Sure enough. Going forward, we'll have to remember to check both "directions" when we're looking for examples contains a specific pair of entities.
# 
# This corpus is not without flaws. As you get more familiar with it, you will likely discover that it contains many examples that are nearly — but not exactly — duplicates. This seems to be a consequence of the web document sampling methodology that was used in the construction of the Wikilinks dataset. However, despite a few warts, it will serve our purposes.
# 
# One thing this corpus does _not_ include is any annotation about relations. Thus, it could not be used for the fully-supervised approach to relation extraction, because the fully-supervised approach requires that each pair of entity mentions be annotated with the relation (if any) that holds between the two entities. In order to make any headway, we'll need to connect the corpus with an external source of knowledge about relations. We need a knowledge base.
# 
# \[ [top](#Relation-extraction-using-distant-supervision:-task-definition) \]

# ## The knowledge base

# The data distribution for this unit includes a _knowledge base_ (KB) ultimately derived from [Freebase](https://en.wikipedia.org/wiki/Freebase). Unfortunately, Freebase was shut down in 2016, but the Freebase data is still available from various sources and in various forms. The KB included here was extracted from the [Freebase Easy data dump](http://freebase-easy.cs.uni-freiburg.de/dump/).
# 
# The KB is a collection of *relational triples*, each consisting of a *relation*, a *subject*, and an *object*. For example, here are three triples from the KB:
# 
# ```
# (place_of_birth, Barack_Obama, Honolulu)
# (has_spouse, Barack_Obama, Michelle_Obama)
# (author, The_Audacity_of_Hope, Barack_Obama)
# ```
# 
# As you might guess:
# 
# - The relation is one of a handful of predefined constants, such as `place_of_birth` or `has_spouse`.
# - The subject and object are entities represented by Wiki IDs (that is, suffixes of Wikipedia URLs).
# 
# Now, just as we did for the corpus, we've created a `KB` class to store the KB triples and some associated indexes. This class makes it easy and efficient to look up KB triples both by relation and by entities.

# In[24]:


kb = rel_ext.KB(os.path.join(rel_ext_data_home, 'example_kb.tsv.gz'))

print('Read {0:,} KB triples'.format(len(kb)))


# Let's get a sense of the high-level characteristics of this KB. Some questions we'd like to answer:
# 
# - How many relations are there?
# - How big is each relation?
# - Examples of each relation.
# - How many unique entities does the KB include?

# In[25]:


len(kb.all_relations)


# How big is each relation? That is, how many triples does each relation contain?

# In[26]:


for rel in kb.all_relations:
    print('{:12d} {}'.format(len(kb.get_triples_for_relation(rel)), rel))


# Let's look at one example from each relation, so that we can get a sense of what they mean.

# In[27]:


for rel in kb.all_relations:
    print(tuple(kb.get_triples_for_relation(rel)[0]))


# The `kb.get_triples_for_entities()` method allows us to look up triples by the entities they contain. Let's use it to see what relation(s) hold between `France` and `Germany`.

# In[28]:


kb.get_triples_for_entities('Apple Inc', 'Corning Inc')


# Relations like `adjoins` and `has_sibling` are intuitively symmetric — if the relation holds between _X_ and _Y_, then we expect it to hold between _Y_ and _X_ as well.

# In[29]:


#Here we actually expect the inverse -> -100% correlation between both relations
kb.get_triples_for_entities('Corning Inc', 'Apple Inc')


# However, there's no guarantee that all such inverse triples actually appear in the KB. (You could write some code to check.)
# 
# Most relations, however, are intuitively asymmetric. Let's see what relation holds between `Tesla_Motors` and `Elon_Musk`.

# In[30]:


#kb.get_triples_for_entities('Tesla_Motors', 'Elon_Musk')


# It's a bit arbitrary that the KB includes a given asymmetric relation rather than its inverse. For example, instead of the `founders` relation with triple `(founders, Tesla_Motors, Elon_Musk)`, we might have had a `founder_of` relation with triple `(founder_of, Elon_Musk, Tesla_Motors)`. It doesn't really matter.
# 
# Although we don't have a `founder_of` relation, there might still be a relation between `Elon_Musk` and `Tesla_Motors`. Let's check.

# In[31]:


#kb.get_triples_for_entities('Elon_Musk', 'Tesla_Motors')


# Aha, yes, that makes sense. So it can be the case that one relation holds between _X_ and _Y_, and a different relation holds between _Y_ and _X_.
# 
# One more observation: there may be more than one relation that holds between a given pair of entities, even in one direction.

# In[32]:


#kb.get_triples_for_entities('Cleopatra', 'Ptolemy_XIII_Theos_Philopator')


# No! What? Yup, it's true — [Cleopatra](https://en.wikipedia.org/wiki/Cleopatra) married her younger brother, [Ptolemy XIII](https://en.wikipedia.org/wiki/Ptolemy_XIII_Theos_Philopator). Wait, it gets worse — she also married her _even younger_ brother, [Ptolemy XIV](https://en.wikipedia.org/wiki/Ptolemy_XIV_of_Egypt). Apparently this was normal behavior in ancient Egypt.
# 
# Moving on ...
# 
# Let's look at the distribution of entities in the KB. How many entities are there, and what are the most common ones?

# In[33]:


counter = Counter()
for kbt in kb.kb_triples:
    counter[kbt.sbj] += 1
    counter[kbt.obj] += 1
print('The KB contains {:,} entities'.format(len(counter)))
counts = sorted([(count, key) for key, count in counter.items()], reverse=True)
print('The most common entities are:')
for count, key in counts[:20]:
    print('{:10d} {}'.format(count, key))


# From Tim:
# Currenlty our News Corpus is v small so our entity count is low. We might want to check for some pre-written work on recognising corporate entities in news

# The number of entities in the KB is less than half the number of entities in the corpus! Evidently the corpus has much broader coverage than the KB.
# 
# Note that there is no promise or expectation that this KB is _complete_. Not only does the KB contain no mention of many entities from the corpus — even for the entities it does include, there may be possible triples which are true in the world but are missing from the KB. As an example, these triples are in the KB:
# 
# ```
# (founders, SpaceX, Elon_Musk)
# (founders, Tesla_Motors, Elon_Musk)
# (worked_at, Elon_Musk, Tesla_Motors)
# ```
# 
# but this one is not:
# 
# ```
# (worked_at, Elon_Musk, SpaceX)
# ```
# 
# In fact, the whole point of developing methods for automatic relation extraction is to extend existing KBs (and build new ones) by identifying new relational triples from natural language text. If our KBs were complete, we wouldn't have anything to do.
# 
# \[ [top](#Relation-extraction-using-distant-supervision:-task-definition) \]

# ## Problem formulation
# 
# With our data assets in hand, it's time to provide a precise formulation of the prediction problem we aim to solve. We need to specify:
# 
# - What is the input to the prediction?
#     - Is it a specific pair of entity *mentions* in a specific context?
#     - Or is it a pair of *entities*, apart from any specific mentions?
# - What is the output of the prediction?
#     - Do we need to predict at most one relation label? (This is [multi-class classification](https://en.wikipedia.org/wiki/Multiclass_classification).)
#     - Or can we predict multiple relation labels? (This is [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification).)
# 
# ### Joining the corpus and the KB
# 
# In order to leverage the distant supervision paradigm, we'll need to connect information in the corpus with information in the KB. There are two possibilities, depending on how we formulate our prediction problem:
# 
# - __Use the KB to generate labels for the corpus.__ If our problem is to classify a pair of entity *mentions* in a specific example in the corpus, then we can use the KB to provide labels for training examples. Labeling specific examples is how the fully supervised paradigm works, so it's the obvious way to think about leveraging distant supervision as well. Although it can be made to work, it's not actually the preferred approach.
# - __Use the corpus to generate features for entity pairs.__ If instead our problem is to classify a pair of *entities*, then we can use all the examples from the corpus where those two entities co-occur to generate a feature representation describing the entity pair. This is the approach taken by [Mintz et al. 2009](https://www.aclweb.org/anthology/P09-1113), and it's the approach we'll pursue here.
# 
# So we'll formulate our prediction problem such that the input is a pair of entities, and the goal is to predict what relation(s) the pair belongs to. The KB will provide the labels, and the corpus will provide the features.
# 
# We've created a `Dataset` class which combines a corpus and a KB, and provides a variety of convenience methods for the dataset.

# In[34]:


dataset = rel_ext.Dataset(corpus, kb)


# In[35]:


dataset


# Let's determine how many examples we have for each triple in the KB. We'll compute averages per relation.

# In[36]:


dataset.count_examples()


# From Tim: We have waaaaay to little examples at the moment....not surprising we only took a few articles

# For most relations, the total number of examples is fairly large, so we can be optimistic about learning what linguistic patterns express a given relation. However, for individual entity pairs, the number of examples is often quite low. Of course, more data would be better — much better! But more data could quickly become unwieldy to work with in a notebook like this.

# ### Negative instances
# 
# By joining the corpus to the KB, we can obtain abundant positive instances for each relation. But a classifier cannot be trained on positive instances alone. In order to apply the distant supervision paradigm, we will also need some negative instances — that is, entity pairs which do not belong to any known relation. If you like, you can think of these entity pairs as being assigned to a special relation called `NO_RELATION`. We can find plenty of such pairs by searching for examples in the corpus which contain two entities which do not belong to any relation in the KB.

# Tim: We will need to amend this as we have created the news articles to be including pairs of companis we know that are linked

# In[37]:


unrelated_pairs = dataset.find_unrelated_pairs()
print('Found {0:,} unrelated pairs, including:'.format(len(unrelated_pairs)))
for pair in list(unrelated_pairs)[:10]:
    print('   ', pair)


# That's a lot of negative instances! In fact, because these negative instances far outnumber our positive instances (that is, the triples in our KB), when we train models we'll wind up downsampling the negative instances substantially.
# 
# Remember, though, that some of these supposedly negative instances may be false negatives. Our KB is not complete. A pair of entities might be related in real life even if they don't appear together in the KB.

# ### Multi-label classification
# 
# A given pair of entities can belong to more than one relation. In fact, this is quite common in our KB.

# In[ ]:





# Tim: We should check that we dont have Company A customer of Company B AND Company A supplier of Company B...well maybe only for two large companies that might well be buying from / supplying different business lins

# In[38]:


dataset.count_relation_combinations()


# While a few of those combinations look like data errors, most look natural and intuitive. Multiple relations per entity pair is a commonplace phenomenon.
# 
# This observation strongly suggests formulating our prediction problem as [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification). We could instead treat it as [multi-class classification](https://en.wikipedia.org/wiki/Multiclass_classification) — and indeed, [Mintz et al. 2009](https://www.aclweb.org/anthology/P09-1113) did so — but if we do, we'll be faced with the problem of assigning a single relation label to entity pairs which actually belong to multiple relations. It's not obvious how best to do this (and Mintz et al. 2009 did not make their method clear).
# 
# There are a number of ways to approach multi-label classification, but the most obvious is the [binary relevance method](https://en.wikipedia.org/wiki/Multi-label_classification#Problem_transformation_methods), which just factors multi-label classification over _n_ labels into _n_ independent binary classification problems, one for each label. A disadvantage of this approach is that, by treating the binary classification problems as independent, it fails to exploit correlations between labels. But it has the great virtue of simplicity, and it will suffice for our purposes.
# 
# So our problem will be to take as input an entity pair and a candidate relation (label), and to return a binary prediction as to whether the entity pair belongs to the relation. Since a KB triple is precisely a relation and a pair of entities, we could say equivalently that our prediction problem amounts to binary classification of KB triples. Given a candidate KB triple, do we predict that it is valid?

# ### Building datasets
# 
# We're now in a position to write a function to build datasets suitable for training and evaluating predictive models. These datasets will have the following characteristics:
# 
# - Because we've formulated our problem as multi-label classification, and we'll be training separate models for each relation, we won't build a single dataset. Instead, we'll build a dataset for each relation, and our return value will be a map from relation names to datasets.
# - The dataset for each relation will consist of two parallel lists:
#   - A list of candidate `KBTriples` which combine the given relation with a pair of entities.
#   - A corresponding list of boolean labels indicating whether the given `KBTriple` belongs to the KB.
# - The dataset for each relation will include `KBTriples` derived from two sources:
#   - Positive instances will be drawn from the KB.
#   - Negative instances will be sampled from unrelated entity pairs, as described above.

# In[39]:


kbts_by_rel, labels_by_rel = dataset.build_dataset(
    include_positive=True, sampling_rate=1, seed=1)


# In[40]:


kbts_by_rel


# In[41]:


import numpy as np


# In[42]:


np.mean(labels_by_rel['supplier of'])
#np.mean(labels_by_rel['customer of'])


# In[43]:


print(kbts_by_rel['customer of'][0], labels_by_rel['customer of'][0])


# In[44]:


print(kbts_by_rel['supplier of'][637], labels_by_rel['supplier of'][637])


# \[ [top](#Relation-extraction-using-distant-supervision:-task-definition) \]

# ## Evaluation
# 
# Before we start building models, let's set up a test harness that allows us to measure a model's performance. This may seem backwards, but it's analogous to the software engineering paradigm of [test-driven development](https://en.wikipedia.org/wiki/Test-driven_development): first, define success; then, pursue it.

# ### Splitting the data
# 
# Whenever building a model from data, it's good practice to partition the data into a multiple _splits_ — minimally, a training split on which to train the model, and a test split on which to evaluate it. In fact, we'll go a bit further, and define three splits:
# 
# - __The `tiny` split (1%).__ It's often useful to carve out a tiny chunk of data to use in place of training or test data during development. Of course, any quantitative results obtained by evaluating on the `tiny` split are nearly meaningless, but because evaluations run extremely fast, using this split is a good way to flush out bugs during iterative cycles of code development.
# - __The `train` split (74%).__ We'll use the majority of our data for training models, both during development and at final evaluation. Experiments with the `train` split may take longer to run, but they'll have much greater statistical power.
# - __The `dev` split (25%).__ We'll use the `dev` split as test data for intermediate (formative) evaluations during development. During routine experiments, all evaluations should use the `dev` split.
# 
# You could also carve out a `test` split for a final (summative) evaluation at the conclusion of your work. The bake-off will have its own test set, so you needn't do this, but this is an important step for projects without pre-defined test splits.
# 
# Splitting our data assets is somewhat more complicated than in many other NLP problems, because we have both a corpus and KB. In order to minimize leakage of information from training data into test data, we'd like to split both the corpus and the KB. And in order to maximize the value of a finite quantity of data, we'd like to align the corpus splits and KB splits as closely as possible. In an ideal world, each split would have its own hermetically-sealed universe of entities, the corpus for that split would contain only examples mentioning those entities, and the KB for that split would contain only triples involving those entities. However, that ideal is not quite achievable in practice. In order to get as close as possible, we'll follow this plan:
# 
# - First, we'll split the set of entities which appear as the subject in some KB triple.
# - Then, we'll split the set of KB triples based on their subject entity.
# - Finally, we'll split the set of corpus examples.
#   - If the first entity in the example has already been assigned to a split, we'll assign the example to the same split.
#   - Alternatively, if the second entity has already been assigned to a split, we'll assign the example to the same split.
#   - Otherwise, we'll assign the example to a split randomly.
#   
# <!-- \[ TODO: figure out whether we actually need to split the _corpus_ -- any lift from testing on train corpus? \] -->
# 
# The `Dataset` method `build_splits` handles all of this:

# In[45]:


splits = dataset.build_splits(
    split_names=['tiny', 'train', 'dev'],
    split_fracs=[0.01, 0.74, 0.25],
    seed=1)

splits


# So now we can use `splits['train'].corpus` to refer to the training corpus, or `splits['dev'].kb` to refer to the dev KB.

# ### Choosing evaluation metrics
# 
# Because we've formulated our prediction problem as a family of binary classification problems, one for each relation (label), choosing evaluation metrics is pretty straightforward. The standard metrics for evaluating binary classification are [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall), which are more meaningful than simple accuracy, particularly in problems with a highly biased label distribution (like ours). We'll compute and report precision and recall separately for each relation (label). There are only two wrinkles:
# 
# 1. __How best to combine precision and recall into a single metric.__ Having two evaluation metrics is often inconvenient. If we're considering a change to our model which improves precision but degrades recall, should we take it? To drive an iterative development process, it's useful to have a single metric on which to hill-climb. For binary classification, the standard answer is the [F<sub>1</sub>-score](https://en.wikipedia.org/wiki/F1_score), which is the harmonic mean of precision and recall. However, the F<sub>1</sub>-score gives equal weight to precision and recall. For our purposes, precision is probably more important than recall. If we're extracting new relation triples from (massively abundant) text on the web in order to augment a knowledge base, it's probably more important that the triples we extract are correct (precision) than that we extract all the triples we could (recall). Accordingly, instead of the F<sub>1</sub>-score, we'll use the F<sub>0.5</sub>-score, which gives precision twice as much weight as recall.
# 
# 1. __How to aggregate metrics across relations (labels).__ Reporting metrics separately for each relation is great, but in order to drive iterative development, we'd also like to have summary metrics which aggregate across all relations. There are two possible ways to do it: _micro-averaging_ will give equal weight to all problem instances, and thus give greater weight to relations with more instances, while _macro-averaging_ will give equal weight to all relations, and thus give lesser weight to problem instances in relations with more instances. Because the number of problem instances per relation is, to some degree, an accident of our data collection methodology, we'll choose macro-averaging.
# 
# Thus, while every evaluation will report lots of metrics, when we need a single metric on which to hill-climb, it will be the macro-averaged F<sub>0.5</sub>-score.

# ### Running evaluations
# 
# It's time to write some code to run evaluations and report results. This is now straightforward. The `rel_ext.evaluate()` function takes as inputs:
# 
# - `splits`: a `dict` mapping split names to `Dataset` instances
# - `classifier`, which is just a function that takes a list of `KBTriples` and returns a list of boolean predictions
# - `test_split`, the split on which to evaluate the classifier, `dev` by default
# - `verbose`, a boolean indicating whether to print output

# ### Evaluating a random-guessing strategy
# 
# In order to validate our evaluation framework, and to set a floor under expected results for future evaluations, let's implement and evaluate a random-guessing strategy. The random guesser is a classifier which completely ignores its input, and simply flips a coin.

# In[46]:


def lift(f):
    return lambda xs: [f(x) for x in xs]

def make_random_classifier(p=0.50):
    def random_classify(kb_triple):
        return random.random() < p
    return lift(random_classify)


# In[47]:


rel_ext.evaluate(splits, make_random_classifier())


# The results are not too surprising. Recall is generally around 0.50, which makes sense: on any given example with label `True`, we are 50% likely to guess the right label. But precision is very poor, because most labels are not `True`, and because our classifier is completely ignorant of the features of specific problem instances. Accordingly, the F<sub>0.5</sub>-score is also very poor — first because even the equally-weighted F<sub>1</sub>-score is always closer to the lesser of precision and recall, and second because the F<sub>0.5</sub>-score weights precision twice as much as recall.
# 
# Actually, the most remarkable result in this table is the comparatively good performance for the `contains` relation! What does this result tell us about the data?
# 
# \[ [top](#Relation-extraction-using-distant-supervision:-task-definition) \]

# ## A simple baseline model
# 
# It shouldn't be too hard to do better than random guessing. But for now, let's aim low — let's use the data we have in the easiest and most obvious way, and see how far that gets us.
# 
# We start from the intuition that the words between two entity mentions frequently tell us how they're related. For example, in the phrase "SpaceX was founded by Elon Musk", the words "was founded by" indicate that the `founders` relation holds between the first entity mentioned and the second. Likewise, in the phrase "Elon Musk established SpaceX", the word "established" indicates the `founders` relation holds between the second entity mentioned and the first.
# 
# So let's write some code to find the most common phrases that appear between the two entity mentions for each relation. As the examples illustrate, we need to make sure to consider both directions: that is, where the subject of the relation appears as the first mention, and where it appears as the second.

# In[48]:


def find_common_middles(split, top_k=3, show_output=False):
    corpus = split.corpus
    kb = split.kb
    mids_by_rel = {
        'fwd': defaultdict(lambda: defaultdict(int)),
        'rev': defaultdict(lambda: defaultdict(int))}
    for rel in kb.all_relations:
        for kbt in kb.get_triples_for_relation(rel):
            for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
                mids_by_rel['fwd'][rel][ex.middle] += 1
            for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
                mids_by_rel['rev'][rel][ex.middle] += 1
    def most_frequent(mid_counter):
        return sorted([(cnt, mid) for mid, cnt in mid_counter.items()], reverse=True)[:top_k]
    for rel in kb.all_relations:
        for dir in ['fwd', 'rev']:
            top = most_frequent(mids_by_rel[dir][rel])
            if show_output:
                for cnt, mid in top:
                    print('{:20s} {:5s} {:10d} {:s}'.format(rel, dir, cnt, mid))
            mids_by_rel[dir][rel] = set([mid for cnt, mid in top])
    return mids_by_rel

_ = find_common_middles(splits['train'], show_output=True)


# A few observations here:
# 
# - Some of the most frequent middles are natural and intuitive. For example, ", son of" indicates a forward `parents` relation, while "and his son" indicates a reverse `parents` relation.
# - Punctuation and stop words such as "and" and "of" are extremely common. Unlike some other NLP applications, it's probably a bad idea to throw these away — they carry lots of useful information.
# - However, punctuation and stop words tend to be highly ambiguous. For example, a bare comma is a likely middle for almost every relation in at least one direction.
# - A few of the results reflect quirks of the dataset. For example, the appearance of the phrase "in 1994 , he became a central figure in the" as a common middle for the `genre` relation reflects both the relative scarcity of examples for that relation, and an unfortunate tendency of the Wikilinks dataset to include duplicate or near-duplicate source documents. (That middle connects the entities [Ready to Die](https://en.wikipedia.org/wiki/Ready_to_Die) — the first studio album by the Notorious B.I.G. — and [East Coast hip hop](https://en.wikipedia.org/wiki/East_Coast_hip_hop).)
# 
# Now it's straightforward task to build and evaluate a classifier which predicts `True` for a candidate `KBTriple` just in case its entities appear in the corpus connected by one of the phrases we just discovered.

# In[49]:


def train_top_k_middles_classifier(top_k=3):
    split = splits['train']
    corpus = split.corpus
    top_k_mids_by_rel = find_common_middles(split=split, top_k=top_k)
    def classify(kb_triple):
        fwd_mids = top_k_mids_by_rel['fwd'][kb_triple.rel]
        rev_mids = top_k_mids_by_rel['rev'][kb_triple.rel]
        for ex in corpus.get_examples_for_entities(kb_triple.sbj, kb_triple.obj):
            if ex.middle in fwd_mids:
                return True
        for ex in corpus.get_examples_for_entities(kb_triple.obj, kb_triple.sbj):
            if ex.middle in rev_mids:
                return True
        return False
    return lift(classify)


# In[50]:


rel_ext.evaluate(splits, train_top_k_middles_classifier())


# Not surprisingly, the performance of even this extremely simplistic model is noticeably better than random guessing. Of course, recall is much worse across the board, but precision and F<sub>0.5</sub>-score are sometimes much better. We observe big gains especially on `adjoins`, `author`, `has_sibling`, and `has_spouse`. Then again, at least one relation actually got worse. (Can you offer any explanation for that?)
# 
# Admittedly, performance is still not great in absolute terms. However, we should have modest expectations for performance on this task — we are unlikely ever to get anywhere near perfect precision with perfect recall. Why?
# 
# - High precision will be hard to achieve because the KB is incomplete: some entity pairs that are related in the world — and in the corpus — may simply be missing from the KB.
# - High recall will be hard to achieve because the corpus is finite: some entity pairs that are related in the KB may not have any examples in the corpus.
# 
# Because of these unavoidable obstacles, what matters is not so much absolute performance, but relative performance of different approaches.
# 
# __Question:__ What's the optimal value for `top_k`, the number of most frequent middles to consider? What choice maximizes our chosen figure of merit, the macro-averaged F<sub>0.5</sub>-score?
# 
# \[ [top](#Relation-extraction-using-distant-supervision:-task-definition) \]
