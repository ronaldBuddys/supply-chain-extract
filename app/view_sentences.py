
import os
import re
import sys
import json
import time
import pandas as pd
import numpy as np

import requests
import dash
from dash import dash_table
from dash import dcc
from dash import html

from pymongo import UpdateOne

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


# TODO: remove this - use pip install -e . to install the package
try:
    # python package (supply_chain_extract) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


from supply_chain_extract.utils import get_database,  get_knowledge_base_from_value_chain_data
from supply_chain_extract import get_configs_path, get_data_path

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 70)


# --
# try getting user named
# --

try:
    user_name = os.getlogin()
except Exception as e:
    print(e)
    print("will use user_name = 'unknown'")
    user_name = "unknown"


# ---
# connect to database
# ---


# get credentials
with open(get_configs_path("mongo.json"), "r+") as f:
    mdb_cred = json.load(f)

# get mongodb client - for connections
client = get_database(**mdb_cred)

art_db = client["news_articles"]


# ---
# read in value chain data / knowledge base
# ---

# read in locally stored valued chains
vc = pd.read_csv(get_data_path("VCHAINS.csv"))
# get knowledge base
kb = get_knowledge_base_from_value_chain_data(vc)

print("kb built")
# ---
# read in full_sentences store locally
# ---

# print("reading in full sentences")
#
# # text_file = get_data_path("full_sentences.json")
# text_file = get_data_path("text_with_weak_labels.json")
#
# assert os.path.exists(text_file), \
#     f"looks like: {text_file}, copy from google drive data/full_sentences.json"
#
# with open(text_file, "r") as f:
#     full_sents = json.load(f)
#
# # store sentence data in dataframe
# df = pd.DataFrame(full_sents)

# ---
# read in split data (stored locally)
# ---

# TODO: tidy the following up
# TODO: consider just sing this - would have to compare with above a bit
split_data = pd.read_csv(get_data_path("df_full.tsv"), sep="\t", na_filter = False)
test_map = {"train": 0, "val": 1, "test": 2}

# HARDCODED: taking only test data
df = split_data.loc[split_data['split'] == test_map["test"]]

# shuffle values - so can use Next and Prev
df = df.sample(frac=1)

# HARDCODED: for now just get the test data
# ids = split_data.loc[split_data['split'] == test_map["test"], "id"].values

# take a subset of data
# df = df.loc[df['id'].isin(ids)]

# read the current gold_labels from the database - for the given splits
filter = {"label_id": {"$in": [i for i in df["id"]]}}
t0 = time.time()
gl_tmp = list(art_db['gold_labels'].find(filter=filter))
gl_tmp = pd.DataFrame(gl_tmp)
gl_tmp.drop("_id", axis=1, inplace=True)
gl_tmp = gl_tmp.drop_duplicates()
t1 = time.time()
print(f"time to get all gold labels : {t1-t0:.2f} seconds")

pre_merge = len(df)

# merge on current gold label with sentence data
df = df.merge(gl_tmp, left_on="id", right_on="label_id", how="left")
df.drop("label_id", axis=1, inplace=True)

assert len(df) == pre_merge, "there was an unexpected dataframe file change after merging labels"

# if there are any missing gold_labels (for some reason label_id is not db)
# - set to None
df.loc[pd.isnull(df["gold_label"]), "gold_label"] = None

# get the current gold label count
gl_count = (~pd.isnull(df["gold_label"])).sum()

# ---
# create an Dash app - referring to some css template
# ---

# use a 'default' css file - could save locally if running offline
app = dash.Dash(__name__,
                external_stylesheets=[
                    'https://codepen.io/chriddyp/pen/bWLwgP.css'
                ])
app.title = "View Sentences"

# ---
# parameters
# ---

text_color = "#1314fc"

# ---
# helper functions - review: these aren't being used?
# ---

print("helper functions")

def replace_with_html(text, split_word, replace_with):
    # split the text on the given 'split_word'
    # TODO: double check this
    text_list = text.split(split_word)
    out = [text_list[0]]
    # start index at one to allow for no matching - i.e. inserting replace_with
    for i in range(1, len(text_list)):
        out.extend([replace_with, text_list[i]])

    # HACK: to avoid having '' at start or end of list
    if out[0] == '':
        out = out[1:]
    if out[-1] == '':
        out = out[:-1]

    return out


def text_to_dash_html(text, color_text=None):

    # replace '\n with line break
    out = replace_with_html(text, split_word="\n", replace_with=html.Br())
    # out = replace_with_html(text, split_word="\n", replace_with="")

    # allow for highlighting of certain words
    if color_text is None:
        color_text = []

    for ct in color_text:
        assert isinstance(ct, dict), f"color_text list element must be a dict with 'word' and optionally style"
        assert "word" in ct, f"'word' expected to be in dict in color_text"
        text_list = out
        out = []
        replace_with = html.Mark(f" {ct['word']} ", style=ct.get('style', None))
        for i in text_list:
            # if the element is a string then split it up
            if isinstance(i, str):
                tmp = replace_with_html(i, split_word=ct['word'], replace_with=replace_with)
                out.extend(tmp)
            # otherwise, just keep the element
            else:
                out.append(i)

    return out


def list_to_text(l):
    out = ", ".join(l)
    return out


def increment_idx(idx_range, cur_idx, action):
    """
    given an allowed (integer) index range
    and a current index value, increment to the next, prev, or random value
    in the index range
    """
    # require the current index value in range
    if cur_idx not in idx_range:
        print(f"cur_idx: {cur_idx} is not in provided idx_range, will return first value in idx_range")
        return idx_range[0]

    if action == "None":
        return cur_idx
    elif action == "Next":
        next_idx = np.argmax(cur_idx == idx_range) + 1
        if next_idx >= len(idx_range):
            next_idx = 0
        return idx_range[next_idx]
    elif action == "Prev":
        prev_idx = np.argmax(cur_idx == idx_range) - 1
        if prev_idx < 0:
            prev_idx = len(idx_range) - 1
        return idx_range[prev_idx]
    elif action == "Random":
        return np.random.choice(idx_range, 1)[0]
    else:
        print(f"action: {action} was not understood returning cur_idx: {cur_idx}")
        return cur_idx

# ---
# app layout
# ---

print("options for layout")

# TODO: consider using long names here?
# TODO: in entities put (count)
entity1_options = [{'label': i, 'value': i} for i in np.sort(df['entity1_full'].unique())]
entity2_options = [{'label': i, 'value': i} for i in np.sort(df['entity2_full'].unique())]
relation_options = [{'label': i, 'value': i} for i in np.sort(df['relation'].unique())]
wl_options = [{'label': i, 'value': i} for i in np.sort(df['weak_label'].unique())]

num_sent_options = [{'label': i, 'value': i} for i in np.sort(df['num_sentence'].unique())]
epair_options = [{'label': i, 'value': i} for i in np.sort(df['epair_count'].unique())]
ulabel_options = [{'label': i, 'value': i} for i in ["True", "False", "Only Labelled"]]


# previous column selection
select_col = ["text", "num_sentence", "entity1", "entity2", "relation", "weak_label"]
table_columns = [{"id": c, "name": c} for c in select_col]

# information on text columns
info_col = ["entity1", "entity2", "entity1_full", "entity2_full",
            "relation", "weak_label", "prob_label", "Confidence Score (%)",
            "num_sentence", "sentence_range", "num_chars", "epair_count"]
if "companies_in_text" in df.columns:
    info_col.append("companies_in_text")


# format some of the float columns
# - would prefer to do this in display..
format_float_cols = ['Confidence Score (%)', "prob_label"]
for ffc in format_float_cols:
    df[ffc] = df[ffc].map("{:,.3f}".format)

# actions after label options
act_after_label = ["None", "Next", "Prev", "Random"]
act_after_label_opts = [{'label': i, "value": i} for i in act_after_label]

# gold label options
gl_opt_list = ["Supplier", "not specified", "unsure", "partnership", "owner", "competitor", "reverse"]
gl_opt = [{'label': i, "value": i} for i in gl_opt_list]
# add None - not needed - could use select unlabelled
# gl_opt.append({"label": "None", "values": None})

# --
# reminder text - with colors added
# --
question_text = 'Does [entity2] supply {entity1} ?'
e1_color = [{'word': "{entity1}", "style": {"backgroundColor": "#6190ff", 'display': 'inline-block'} }]
e2_color = [{'word': "[entity2]", "style": {"backgroundColor": "yellow", 'display': 'inline-block'} }]

question_text = text_to_dash_html(question_text,
                                  color_text = e1_color+e2_color)
# formated_text = text_to_dash_html(cur_sent,
#                                   color_text)


# ---
# make some columns shorter ?
# ---

# make some sentences shorter? - replace space with carraige return
# rename_map = {ic: re.sub("_| ", "\n", ic) for ic in info_col}
# info_col = [v for k, v in rename_map.items()]
# df.rename(columns=rename_map, inplace=True)
info_table_columns = [{"id": c, "name": re.sub("_| ", "\n", c)} for c in info_col]

print("app layout")

app.layout = html.Div([

    html.Div([
        html.H3(children='Sentence Review:',
                style={'color': text_color, 'textAlign': 'left', "font-weight": "bold", "display": 'inline-block'},
                className="threeColumns"),
        html.H3(children=question_text,
                style={"margin-left": "15px", "font-weight": "bold", "display": 'inline-block'},
                className="fiveColumns"),
    ], className='row'),

    html.Div([
        html.H6(children='Filters:',
                style={'color': text_color, 'textAlign': 'left', "font-weight": "bold", 'margin-top': '-25px'}),
    ]),

    # Main Selection
    html.Div([
        # html.H6(children='Sentence Filternasdf',
        #         style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),

        html.Div([
            html.Label("Entity1"),
            dcc.Dropdown(
                id='entity1_select',
                options=entity1_options,
                placeholder="select an entity1",
                style={'fontSize': '12px'}),
        ], className="three columns"),
        # entity2 selection
        html.Div([
            html.Label("Entity2"),
            dcc.Dropdown(
                id='entity2_select',
                options=entity2_options,
                placeholder="select an entity2",
                style={'fontSize': '12px'}),
        ], className="three columns"),
        # relation selection
        html.Div([
            html.Label("Relation"),
            dcc.Dropdown(
                id='relation_select',
                options=relation_options,
                placeholder="select an relation",
                style={'fontSize': '12px'}),
        ], className="three columns"),
        # weak label
        html.Div([
            html.Label("Weak Label"),
            dcc.Dropdown(
                id='weak_label_select',
                options=wl_options,
                placeholder="select a weak label",
                style={'fontSize': '12px'}),
        ], className="three columns"),


    ], className='row', style={'margin-top': '-10px'}),
    # -----
    # filtering Selection
    # -----
    # html.H6(children='Filtering',
    #         style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),

    html.Div([
        # number of sentences
        html.Div([
            html.Label("# Sentences"),
            dcc.Dropdown(
                id='num_sent_select',
                options=num_sent_options,
                placeholder="select max. # ",
                style={'fontSize': '12px'}),
        ], className="two columns"),
        # count of text per entity pair
        html.Div([
            html.Label("# / entity pair"),
            dcc.Dropdown(
                id='epair_select',
                options=epair_options,
                placeholder="select max. # ",
                style={'fontSize': '12px'}),
        ], className="two columns"),
        html.Div([
            html.Label("show unlabeled only?"),
            dcc.Dropdown(
                id='ulabel_select',
                options=ulabel_options,
                # placeholder="select max. # ",
                value="False",
                style={'fontSize': '12px'}),
        ], className="three columns"),
        html.Div([
            html.Label("gold label values"),
            dcc.Dropdown(
                id='glabel_select',
                options=gl_opt,
                placeholder="select gold label",
                # value="False",
                style={'fontSize': '12px'}),
        ], className="three columns"),
    ], className='row'),

    # information about the sentences
    html.H6(children='Sentence Info:',
            style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),

    html.Div([
        dash_table.DataTable(
            id='table_sent_info',
            data=df.iloc[[0]][info_col].to_dict('records'),
            columns=info_table_columns,
            # editable=True,
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'text'},
                    'textAlign': 'left'
                }
            ],
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold',
                'whiteSpace': 'pre-line'
            },
            # page_current=0,
            # page_size=10,
            # page_action="custom",
        ),
    ], className='row'),

    # --
    # global label / current position / after action
    # --
    html.Div([
        html.Div([
            html.Label(f"Gold Label Count:", style={'color': "yellow", "font-weight": "bold"}),
            html.P(str(gl_count), id="gold_label_count"),
        ], className="twoColumns", style={"display": 'inline-block'}),
        html.Div([
            html.Label("pos. / # filtered"),
            html.Div(children="0/0", id="relative_index"),
            html.Div(children="0", id="current_index", style={'display': 'none'}),
        ], className="twoColumns", style={"display": 'inline-block', "margin-left": "15px"}),
        html.Div([
            html.Label("Action After Label", style={"font-weight": "bold"}),
            dcc.Dropdown(
                options=act_after_label_opts,
                id="act_after_label",
                value="Next",
                style={"display": 'inline-block'}
            )
        ], className="twoColumns", style={"display": 'inline-block', "margin-left": "15px"}),

    ], className="row"),

    # Navigation buttons
    html.H6(children='Sentence Navigation Buttons:',
            style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),

    html.Div([

        html.Div([
            html.Button("Prev", id="prev_btn"),
            html.Button("Next", id="next_btn", style={"margin-left": "15px"}),
            html.Button("Random", id="rnd_btn", style={"margin-left": "15px"}),
        ], style={'width': '100%', 'display': 'flex', 'align-items': 'center'}),
    ], className="row"),

    # Labelling buttons
    html.Div([
        # -
        html.H6(children='Choose Label',
                style={'color': text_color, "font-weight": "bold", "display": 'inline-block'}),
        html.H6(children=question_text,
                style={"margin-left": "15px", "font-weight": "bold",  "display": 'inline-block'}),
    ], className="row"),

    html.Div([

        html.Div([
            html.Button("Supplier", id="supply_btn"),
            html.Button("Not Specified", id="norel_btn", style={"margin-left": "15px"}),
            html.Button("Unsure", id="unsure_btn", style={"margin-left": "15px"}),
            html.Button("Partnership", id="prtnr_btn", style={"margin-left": "15px"}),
            html.Button("Competitor", id="compet_btn", style={"margin-left": "15px"}),
            html.Button("Owner", id="owner_btn", style={"margin-left": "15px"}),
            html.Button("Reverse", id="rvrsd_btn", style={"margin-left": "15px"})
        ], style={'width': '100%', 'display': 'flex', 'align-items': 'left'})
    ], className="row"),


    # ---
    # sentence (text)
    # ---
    html.Div([
        html.H6(children='Sentence Text',
                style={'color': text_color,
                       'textAlign': 'left',
                       "font-weight": "bold",
                       "display": 'inline-block'}),
        html.Label("Current Gold Label:", style={'color': 'yellow',
                                                 'textAlign': 'left',
                                                 "font-weight": "bold",
                                                 "margin-left": "15px",
                                                 "display": 'inline-block'}),
        html.P(id="gold_label_value", style={"display": 'inline-block',
                                             "margin-left": "15px"}),
    ], className="row"),


    html.Div([

        html.Div([
            html.P(children="text goes here", id="current_sent"),
            # html.Div(children="0", id="current_index"),# style={'display': 'none'}),
        ], className="eightColumns"),


    ], className="row"),

], style={'backgroundColor': '#DCDCDC'})

# ----
# callbacks
# ----

print("callbacks")

@app.callback([Output("table_sent_info", "data"),
               # Output("table-dropdown", "page_count"),
               Output("current_sent", "children"),
               Output("current_index", "children"),
               Output("gold_label_value", "children"),
               Output("gold_label_count", "children"),
               Output("relative_index", "children")],
              [Input('entity1_select', 'value'),
               Input('entity2_select', 'value'),
               Input('relation_select', 'value'),
               Input('weak_label_select', 'value'),
               Input('num_sent_select', 'value'),
               Input('epair_select', 'value'),
               Input('ulabel_select', 'value'),
               Input('glabel_select', 'value'),
               # labelling buttons from here
               Input('supply_btn', 'n_clicks'),
               Input('norel_btn', 'n_clicks'),
               Input('unsure_btn', 'n_clicks'),
               Input('prtnr_btn', 'n_clicks'),
               Input('compet_btn', 'n_clicks'),
               Input('owner_btn', 'n_clicks'),
               Input('rvrsd_btn', 'n_clicks'),
               # navigation buttons from here
               Input('next_btn', 'n_clicks'),
               Input('prev_btn', 'n_clicks'),
               Input('rnd_btn', 'n_clicks'),
               # Input("table-dropdown", "page_current")
              ],
              [
                State("current_index", "children"),
                State("gold_label_count", "children"),
                State("act_after_label", "value")
                   # State("table-dropdown", "page_size"),
             ])
def available_titles(e1, e2, rel, wl, ns, ep, ul, gl_sel,
                     s_btn, nr_btn, us_btn, pt_btn, cmp_btn, own_btn, rvs_btn,
                     nx_btn, pv_btn, rn_btn,
                     # pc, ps,
                     cr_idx, glc, aal):

    print("-"*50)
    # ---
    # determine what triggered callback (?)
    # ---

    # determine which Input triggered callback
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    print(f"button_id: {button_id}")

    selection_buttons = [
        'entity1_select',
        'entity2_select',
        'relation_select',
        'weak_label_select',
        'num_sent_select',
        'epair_select',
        'ulabel_select',
        'glabel_select'
    ]

    # TODO: allow for multiple values
    select_bool = np.ones(len(df), dtype=bool)
    if e1 is not None:
        if e1 != "":
            print(f"selecting matching entity1: {e1}")
            b = (df['entity1_full'].values == e1)
            print(b.sum())
            select_bool = select_bool & b
    if e2 is not None:
        print(f"selecting matching entity2: {e2}")
        b = (df['entity2_full'].values == e2)
        print(b.sum())
        select_bool = select_bool & b
    if rel is not None:
        print(f"selecting matching relation: {rel}")
        b = (df['relation'].values == rel)
        print(b.sum())
        select_bool = select_bool & b

    if wl is not None:
        print(f"selecting matching weak label: {wl}")
        b = (df['weak_label'].values == wl)
        print(b.sum())
        select_bool = select_bool & b

    if ns is not None:
        print(f"max number of sentences is: {ns}")
        b = (df['num_sentence'].values <= ns)
        print(b.sum())
        select_bool = select_bool & b

    if ep is not None:
        print(f"max number of sentences per entity pair: {ep}")
        b = (df['epair_count'].values <= ep)
        print(b.sum())
        select_bool = select_bool & b

    if gl_sel is not None:
        print(f"selecting gold labels: {gl_sel}")
        b = (df['gold_label'] == gl_sel)
        print(b.sum())
        select_bool = select_bool & b

    # HARDCODED: for now only show entries with no gold label

    print(f"return: {select_bool.sum()} values")
    print(f"current index: {cr_idx}")
    # print(type(cr_idx))

    # --
    # if a selection triggered callback
    # --

    # select data
    tmp = df.loc[select_bool]

    # select only unlabelled data?
    if ul is not None:
        if ul == "True":
            print(f"only will show unlabelled text")
            possible_idx = pd.isnull(tmp["gold_label"]).values
        elif ul == "False":
            possible_idx = np.ones(len(tmp), dtype=bool)
        elif ul == "Only Labelled":
            print(f"only will show unlabelled text")
            possible_idx = ~pd.isnull(tmp["gold_label"]).values
        else:
            print("unlabelled text option not understood")
            print(ul)
            raise PreventUpdate
    else:
        possible_idx = np.ones(len(tmp), dtype=bool)

    # select possible (integer) index values
    idx_range = np.arange(len(tmp))
    idx_range = idx_range[possible_idx]

    if len(idx_range) == 0:
        print("the selection to strict! can't do anything, preventing update")
        raise PreventUpdate

    # TODO: be more sensible with cr_idx being string
    #  - convert immediately when passed into function and right before passed out only

    # HACK: to make sure current index is in range
    try:
        _ = tmp.iloc[int(cr_idx)]["id"]
    except Exception as e:
        print(e)
        print("taking first index in range")
        # cr_idx = str(0)
        cr_idx = str(idx_range[0])

    # if selection generated the callback
    if button_id in selection_buttons:
        # set current index values to zero
        # cr_idx = str(0)
        cr_idx = str(idx_range[0])

    # otherwise, a iteration button has been clicked
    else:
        if button_id in ["supply_btn", "norel_btn", "unsure_btn", "prtnr_btn",
                         "compet_btn", "rvrsd_btn", "owner_btn"]:
            # TODO: here should confirm status - and write to data
            cur_id = tmp.iloc[int(cr_idx)]["id"]
            print(f"current sentence id: {cur_id}")

            if button_id == "supply_btn":
                glabel = "Supplier"
            elif button_id == "norel_btn":
                glabel = "not specified"
            elif button_id == "unsure_btn":
                glabel = "unsure"
            elif button_id == "prtnr_btn":
                glabel = "partnership"
            elif button_id == "owner_btn":
                glabel = "owner"
            elif button_id == "compet_btn":
                glabel = "competitor"
            elif button_id == "rvrsd_btn":
                glabel = "reverse"
            else:
                print(f"button: {button_id}\nnot understood, preventing update")
                raise PreventUpdate

            print(f"setting gold label as {glabel}")

            art_db['gold_labels'].update_one(filter={"label_id": cur_id},
                                             update={"$set": {"gold_label": glabel,
                                                              "labeller": user_name}},
                                             upsert=True)

            print("setting label stored locally")

            # update value in dataframe
            df.loc[df['id'] == cur_id, "gold_label"] = glabel

            # increment the gold label count
            glc = str(int(glc) + 1)
            print(f"new gold label count: {glc}")

            # get the next (integer) index value
            cr_idx = increment_idx(idx_range, int(cr_idx), action = aal)
            cr_idx = str(cr_idx)

        elif button_id in ["prev_btn", "next_btn", "rnd_btn"]:
            print("changing sentence")
            if button_id == "prev_btn":
                cr_idx = increment_idx(idx_range, int(cr_idx), action = "Prev")
            elif button_id == "next_btn":
                cr_idx = increment_idx(idx_range, int(cr_idx), action = "Next")
            else:
                cr_idx = increment_idx(idx_range, int(cr_idx), action = "Random")
            # convert to string
            cr_idx = str(cr_idx)
        elif button_id in "No clicks yet":
            pass
        else:
            print(f"button_id: {button_id}\n not handled, doing nothing ")
            raise PreventUpdate


    # select sentence
    # HACK: to avoid cr_idx being out of range
    # - it's possible the bool select could be too strict
    # - which case a PreventUpdate should be used
    try:
        cur_sent = tmp.iloc[int(cr_idx)]["text"]
    except Exception as e:
        print(e)
        cr_idx = "0"
        cur_sent = tmp.iloc[int(cr_idx)]["text"]


    # current sentence id
    # cur_id = tmp.iloc[int(cr_idx)]["id"]

    # get the gold label from database
    cur_id = tmp.iloc[int(cr_idx)]["id"]
    _ = art_db["gold_labels"].find_one(filter={"label_id": cur_id})

    print(f"cur_id: {cur_id}")
    # TODO: fix this / tidy up: trying to handle missing
    _ = {} if _ is None else _
    glabel = _.get("gold_label", "no label provided yet")
    glabel = "no label provided yet" if glabel is None else glabel

    print(f'glabel: {glabel}')
    # print(f"current page: {pc}")

    # select_range = int(cr_idx) + np.arange(pc*ps, (pc + 1)*ps)
    select_range = int(cr_idx)

    # ---
    # format text
    # ---

    e1 = tmp.iloc[int(cr_idx)]["entity1"]
    e2 = tmp.iloc[int(cr_idx)]["entity2"]

    # put brackets around entity
    e1_color = [{'word': "{%s}" % e1, "style": {"backgroundColor": "#6190ff", 'display': 'inline-block'} }]
    e2_color = [{'word': "[%s]" % e2, "style": {"backgroundColor": "yellow", 'display': 'inline-block'} }]

    # add brackets to current text
    #
    cur_sent = re.sub(f"{e1}", "{%s}" % e1, cur_sent)
    cur_sent = re.sub(f"{e2}", "[%s]" % e2, cur_sent)

    formated_text = text_to_dash_html(cur_sent,
                                      color_text=e1_color+e2_color)

    rl_idx = np.argmax(int(cr_idx) == idx_range)

    # the current position of those filtered
    rl_pos = f"{rl_idx} / {len(idx_range)}"

    # page_count, \
    return tmp.iloc[[select_range]][info_col].to_dict('records'), \
           formated_text,  \
           str(cr_idx), \
           glabel,\
           glc, \
           rl_pos


if __name__ == "__main__":

    app.run_server(port=8051,
                   debug=True,
                   use_reloader=True,
                   host='0.0.0.0',
                   passthrough_errors=True)  # Turn off reloader if inside Jupyter

    # close mongo connection
    client.close()
