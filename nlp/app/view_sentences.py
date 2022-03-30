
import os
import re
import sys
import json
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


try:
    # python package (nlp) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


from nlp.utils import get_database,  get_knowledge_base_from_value_chain_data
from nlp import get_configs_path, get_data_path

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 70)

# ---
# connect to database
# ---


# get credentials
with open(get_configs_path("mongo0.json"), "r+") as f:
    mdb_cred = json.load(f)

# # get mongodb client - for connections
client = get_database(username=mdb_cred["username"],
                      password=mdb_cred["password"],
                      clustername=mdb_cred["cluster_name"])

art_db = client["news_articles"]

# ---
# read in gold labels
# ---

# get a count of number of gold labels
pipeline = [
    {
        "$match":
            {
                "gold_label": {"$ne": None}
            }
    },
    {
        "$count": "has gold label"
    }
]

gl_count = list(art_db["gold_labels"].aggregate(pipeline))
gl_count = gl_count[0]["has gold label"]

# ---
# read in value chain data / knowledge base
# ---

# read in locally stored valued chains
vc = pd.read_csv(get_data_path("VCHAINS.csv"))
# get knowledge base
kb = get_knowledge_base_from_value_chain_data(vc)

# ---
# read in full_sentences store locally
# ---

print("reading in full sentences")

# text_file = get_data_path("full_sentences.json")
text_file = get_data_path("text_with_weak_labels.json")

assert os.path.exists(text_file), \
    f"looks like: {text_file}, copy from google drive data/full_sentences.json"

with open(text_file, "r") as f:
    full_sents = json.load(f)

# store sentence data in dataframe
df = pd.DataFrame(full_sents)


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

# ---
# app layout
# ---

# TODO: consider using long names here?
# TODO: in entities put (count)
entity1_options = [{'label': i, 'value': i} for i in np.sort(df['entity1_full'].unique())]
entity2_options = [{'label': i, 'value': i} for i in np.sort(df['entity2_full'].unique())]
relation_options = [{'label': i, 'value': i} for i in np.sort(df['relation'].unique())]
wl_options = [{'label': i, 'value': i} for i in np.sort(df['weak_label'].unique())]

num_sent_options = [{'label': i, 'value': i} for i in np.sort(df['num_sentence'].unique())]
epair_options = [{'label': i, 'value': i} for i in np.sort(df['epair_count'].unique())]


# previous column selection
select_col = ["text", "num_sentence", "entity1", "entity2", "relation", "weak_label"]
table_columns = [{"id": c, "name": c} for c in select_col]

info_col = ["entity1", "entity2", "entity1_full", "entity2_full",
            "relation", "weak_label", "prob_label", "Confidence Score (%)",
            "num_sentence", "sentence_range", "num_chars", "epair_count"]
info_table_columns = [{"id": c, "name": c} for c in info_col]


# format some of the float columns
# - would prefer to do this in display..
format_float_cols = ['Confidence Score (%)', "prob_label"]
for ffc in format_float_cols:
    df[ffc] = df[ffc].map("{:,.3f}".format)

# actions after label options
act_after_label = ["None", "Next", "Prev", "Random"]
act_after_label_opts = [{'label': i, "value": i} for i in act_after_label]


app.layout = html.Div([

    # Header
    # html.Div([
    #     # app 'name'
    #     html.H1(children='Text Labeller',
    #             style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"},
    #             className='twelve columns'),
    # ], className='row'),
    #

    html.H3(children='Main Selection',
            style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),
    # Main Selection
    html.Div([

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


    ], className='row'),
    # filtering Selection
    html.H6(children='Filtering',
            style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),

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
    ], className='row'),

    # information about the sentences
    html.H6(children='Sentence Info',
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
            # page_current=0,
            # page_size=10,
            # page_action="custom",
        ),
    ], className='row'),

    # --
    # global label / count / after action
    # --
    html.Div([
        html.Div([
            html.Label(f"Gold Label Count:", style={'color': "yellow", "font-weight": "bold"}),
            html.P(str(gl_count), id="gold_label_count"),
        ], className="twoColumns", style={"display": 'inline-block'}),
        html.Div([
            html.Label("Current Gold Label:", style={'color': 'yellow',
                                                     'textAlign': 'left',
                                                     "font-weight": "bold"}),
            html.P(id="gold_label_value"),
        ], className="twoColumns", style={"margin-left": "15px", "display": 'inline-block'}),
        html.Div([
            html.Label("Action After Label", style={"font-weight": "bold"}),
            dcc.Dropdown(
                options=act_after_label_opts,
                id="act_after_label",
                value="None",
                style={"display": 'inline-block'}
            )
        ], className="twoColumns", style={"display": 'inline-block', "margin-left": "15px"})
    ], className="row"),

    # Buttons for labellings (and navigating)
    html.H6(children='Labelling and Nav. Buttons',
            style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),

    html.Div([

        html.Div([
            html.Button("Prev", id="prev_btn"),
            html.Button("Next", id="next_btn", style={"margin-left": "15px"}),
            html.Button("Random", id="rnd_btn", style={"margin-left": "15px"}),
        ], style={'justify-content': 'center',
                  'width': '100%', 'display': 'flex', 'align-items': 'center'}),
        html.Br(),
        html.Div([
            html.Button("Supplier", id="supply_btn"),
            html.Button("No Relation", id="norel_btn", style={"margin-left": "15px"}),
            html.Button("Unsure", id="unsure_btn", style={"margin-left": "15px"})
        ], style={'justify-content': 'center',
                   'width': '100%', 'display': 'flex', 'align-items': 'center'})
    ], className="row"),

    # text
    html.H6(children='Sentence Text',
            style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),

    html.Div([

        html.Div([
            html.P(children="text goes here", id="current_sent"),
            html.Div(children="0", id="current_index", style={'display': 'none'}),
        ], className="eightColumns"),


    ], className="row"),

    # other sentences
    # html.H6(children='More Sentences',
    #         style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"}),
    #
    # html.Div([
    #     dash_table.DataTable(
    #         id='table-dropdown',
    #         data=df.loc[:100, select_col].to_dict('records'),
    #         columns=table_columns,
    #         editable=True,
    #         style_data={
    #             'whiteSpace': 'normal',
    #             'height': 'auto',
    #         },
    #         style_cell_conditional=[
    #             {
    #                 'if': {'column_id': 'text'},
    #                 'textAlign': 'left'
    #             }
    #         ],
    #         page_current=0,
    #         page_size=10,
    #         page_action="custom",
    #     )
    # ], className='row')
], style={'backgroundColor': '#DCDCDC'})

# ----
# callbacks
# ----

@app.callback([Output("table_sent_info", "data"),
               # Output("table-dropdown", "page_count"),
               Output("current_sent", "children"),
               Output("current_index", "children"),
               Output("gold_label_value", "children"),
               Output("gold_label_count", "children")],
              [Input('entity1_select', 'value'),
               Input('entity2_select', 'value'),
               Input('relation_select', 'value'),
               Input('weak_label_select', 'value'),
               Input('num_sent_select', 'value'),
               Input('epair_select', 'value'),
               Input('supply_btn', 'n_clicks'),
               Input('norel_btn', 'n_clicks'),
               Input('unsure_btn', 'n_clicks'),
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
def available_titles(e1, e2, rel, wl, ns, ep,
                     s_btn, nr_btn, us_btn, nx_btn, pv_btn, rn_btn,
                     # pc, ps,
                     cr_idx, glc, aal):

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
        'num_sent_select'
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


    print(f"return: {select_bool.sum()} values")
    print(f"current index: {cr_idx}")
    # print(type(cr_idx))

    # --
    # if a selection triggered callback
    # --

    # select data
    tmp = df.loc[select_bool]

    # if selection but generated the callback
    if button_id in selection_buttons:
        # set current index values to zero
        cr_idx = str(0)
        # get the gold label from database
        cur_id = tmp.iloc[int(cr_idx)]["id"]
        _ = art_db["gold_labels"].find_one(filter={"label_id": cur_id})
        glabel = _.get("gold_label", None)

    # otherwise, a iteration button has been clicked
    else:
        if button_id in ["supply_btn", "norel_btn", "unsure_btn"]:
            # TODO: here should confirm status - and write to data
            cur_id = tmp.iloc[int(cr_idx)]["id"]
            print(f"current sentence id: {cur_id}")

            if button_id == "supply_btn":
                glabel = "Supplier"
            elif button_id == "norel_btn":
                glabel = "NA"
            else:
                glabel = "unsure"
            print(f"setting gold label as {glabel}")
            art_db['gold_labels'].update_one(filter={"label_id": cur_id},
                                             update={"$set": {"gold_label": glabel}})

            # increment the gold label count
            glc = str(int(glc) + 1)

            # action after label
            # TODO: avoid the duplicate code
            if aal == "None":
                pass
            elif aal == "Next":
                cr_idx = int(cr_idx) + 1
                # allow for wrapping around
                cr_idx = 0 if cr_idx >= len(tmp) else cr_idx
            elif aal == "Prev":
                cr_idx = int(cr_idx) - 1
                # allow for wrapping around
                cr_idx = len(tmp) -1 if cr_idx < 0 else cr_idx
            elif aal == "Random":
                cr_idx = np.random.choice(np.arange(len(tmp)))

            # TODO: here decide if want to auto change sentence (randomly)

        else:
            print("changing sentence")
            if button_id == "prev_btn":
                cr_idx = int(cr_idx) - 1
                # allow for wrapping around
                cr_idx = len(tmp) -1 if cr_idx < 0 else cr_idx

            elif button_id == "next_btn":
                cr_idx = int(cr_idx) + 1
                # allow for wrapping around
                cr_idx = 0 if cr_idx >= len(tmp) else cr_idx
            else:
                # print("random button")
                # TODO: if picking random need to update pc and ps
                cr_idx = np.random.choice(np.arange(len(tmp)))

            # get the gold label from database
            cur_id = tmp.iloc[int(cr_idx)]["id"]
            _ = art_db["gold_labels"].find_one(filter={"label_id": cur_id})
            glabel = _.get("gold_label", None)

    # page_count = (len(tmp) // ps)

    # select sentence
    cur_sent = tmp.iloc[int(cr_idx)]["text"]

    # current sentence id
    # cur_id = tmp.iloc[int(cr_idx)]["id"]

    if glabel is None:
        glabel = "no label provided yet"

    # print(f"current page: {pc}")

    # select_range = int(cr_idx) + np.arange(pc*ps, (pc + 1)*ps)
    select_range = int(cr_idx)

    # page_count, \
    return tmp.iloc[[select_range]][info_col].to_dict('records'), \
           cur_sent,  \
           str(cr_idx), \
           glabel,\
           glc


if __name__ == "__main__":

    app.run_server(port=8051,
                   debug=True,
                   use_reloader=True,
                   host='0.0.0.0',
                   passthrough_errors=True)  # Turn off reloader if inside Jupyter

    # close mongo connection
    client.close()
