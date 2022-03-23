
import os
import re
import sys
import json

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


from nlp.utils import get_database, get_list_from_tree, make_reg_tree
from nlp import get_configs_path, get_data_path


# ---
# connect to database
# ---


# get credentials
with open(get_configs_path("mongo.json"), "r+") as f:
    mdb_cred = json.load(f)

# get mongodb client - for connections
client = get_database(username=mdb_cred["username"],
                      password=mdb_cred["password"],
                      clustername=mdb_cred["cluster_name"])

art_db = client["news_articles"]


# ---
# create an Dash app - referring to some css template
# ---

# use a 'default' css file - could save locally if running offline
app = dash.Dash(__name__,
                external_stylesheets=[
                    'https://codepen.io/chriddyp/pen/bWLwgP.css'
                ])
app.title = "View Articles"

# ---
# parameters
# ---

text_color = "#1314fc"

# a = art_db["articles"].find_one()

# get the unique names
unique_names = art_db["articles"].distinct("names_in_text")

# put the long names into a list of dictionaries - use for selection
unames_list_dict = [{'label': i, 'value': i} for i in unique_names]

# find an article that has more than one names_in_text
# ref: https://stackoverflow.com/questions/7811163/query-for-documents-where-array-size-is-greater-than-1
# a = art_db["articles"].find_one(filter={"names_in_text.2": {"$exists": True}})

# titles = art_db["articles"].find_one(filter={"names_in_text": {"$in": ['3M Co']}},
#                                      projection={"title": 1})


# ---
# One off - should move this else where
# ---
#
# bulk_update = [UpdateOne(filter={"long_name": ln},
#                          update={'$addToSet':
#                                       {"short_names": {"$each": []}}
#                                   },
#                          upsert=True)
#                for ln in unique_names]
#
# art_db["long_to_short_name"].bulk_write(bulk_update)
# add a 'checked' field to long to short name documents
# art_db["long_to_short_name"].update_many({"short_names": {"$not": {"$size": 0}}},
#                                          update={"$set": {"checked": True}})
#
# art_db["long_to_short_name"].update_many({"short_names":  {"$size": 0}},
#                                          update={"$set": {"checked": False}})



# ---
# helper functions
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

default_text = ["here is some text", html.Mark(" highlight "), "more text ", html.Br(), "after"]

app.layout = html.Div([

    # Header
    html.Div([
        # app 'name'
        html.H1(children='Article Viewer: Long to Short Name Review',
                style={'color': text_color, 'textAlign': 'left', "font-weight": "bold"},
                className='twelve columns'),
    ], className='row'),

    # Row 1 - each row is twelve columns wide
    # row 1
    html.Div([
        # selection criteria
        html.Div([
            # ----
            # Company Selection
            # ----
            html.H3("Select Company"),
            # radio dial to only select those with no short names
            html.Label('Show only those un-checked?', style={'color': text_color, "font-weight": "bold"}    ),
            dcc.RadioItems(
                options=[{'label': i, 'value': i} for i in ['True', 'False']],
                id='missing_short_names',
                value='None',
                style={'color': 'black'},
                labelStyle={'display': 'inline-block'}
            ),
            html.Div("", id="number_of_companies_to_select_from"),
            # TODO: provide some stats
            # long name select
            html.Label("Long Name Select"),
            dcc.Dropdown(
                id='long_name_dropdown',
                options=unames_list_dict,
                placeholder="select long name",
                style={'fontSize': '12px'}),
            # buttons for moving along to company
            # html.Div([
            #     html.Button("Prev", id="prev_company", className="one columns"),
            #     html.Button("Next", id="next_company", className="one columns")
            # ]),
            # short names - show below long name
            html.Label("Short Names:"),
            html.Div(children="", id="short_name_values", style={"font-weight": "bold"}),
            # add - short name
            # hidden div - need for output for confirm short name button click
            html.Div(children="", id="is_checked_status"),#, style={'display': 'none'}),
            html.Button("Confirm Short Name(s)", id="confirm_shortnames_btn"),
            html.Br(),
            html.Label("Add"),
            dcc.Input(id="add_short_input",
                      value='',
                      type="text",
                      placeholder="enter short name to add",
                      debounce=True),
            # remove - short name(s)
            html.Label("Remove"),
            dcc.Dropdown(
                id='remove_names',
                # options=unames_list_dict,
                placeholder="select short name to remove",
                style={'fontSize': '12px'},
                multi=True),
            html.Button("Confirm Remove", id="remove_shortnames_btn"),

            # ----
            # Article Information
            # ----
            # article titles
            html.Br(),
            html.H3("Article Info"),
            html.Div("", id="number_of_articles"),
            html.Div("article index: 0", id="current_article_idx"),
            html.Br(),
            html.Label("Articles Containing Long Name"),
            html.Br(),
            dcc.Dropdown(
                id='title_w_long_name',
                # options=unames_list_dict,
                placeholder="select an article",
                style={'fontSize': '12px'}),
            html.Div(id="tmp"),
            # next / prev buttons for articles
            # TODO: fix text alignment
            html.Br(),
            html.Div([
                html.Button("Prev", id="prev_article", className="one columns",  style={'textAlign': 'center'}),
                html.Button("Next", id="next_article", className="one columns",  style={'textAlign': 'left'})
            ])
        ], className="three columns"),
        # article contents
        html.Div([
            html.Label('Title', style={'color': text_color, "font-weight": "bold"}),
            html.Div(id='title_value', style={'fontSize': 14, 'fontWeight': 'bold'}),
            html.Label('Source', style={'color': text_color, "font-weight": "bold"}),
            html.Div(id='source_value', style={'fontSize': 14, 'fontWeight': 'bold'}),
            html.Label('Publish Date', style={'color': text_color, "font-weight": "bold"}),
            html.Div(id='pub_date', style={'fontSize': 14, 'fontWeight': 'bold'}),
            html.Label('Names In Text', style={'color': text_color, "font-weight": "bold"}),
            html.Div(id="names_in_text"),
            html.Label('Main text', style={'color': text_color, "font-weight": "bold"}),
            html.Div(children="", id="main_text", style={'display': 'inline-block'})
            # in practice, it's a bad idea to allow this - due to nefarious javascript injection (?)
            # dcc.Markdown(a["maintext"], dangerously_allow_html=True)
            # html.Div([
            #     html.P(a["maintext"])
            # ], id='text_value', style={'fontSize': 14, 'fontWeight': 'bold'}),
            # html.Iframe(srcDoc=a["maintext"])
        ], className="nine columns"),

    ], className="row"),

], style={'backgroundColor': '#DCDCDC'})

# ----
# callbacks
# ----

@app.callback([Output('long_name_dropdown', 'options'),
               Output('number_of_companies_to_select_from', 'children')],
              Input('missing_short_names', 'value'))
def show_only_long_names_without_short_name(missing_short_names):

    if missing_short_names is None:
        raise PreventUpdate

    if missing_short_names == "True":
        # get only the long_names that have short_name array length = 0
        out = [{"label": i["long_name"], "value": i["long_name"]}
               for i in
               art_db["long_to_short_name"].find(filter={"checked": False})]
        return out, f"Number of companies in list: {len(out)}"
    else:
        #
        return [{'label': i, 'value': i} for i in unique_names],  f"Number of companies in list: {len(unique_names)}"

# given the selected long name, show the short names
@app.callback([Output('short_name_values', 'children'),
               Output('remove_names', 'options'),
               Output('add_short_input', 'value')],
              [Input('long_name_dropdown', 'value'),
               Input('add_short_input', 'value'),
               Input('remove_shortnames_btn', 'n_clicks')],
              [State('remove_names', 'value')])
def show_short_names(long_name, add_short_name, remove_button, names_to_remove):

    # determine which Input triggered callback
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "No clicks yet":
        raise PreventUpdate

    if long_name is None:
        raise PreventUpdate
    elif long_name == "":
        raise PreventUpdate

    # if selected a long_name - get current short names from database
    if button_id =='long_name_dropdown':
        # when app is initialised the Input can be None initially, so just ignore
        print(f"selected long name: {long_name}")

    # add short name to database
    elif button_id == 'add_short_input':

        if add_short_name is None:
            raise PreventUpdate
        elif add_short_name == "":
            raise PreventUpdate
        print(f"adding short name: {add_short_name}")
        art_db["long_to_short_name"].update_one(filter={"long_name": long_name},
                                                update={"$addToSet": {"short_names": add_short_name}},
                                                upsert=True)
    elif button_id == 'remove_shortnames_btn':

        if len(names_to_remove) == 0:
            print("no names to remove, doing nothing")
            raise PreventUpdate
        # xx = 1
        print(f"removing short name(s): {names_to_remove}")
        art_db["long_to_short_name"].update_one(filter={"long_name": long_name},
                                                update={"$pull": {"short_names": {"$in": names_to_remove}}},
                                                upsert=True)

    # search the database for the long - to get the current results
    names_dict = art_db["long_to_short_name"].find_one(filter={"long_name": long_name})
    if names_dict is None:
        return f"long_name: {long_name} was not found!!", \
                [{'label': i, 'value': i} for i in []], \
                ""
    else:
        return list_to_text(names_dict["short_names"]), \
               [{'label': i, 'value': i} for i in names_dict["short_names"]], \
                ""


@app.callback(Output("is_checked_status", "children"),
              [Input("confirm_shortnames_btn", "n_clicks"),
              Input('long_name_dropdown', 'value')])
def confirm_short_names(conf_btn, long_name):
    # change the 'checked' field in long_to_short_name doc to True

    # determine which Input triggered callback
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if conf_btn is None:
        raise PreventUpdate

    if long_name is None:
        raise PreventUpdate

    if long_name == "":
        raise PreventUpdate

    if button_id == "confirm_shortnames_btn":
        print("confirming short names")

        # art_db["long_to_short_name"]
        art_db["long_to_short_name"].update_many({"long_name": long_name},
                                                 update={"$set": {"checked": True}})
        output = "short name(s) confirmed"
    elif button_id == "long_name_dropdown":
        tmp = art_db["long_to_short_name"].find_one({"long_name": long_name})
        output = "short name(s) confirmed" if tmp["checked"] else "short name(s) UNCONFIRMED"
    else:
        # this should never be the case
        output = "unknown input triggered callback confirm_short_names"

    return output

@app.callback([Output('title_value', 'children'),
               Output('source_value', 'children'),
               Output('pub_date', 'children'),
               Output('names_in_text', 'children'),
               Output('main_text', 'children')],
              [Input('title_w_long_name', 'value'),
               Input('short_name_values', 'children')],
              [State('long_name_dropdown', 'value')])
def article_update(title, short_names, long_name):
    # given a long name select an article

    if long_name is None:
        raise PreventUpdate
    elif long_name == "":
        raise PreventUpdate

    if title is None:
        raise PreventUpdate
    elif title == "":
        raise PreventUpdate

    # TODO: include Article title as input
    # find one (for now)
    a = art_db["articles"].find_one(filter={"names_in_text": {"$in": [long_name]},
                                            "title": title})
    if a is None:
        print(f"NO ARTICLES FOUND FOR: {long_name}")
        raise PreventUpdate

    print(f"updating article for: {long_name}, title: {a['title'][:50]}...")

    # color text for long name
    color_text = [{'word': long_name, "style": {"backgroundColor": "yellow", 'display': 'inline-block'}}]
    # if short names
    if short_names is None:
        pass
    elif short_names == "":
        pass
    else:
        # get short names in decreasing order
        short_name_list = short_names.split(",")
        short_name_list.sort(reverse=True, key=lambda x: len(x))
        # HARDCODED: splitting names on ',' - don't expect comma to be in company name
        more_color_text = [{'word': i.lstrip().rstrip(), "style": {"backgroundColor": "pink", 'display': 'inline-block'} }
                           for i in short_name_list]
        color_text += more_color_text

    formated_text = text_to_dash_html(a['maintext'],
                                      color_text=color_text)

    return a['title'], \
           a['source_domain'],\
           a['date_publish'], \
           list_to_text(a["names_in_text"]), \
           formated_text


@app.callback([Output("title_w_long_name", "options"),
               Output("number_of_articles", "children")],
              [Input('long_name_dropdown', 'value')])
def available_titles(long_name):
    # get a list of available titles, and provide the first one
    if long_name is None:
        raise PreventUpdate
    elif long_name == "":
        raise PreventUpdate

    # print("available_titles")
    # print(long_name)
    # get (just) the title of the articles with names_in_text
    # projection={"title": 1}
    titles = art_db["articles"].find(filter={"names_in_text": {"$in": [long_name]}},
                                         projection={"title": 1})

    # HARDCODED: reduce the size of the label
    out = [{'label': t["title"][:60], 'value': t["title"]} for t in titles]

    return out, f"Number of Articles: {len(out)}"


@app.callback([Output("title_w_long_name", "value"),
               Output("current_article_idx", "children")],
              [Input("title_w_long_name", "options"),
               Input('prev_article', 'n_clicks'),
               Input('next_article', 'n_clicks')],
              [State("title_w_long_name", "value"),
               State("current_article_idx", "children")])
def select_title(title_options, prev_btn, next_btn, current_title, current_article_idx):
    # if the title options are update - select a value (title)

    # determine which Input triggered callback
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if title_options is None:
        raise PreventUpdate
    elif len(title_options) == 0:
        raise PreventUpdate

    # if options for title with long name were update (triggered callback)
    # - just return the first (or current?) article index
    if button_id == "title_w_long_name":
        current_article_idx = int(re.sub("\D", "", current_article_idx)) if current_article_idx is None else 0
        return title_options[current_article_idx]["value"], f"article index: {current_article_idx}"
    # otherwise, it was next/prev button click
    elif (button_id == "prev_article") | (button_id == "next_article"):

        if current_title is None:
            raise PreventUpdate

        if current_article_idx is None:
            i = 0
        else:
            i = int(re.sub("\D", "", current_article_idx))

        if button_id == "next_article":
            if next_btn is None:
                raise PreventUpdate
            i_increment = 0 if (i + 1) == len(title_options) else (i + 1)
        else:
            if prev_btn is None:
                raise PreventUpdate
            i_increment = len(title_options) - 1 if (i-1) < 0 else i-1

        return title_options[i_increment]["value"], f"article index: {i_increment}"



if __name__ == "__main__":

    app.run_server(port=8050,
                   debug=True,
                   use_reloader=True,
                   host='0.0.0.0',
                   passthrough_errors=True)  # Turn off reloader if inside Jupyter

    # close mongo connection
    client.close()
