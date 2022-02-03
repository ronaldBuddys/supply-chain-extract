# example file of how to download excel workbook from refinitiv

import json
import os
import sys
import re
import time
import numpy as np
import pandas as pd

import pyautogui

import selenium
from selenium.webdriver.common.keys import Keys

try:
    # python package (nlp) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None

from nlp.browser import start_browser
from nlp import get_configs_path, get_data_path, get_image_path



def ran_sleep(low=1, upper=None, scale=None):
    if upper is None:
        if scale is None:
            scale = 0.1
        upper = low * (1 + scale)
    time.sleep(np.random.uniform(low, upper))


def type_text(inp, text, low=0.1, upper=0.25, scale=None):
    assert isinstance(text, str), f"text needs to be string, its: {type(text)}"
    for t in text:
        ran_sleep(low=low, upper=upper, scale=scale)
        inp.send_keys(t)


def and_list_of_bools(x):
    """given a list of bools array (assumed to equal size!) combine with '&' operator"""
    assert isinstance(x, list), "input needs to be a list"
    if len(x) == 1:
        return x[0]
    elif len(x) == 0:
        return None
    else:
        return (x[0] &  and_list_of_bools(x[1:]))


def read_fetched_file(file_path):
    """
    read in company information file (i.e. VCHAIN)
    extract fetch_time,  parent company name and id
    return dataframe
    """

    assert os.path.exists(file_path), f"file_path:\n{file_path}\nDOES NOT EXIST"

    # get the bulk of info
    df = pd.read_excel(file_path, skiprows=5)
    # get the company info
    top = pd.read_excel(file_path, nrows=4, header=None)
    top = top.iloc[:, :2]

    # get the parent company information
    parent_company = top.loc[top[0] == "Company Name", 1].values[0]
    parent_id = top.loc[top[0] == "Company Id", 1].values[0]
    fetch_time = top.iloc[0, 1]
    fetch_time = fetch_time.strftime("%Y-%m-%d %H:%M:%S")

    # insert parent company information
    df.insert(loc=0, value=parent_id, column="Parent Id")
    df.insert(loc=0, value=parent_company, column="Parent Name")
    df["fetch_time"] = fetch_time

    return df


def get_previously_fetch_and_select_subset(data_dir, page_info, select=None, seed_tickers=None, industry_select=None):
    """given a data_dir and page_info get the previously fetched company info
    use select to take a subset, if None will take all
    """
    # previously downloaded files
    # - get just the ticker name
    fetched_files = [i for i in os.listdir(data_dir) if re.search(f"_{page_info}.xlsx$", i)]
    fetched_tickers = [re.sub(f"_{page_info}.xlsx$", "", i) for i in fetched_files]

    # read in previously fetched - to determine what to fetch next
    prev_fetched = [read_fetched_file(os.path.join(data_dir, i)) for i in fetched_files]

    # TODO: handle if None were previously fetched
    # TODO: determine if fetched data will always have the same columns! - ohh the inflexibility of tabular data
    try:
        prev_fetched = pd.concat(prev_fetched)
    except Exception as e:
        prev_fetched = pd.DataFrame(columns=['Parent Name', 'Parent Id', 'Identifier', 'Company Name', 'Type',
                                             'Relationship', 'Country/Region', 'Industry', 'Confidence Score (%)',
                                             'Last Update Date', 'Days Since Last Update', 'Freshness',
                                             'Snippet Count', 'Revenue (USD)', 'EQ Score', 'Implied Rating',
                                             'fetch_time'])

    # convert some columns to str
    prev_fetched["Identifier"] = prev_fetched["Identifier"].values.astype('str')
    # prev_fetched["Last Update Date"] = prev_fetched["Last Update Date"].astype(int)

    # ---
    # select subset based on some criteria
    # ---

    # potential new fetch should be those only that meet selection criteria
    if select is None:
        print("no selection criteria given: selecting all!")
        select_bool = [np.ones(len(prev_fetched), dtype=bool)]
    else:
        print("selecting entries based on:")
        print(json.dumps(select, indent=4))

        # slct = select.copy()

        # HACK: change value of key with date
        # because changed
        # for k in slct.keys():
        #     if re.search("Date", k, re.IGNORECASE):
        #         # THIS IS GROSS!
        #         slct[k]['value'] = pd.to_datetime(slct[k]['value']).to_numpy().astype(int)

        select_bool = [(prev_fetched[k] >= v).values for k,v in select.items()]
        # this is likely to be slow - very clunky
        # select_bool = [np.array([eval(f"'{kk}' {v['operator']} {v['value']}")
        #                               if isinstance(kk, str) else
        #                               eval(f"{kk} {v['operator']} {v['value']}")
        #                          for kk in prev_fetched[k].values])
        #                for k, v in slct.items()]

    # selection criteria will be AND together
    # TODO: review this, allow for OR
    select_bool = and_list_of_bools(select_bool)

    print(f"of {len( prev_fetched )} companies to fetch from, will select a subset of: {select_bool.sum()}")

    prev_fetched = prev_fetched.loc[select_bool]

    # ---
    # industry selection
    # ---

    if industry_select is not None:
        print(f"selecting only companies from Industries: {json.dumps(industry_select, indent=4)}")
        industry_select = industry_select if isinstance(industry_select, list) else list(industry_select)
        prev_fetched = prev_fetched.loc[prev_fetched["Industry"].isin(industry_select)]

    # change "Last Update Date" back to datetime64[ns]
    # prev_fetched["Last Update Date"] = prev_fetched["Last Update Date"].astype("datetime64[ns]")
    # ---
    # include any 'seed_tickers'
    # ---

    # get a list of potential company (IDs) to fetch
    if seed_tickers is None:
        seed_tickers = []
    elif isinstance(seed_tickers, list):
        pass
    else:
        seed_tickers = [seed_tickers]

    # fetch_id = np.unique(seed_tickers + [i for i in np.unique(prev_fetched["Identifier"].values)])
    # let the seed_tickers be first
    new_id = np.unique(prev_fetched["Identifier"].values)
    seed_tickers = np.array(seed_tickers)
    # drop any new_id in seed_tickers - to avoid duplication
    new_id = new_id[~np.in1d(new_id, seed_tickers)]
    fetch_id = np.concatenate([seed_tickers, new_id])

    print("excluding those already fetched")
    fetch_id = fetch_id[~np.in1d(fetch_id, fetched_tickers)]

    return fetch_id, prev_fetched


def get_all_suppliers(data_dir, page_info, parent_ticker, select=None, max_depth=2):

    print("getting ALL previously fetched data")
    # previously downloaded files
    # - get just the ticker name
    fetched_files = [i for i in os.listdir(data_dir) if re.search(f"_{page_info}.xlsx$", i)]
    # fetched_tickers = [re.sub(f"_{page_info}.xlsx$", "", i) for i in fetched_files]

    # read in previously fetched - to determine what to fetch next
    prev_fetched = [read_fetched_file(os.path.join(data_dir, i)) for i in fetched_files]

    # TODO: handle if None were previously fetched
    # TODO: determine if fetched data will always have the same columns! - ohh the inflexibility of tabular data
    try:
        prev_fetched = pd.concat(prev_fetched)
    except Exception as e:
        prev_fetched = pd.DataFrame(columns=['Parent Name', 'Parent Id', 'Identifier', 'Company Name', 'Type',
                                             'Relationship', 'Country/Region', 'Industry', 'Confidence Score (%)',
                                             'Last Update Date', 'Days Since Last Update', 'Freshness',
                                             'Snippet Count', 'Revenue (USD)', 'EQ Score', 'Implied Rating',
                                             'fetch_time'])

    # get all previously fetched
    # all_parent_id = np.unique(prev_fetched["Parent Id"].values)

    # convert some columns to str
    prev_fetched["Identifier"] = prev_fetched["Identifier"].values.astype('str')
    prev_fetched["Parent Id"] = prev_fetched["Parent Id"].astype('str')

    # store the full previously fetched data - for reference
    pf = prev_fetched.copy(True)

    # ---
    # select subset based on some criteria
    # ---

    # potential new fetch should be those only that meet selection criteria
    if select is None:
        print("no selection criteria given: selecting all!")
        select_bool = [np.ones(len(prev_fetched), dtype=bool)]
    else:
        print("selecting entries based on:")
        print(json.dumps(select, indent=4))

        select_bool = [(prev_fetched[k] >= v).values for k,v in select.items()]

    # selection criteria will be AND together
    # TODO: review this, allow for OR
    select_bool = and_list_of_bools(select_bool)

    print(f"of {len( prev_fetched )} companies to fetch from, will select a subset of: {select_bool.sum()}")

    prev_fetched = prev_fetched.loc[select_bool]

    # ----
    # given a parent ticker - get its suppliers, and theirs, and theirs, and so on
    # ---

    # given the parent - get
    keep_looking = True
    # get track of parents where already 'got suppliers'
    got_suppliers = []
    parents = np.array([parent_ticker])
    # TODO: could put in a max_depth into get_all_suppliers
    # depth = 1
    while keep_looking:
        # depth += 1
        parents_len = len(parents)
        print(f"number of parents: {parents_len}")
        # for each 'parent'/customer get their suppliers
        all_suppliers = []
        for p in parents:
            if p not in got_suppliers:
                b = (prev_fetched["Relationship"] == "Supplier") & (prev_fetched["Parent Id"] == p)
                supplies = np.unique(prev_fetched.loc[b, "Identifier"].values)
                got_suppliers += [p]
                all_suppliers += [supplies]
        # concatenate all suppliers to parents
        parents = np.concatenate([parents] + all_suppliers)
        parents = np.unique(parents)

        if parents_len == len(parents):
            keep_looking = False

    # ----
    # return
    # ---

    # company ids to search for - those not yet found
    fetch_id = parents[~np.in1d(parents, pf["Parent Id"].values)]


    return fetch_id, pf



def search_page_info(browser, ticker, page_info, sleep=2):

    # TODO: wrap up searching for a tickername (plus page_info) into a method
    # switch to AppFrame (iframe)
    # HARDCODE: iframe order on home page
    browser.switch_to.default_content()
    print("switching to relevant iframes")
    browser.switch_to.frame("AppFrame")
    browser.switch_to.frame("internal")
    browser.switch_to.frame("AppFrame")

    # get the (top) toolbar
    toolbar = browser.find_element_by_id("toolbar")
    tb_input = toolbar.find_element_by_tag_name("input")

    # send keys (search)
    print(f"entering ticker: {ticker} in search bar. sleep: {sleep} b/w actions")
    tb_input.click()
    ran_sleep(sleep/2)
    # clear is not working?
    tb_input.clear()
    # add BACKSPACE
    tb_input.send_keys(Keys.BACKSPACE)
    ran_sleep(sleep/2)
    # 'simulate' typing
    type_string = ticker + " " + page_info
    type_text(tb_input, type_string)
    # for ts in type_string:
    #     time.sleep(np.random.uniform(*type_sleep))
    #     tb_input.send_keys(ts)
    ran_sleep(sleep)
    tb_input.send_keys(Keys.RETURN)
    ran_sleep(1.5*sleep)
    print("search_page_info: COMPLETE")


def excel_download_button(browser, page_info):
    if page_info == "TREE":
        excel_btn = browser.find_element_by_tag_name("app-excel-export")
    # excel_btn.click()
    elif page_info == "VCHAINS":
        print("downloading excel file")
        # find div with "excel" in class name
        # divs = [d for d in browser.find_elements_by_tag_name("div") if re.search("excel", d.get_attribute("class"))]
        # HARDCODE: expect only one!
        # btn = [b for b in divs[0].find_elements_by_tag_name("button") if b.get_attribute("class") == "icon"]
        # [ for i in browser.find_elements_by_id("exportExcelButtonContainer") if i.get_attribute]
        span_btn = browser.find_element_by_id("exportExcelButtonContainer")
        # getting error: button is not displayed....
        span_btn.is_displayed()
        btn = [b for b in span_btn.find_elements_by_tag_name("button") if b.get_attribute("class") == "icon"]

        # span_btn.click()
        # btn[0].click()
        excel_btn = btn[0]

    return excel_btn


def switch_to_relevant_iframe(browser, page_info, page_long_name):
    print("switching to relevant iframes")
    # switch to default content - top iframe?
    browser.switch_to.default_content()
    browser.switch_to.frame("AppFrame")
    browser.switch_to.frame("internal")
    browser.switch_to.frame("AppFrame")

    # HARCODED: find iframe with 'Corp' in src? - will this always work?
    iframe = [i
              for i in browser.find_elements_by_tag_name("iframe")
              if re.search("Corp", re.sub(base_url, "", i.get_attribute("src")))]
    try:
        assert len(iframe) > 0, "expected more than one iframe with 'Corp' in 'src' attribute"
    except AssertionError as e:
        print(e)
        return -1
        # except_count += 1
        # continue

    browser.switch_to.frame(iframe[0])

    # AppFrame - again
    browser.switch_to.frame("AppFrame")

    # HARCODED: found page_long_name[page_info] in 'src'
    iframe = [i
              for i in browser.find_elements_by_tag_name("iframe")
              if re.search(page_long_name[page_info], re.sub(base_url, "", i.get_attribute("src")))]

    assert len(iframe) > 0, "expected more than one iframe from page_long_name[page_info] in 'src' attribute"
    browser.switch_to.frame(iframe[0])

    # AppFrame - again!!
    browser.switch_to.frame("AppFrame")

    return 0


def write_bad_ticker_no_result_to_file(bad_ticker, data_dir, suffix=None):

    if suffix is None:
        suffix = ''
    out_file = os.path.join(data_dir, f"bad_ticker{suffix}.csv")

    # if previous files exist - read those in, then append and re-write
    if os.path.exists(out_file):
        print("reading in previous bad_ticker info from")
        print(out_file)
        df = pd.read_csv(out_file)
        out = pd.concat([df, bad_ticker], axis=0)
    else:
        out = bad_ticker

    # write to file
    out.to_csv(out_file, index=False)


def check_page_for_no_result():

    no_res = [i for i in os.listdir(get_image_path()) if re.search("^no_result", i)]
    for nr in no_res:
        # search for image
        nr_loc = pyautogui.locateOnScreen(get_image_path(nr))
        # if found the image (it's not None) the have a bad / no search result

        if nr_loc is not None:
            print("looks like there was no results")
            return True

    return False


def refin_login(uname,pwd ):


    # NOTE: there are name="IDToken1"
    # NOTE: was having issuing with sending test to input, need to click on it?

    # TODO: add try /excepts here for error handling

    all_inputs = browser.find_elements_by_tag_name('input')

    # identify input cells
    uname_input = [i for i in all_inputs if i.get_attribute("placeholder") == "User ID"]
    pwd_input = [i for i in all_inputs if i.get_attribute("placeholder") == "Password"]

    # TODO: but a check here to sleep and try again if len(uname_input) == 0

    # ---
    # username input
    # ---
    print("entering username")
    uname_input = uname_input[0]
    uname_input.click()
    time.sleep(2)
    type_text(uname_input, uname)
    # uname_input.send_keys(uname)
    time.sleep(2)
    # ---
    # password input
    # ---

    print("entering password")
    pwd_input = pwd_input[0]
    pwd_input.click()
    time.sleep(2)
    # pwd_input.send_keys(pwd)
    type_text(pwd_input, pwd)
    time.sleep(2)
    pwd_input.send_keys(Keys.RETURN)

    # just hit enter? or click on 'Sign In'?
    # browser.send_keys(Keys.RETURN)

    # TODO: check if asking to Sign out of previous session
    try:
        time.sleep(base_sleep)
        form = [f for f in browser.find_elements_by_tag_name("form") if f.get_attribute("name") == "frmSignIn"]
        form = form[0]
        sign_in = [d for d in form.find_elements_by_tag_name("div") if d.text == 'Sign In'][0]
        sign_in.click()
        # time.sleep(4*base_sleep)

    except Exception as e:
        print(e)



if __name__ == "__main__":

    pd.set_option("display.max_columns", 100)
    # TODO: for those that are not downloading - add to a list, of issue_downloading to do manually later
    # TODO: wrap up below into function(s) or maybe class
    # TODO: should read from file / table
    # TODO: should cross reference against already fetched
    # TODO: should have selection criteria of which to take next (market cap, date, or something)
    # TODO: should continue to loop over next companys
    # TODO: save files / write tables using company ID - will avoid special characters
    # TODO: have a file / table of with ID, company name
    # TODO: add noise to sleep amount
    # TODO: let sleep (base) amount be a parameter
    # TODO: change select criteria to be more robust, allow for matching on strings (i.e. include an operator to use)

    # ---
    # parameters
    # ---

    # number of passes
    # - the number of times to check previously fetched, get company ids and search those
    num_passes = 4

    # base sleep amount
    base_sleep = 10

    # starting or seed ticker
    # ticker = "AAPL"
    # GM:
    seed_ticker = "4298546138"
    # to find a 'permid' go to company page -> Overview (in ribbon) -> Codes and Schemes ->
    # under CODES: Company PERMID

    # maximum number of tickers to fetch
    max_fetch = 100

    # maximum number of exceptions to allow
    max_errors = 10

    # load a json containing a path to firefox 'profile'
    # REMOVE: this if not needed
    # TODO: allow for a default if not provided
    with open(get_configs_path("profiles.json"), "r") as f:
        profiles = json.load(f)

    # fetch modulus
    # - determine which company numbers to fetch
    fetch_mod_file = get_configs_path("fetch_mod.json")

    assert os.path.exists(fetch_mod_file), \
        f"file:\n{fetch_mod_file}\ndoes not exist, it should and be in format of:\n{json.dumps({'fetch_modulus': '#'})}\n" \
        f"with '#' being on of [0,1,2,3]"

    with open(fetch_mod_file, "r") as f:
        fmod = json.load(f)

    fmod = fmod["fetch_modulus"]

    page_info = "VCHAINS"

    # need to determine which iframe to select
    page_long_name = {
        "VCHAINS": "ValueChains",
        "TREE": "CompanyTreeStructure"
    }

    # base url for workspace - use to deal with src
    # - this is the page should check landed on after login
    base_url = 'https://emea1.apps.cp.thomsonreuters.com'

    # selection criteria - will take those above
    # TODO: this should be an input arg or read from file - input arg could be file name
    select = {
        "Confidence Score (%)": 0.95,
        "Last Update Date": "2017-01-01"
    }
    # "Industry": {"operator": "in", "value": ['Auto, Truck & Motorcycle Parts', 'Auto & Truck Manufacturers']}
    # TODO: need to have a think about how to best keep focus - not branch too far out (if so desired)
    # select from specific industries? set to None if want everything
    industry_select = [
        'Auto & Truck Manufacturers',
        'Auto Vehicles, Parts & Service Retailers',
        'Auto, Truck & Motorcycle Parts',
        'Ground Freight & Logistics',
        'Heavy Machinery & Vehicles',
        'Iron & Steel',
        'Tires & Rubber Products',
        'Semiconductor Equipment & Testing']
    # industry_select = None

    # --
    # location to download files to
    # --

    # store in data directory in package
    # TODO: review, ideally would be storing in a shared database
    data_dir = get_data_path(page_info)
    os.makedirs(data_dir, exist_ok=True)

    # ---
    # fire up browser
    # ---

    # provide the
    browser = start_browser(profile_loc=profiles["firefox"], out_dir=data_dir)
    # no profile - easier to identify as a bot?
    # browser = start_browser(profile_loc=None, out_dir=data_dir)

    # visit refinitiv workspace
    refin_url = "https://workspace.refinitiv.com/web"

    print(f"loading page: {refin_url}")
    browser.get(refin_url)

    print("sleep while login page loads, if already logged in should eventually go to workspace")
    ran_sleep(3*base_sleep)

    # TODO: handle login
    # NOTE: found that the below if statement could be true, but then previously
    # used credential would work and
    # - check if login is in name?
    if re.search("login", browser.current_url):
        print("need to login?")

        # config containing refinitiv username and password
        # NOTE: found logging on with a profile that was already logged on did not require a password
        # TODO: only need username and password if not already signed in - get this only if required
        login_conf = get_configs_path("refinitiv.json")
        assert os.path.exists(login_conf), \
            f"login config:\n{login_conf}\nnot found, should exist and be in format:\n" \
            f"{json.dumps({'username': 'first.lastname.21@ucl.ac.uk', 'password': 'refinitiv_password'})}\n"

        with open(login_conf, "r") as f:
            ref_dets = json.load(f)

        uname = ref_dets["username"]
        pwd = ref_dets["password"]

        # login to refinitiv
        refin_login(uname, pwd)


    # NOTE: can be very, very slow to load!
    print("having a quick nap while page loads")
    # TODO: change this to react to page - check for elements to be loaded
    ran_sleep(4.5 * base_sleep)
    print("awake")

    # TODO: here check if landed on base_url
    # TODO: do this better, besides just waiting more
    if not bool(re.search(base_url, browser.current_url)):
        print(f"current url does not contain:\n{base_url}\nsleeping more")
        ran_sleep(3 * base_sleep)

    assert bool(re.search(base_url, browser.current_url)), f"current_url does not contain: {base_url}"

    # ---
    # get downloaded file information
    # ---

    # keep track of exceptions / errors
    except_count = 0
    fetch_count = 0

    for pass_num in range(num_passes):
        print("*"*100)
        print("*"*100)
        print(f"starting pass: {pass_num+1} of {num_passes}")

        # get the company ids to fetch, and information on previously fetched
        # fetch_id, prev_fetched = get_previously_fetch_and_select_subset(data_dir,
        #                                                                 page_info,
        #                                                                 select=select,
        #                                                                 seed_tickers=seed_ticker,
        #                                                                 industry_select=industry_select)

        fetch_id, prev_fetched = get_all_suppliers(data_dir, page_info,
                                                   parent_ticker=seed_ticker,
                                                   select=select)

        # read all the bad_ticker data
        # TODO: should this read all bad ticker data ?
        # TODO: the content
        bad_tickers = [pd.read_csv(os.path.join(data_dir, i))
                       for i in os.listdir(data_dir)
                       if re.search("^bad_ticker", i)]
        bad_tickers = pd.concat(bad_tickers)
        bad_ids = bad_tickers["Identifier"].values.astype('str')

        print("dropping ids that were previously identified as 'bad' - had issues when searching for them")
        fetch_id = [i for i in fetch_id if i not in bad_ids]

        if fmod is not None:
            print(f"will only select company ids that have ID%4 == {fmod}")
            fetch_id = [i for i in fetch_id if int(i) % 4 == fmod]

        assert len(fetch_id) > 0, "there are no company ids to fetch, change select criteria?"

        print(f"have: {len(fetch_id)} to fetch. max_fetch has been set to: {max_fetch}")

        # ---
        # increment over each company id
        # ---

        for tick_id, ticker in enumerate(fetch_id):

            # --
            # checks on iteration
            # --

            if fetch_count >= max_fetch:
                print("hit max_fetch, stopping")
                break

            if except_count > max_errors:
                print("too many exception occurred, forcing stop")
                break

            print("-"*10)
            # show why fetching this ticker?
            if ticker in prev_fetched["Identifier"].values:
                print(f"fetching ticker: {ticker}, which came up previously:")
                print(prev_fetched.loc[ prev_fetched["Identifier"] == ticker, :])

            # ---
            # search page information for ticker
            # ---

            old_url = browser.current_url
            search_page_info(browser, ticker, page_info)

            print("quick snooze")
            ran_sleep(1.0 * base_sleep)

            new_url = browser.current_url

            if old_url == new_url:
                print("URL did not change! will re-load")
                except_count += 1
                browser.get(base_url)
                ran_sleep(2.5 * base_sleep)

            # ---
            # check page for no_results
            # --

            no_res = check_page_for_no_result()

            if no_res:
                # HACK: if there are no results the search bar gets affected
                # - and does not get cleared as expected
                # - so just reload the app
                # - however re-loading page seems slow as
                print("re-loading web app")
                browser.get(base_url)
                # using back does not work?
                # print("going back to previous page")
                # browser.back()

                ran_sleep(1.5 * base_sleep)

                # write bad search results to file - to keep track of
                # TODO: this could break if ticker is a seed_ticker, should handle
                #  - that case bad_ticker will have len = 0
                bad_ticker = prev_fetched.loc[prev_fetched["Identifier"] == ticker, :]

                # effectively append current bad ticker data to existing file
                # TODO: when using database just use append
                write_bad_ticker_no_result_to_file(bad_ticker, data_dir, suffix=fmod)
                print("skipping")
                continue

            # ---
            # download excel file
            # ---

            # TODO: should have checks here to make sure the page has loaded

            # REMOVE!
            # check files in download directory - used to determine new files
            print(f"getting the files currently in data_dir:\n{data_dir}")
            old_file_list = np.array(os.listdir(data_dir))

            # wait to load - should be checking page
            # TODO: should be checking if page has loaded

            # download excel images
            excel_button = [i for i in os.listdir(get_image_path()) if re.search("^download", i)]
            for eb in excel_button:
                button_loc = pyautogui.locateOnScreen(get_image_path(eb))
                if button_loc is not None:
                    print("found download button")
                    break

            if button_loc is None:
                print("could not find excel download button! skipping")
                # TODO: consider including as 'bad' ticker - or perhaps use different category
                # bad_ticker = prev_fetched.loc[prev_fetched["Identifier"] == ticker, :]
                # write_bad_ticker_no_result_to_file(bad_ticker, data_dir, suffix=fmod)
                except_count += 1
                continue

            # get the center of button
            button_point = pyautogui.center(button_loc)
            # and click it
            print("clicking download button")
            pyautogui.click(button_point)

            # HACK: sleep to allow for download
            # - should be more careful checking -  see if there are an any 'part' files
            # - or keep checking until there is only one new file
            print("sleeping to let file download")
            ran_sleep(0.5 * base_sleep)

            # ---
            # check for new file
            # ---

            # TODO: want to avoid having .part files - should only check for xlsx in both file_lists
            check = True
            check_count = 0
            while check:
                new_file_list = np.array(os.listdir(data_dir))

                new_file = new_file_list[~np.in1d(new_file_list, old_file_list)]
                if len(new_file) > 0:
                    new_file = new_file[0]
                    check = False
                else:
                    check_count += 1
                    time.sleep(2)

                if check_count > 10:
                    check = False

            if not isinstance(new_file, str):
                print("issue finding new file: didn't get saved?")
                continue

            # rename file
            src_file = os.path.join(data_dir, new_file)
            dst_file = os.path.join(data_dir, f"{ticker}_{page_info}.xlsx")
            print(f"renaming:\n{src_file}\nto\n{dst_file}")
            os.rename(src=src_file,
                      dst=dst_file)

            fetch_count += 1
    print("FIN!")
