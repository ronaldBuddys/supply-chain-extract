# example file of how to download excel workbook from refinitiv

import json
import os
import sys
import re
import time
import numpy as np
import pandas as pd

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
from nlp import get_configs_path, get_data_path


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


def get_previously_fetch_and_select_subset(data_dir, page_info, select=None, seed_tickers=None):
    """given a data_dir and page_info get the previously fetched company info
    use select to take a subset, if None will take all
    """
    # previously downloaded files
    # - get just the ticker name
    fetched_files = [i for i in os.listdir(data_dir) if re.search(f"_{page_info}.xlsx", i)]
    fetched_tickers = [re.sub(f"_{page_info}.xlsx", "", i) for i in fetched_files]

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

    # convert 'Identifier' to str
    prev_fetched["Identifier"] = prev_fetched["Identifier"].values.astype('str')

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
    time.sleep(sleep/2)
    # clear is not working?
    tb_input.clear()
    # add BACKSPACE
    tb_input.send_keys(Keys.BACKSPACE)
    time.sleep(sleep)
    tb_input.send_keys(ticker + " " + page_info)
    time.sleep(sleep)
    tb_input.send_keys(Keys.RETURN)
    time.sleep(1.5 * sleep)
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
    max_fetch = 30

    # maximum number of exceptions to allow
    max_except = 10

    # load a json containing a path to firefox 'profile'
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
        "Last Update Date": "2018-01-01"
    }

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
    time.sleep(3*base_sleep)

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
        uname_input.send_keys(uname)
        time.sleep(2)
        # ---
        # password input
        # ---

        print("entering password")
        pwd_input = pwd_input[0]
        pwd_input.click()
        time.sleep(2)
        pwd_input.send_keys(pwd)
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


    # NOTE: can be very, very slow to load!
    print("having a quick nap while page loads")
    # TODO: change this to react to page - check for elements to be loaded
    time.sleep(5 * base_sleep)
    print("awake")

    # TODO: here check if landed on base_url
    # TODO: do this better, besides just waiting more
    if not bool(re.search(base_url, browser.current_url)):
        print(f"current url does not contain:\n{base_url}\nsleeping more")
        time.sleep(3 * base_sleep)

    assert bool(re.search(base_url, browser.current_url)), f"current_url does not contain: {base_url}"

    # ---
    # get downloaded file information
    # ---

    except_count = 0

    for pass_num in range(num_passes):
        print("*"*100)
        print("*"*100)
        print(f"starting pass: {pass_num+1} of {num_passes}")

        # get the company ids to fetch, and information on previously fetched
        fetch_id, prev_fetched = get_previously_fetch_and_select_subset(data_dir,
                                                                        page_info,
                                                                        select=select,
                                                                        seed_tickers=seed_ticker)

        print(f"will only select company ids that have ID%4 == {fmod}")
        fetch_id = [i for i in fetch_id if int(i) % 4 == fmod]

        assert len(fetch_id) > 0, "there are no company ids to fetch, change select criteria?"

        print(f"have: {len(fetch_id)} to fetch. max_fetch has been set to: {max_fetch}")

        # ---
        # increment over each company id
        # ---

        for tick_id, ticker in enumerate(fetch_id):


            if tick_id >= max_fetch:
                print("hit max_fetch, stopping (this pass)")
                break

            if except_count > max_except:
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

            search_page_info(browser, ticker, page_info)

            # ---
            # download information
            # ---

            # wait to load - should be checking page
            # TODO: should be checking if page has loaded
            print("quick snooze")
            time.sleep(base_sleep/2)

            # TODO: add print statements
            switch_resp = switch_to_relevant_iframe(browser, page_info, page_long_name)

            if switch_resp < 0:
                print("issue switching to iframe, skipping")
                continue

            time.sleep(2)

            # REMOVE!
            # check files in download directory - used to determine new files
            print(f"getting the files currently in data_dir:\n{data_dir}")
            old_file_list = np.array(os.listdir(data_dir))

            # NOTE: specifying not to be prompted to save xlsx file was
            # not working as expected - needed to change in
            # FireFox -> (hamburger, 3 horizontal lines, right side) -> (scroll down to) Applications ->
            # select action for Microsoft Excel Worksheet: Save File
            # - this needs to be done in the profile before running with selenium (profile gets copied?)

            # download button

            try:
                excel_btn = excel_download_button(browser, page_info)
            except selenium.common.exceptions.NoSuchElementException as e:
                # This error happens with big companies?
                print(e)
                except_count += 1
                continue
                # print(e)
                # print("trouble finding button? refresh page, have shnooze and then try again")
                # browser.refresh()
                # time.sleep(5*base_sleep)
                # try:
                #     excel_btn = excel_download_button(browser, page_info)
                #     time.sleep(base_sleep)
                # except selenium.common.exceptions.NoSuchElementException as e:
                #     print(e)
                #     print("giving up and skipping")
                #     base_sleep *= 1.1
                #     except_count += 1
                #     continue

            if excel_btn.is_displayed():
                excel_btn.click()
            else:
                # print("download button is not displayed!? - refreshing page and trying again")
                print("download button is not displayed!?, skipping")
                continue
                # browser.refresh()
                # time.sleep(2.5*base_sleep)

                try:
                    excel_btn = excel_download_button(browser, page_info)
                    excel_btn.click()
                except selenium.common.exceptions.ElementNotInteractableException as e:
                    print(e)
                    print("ElementNotInteractableException: issue clicking download button, moving on")
                    # TODO: keep track of how often this occurs? break if too many
                    # browser.refresh()
                    base_sleep *= 1.1
                    except_count += 1
                    continue
                except selenium.common.exceptions.NoSuchElementException as e:
                    print(e)
                    print("NoSuchElementException: issue clicking download button, moving on")
                    # TODO: keep track of how often this occurs? break if too many
                    # browser.refresh()
                    base_sleep *= 1.1
                    except_count += 1
                    continue

            # HACK: sleep to allow for download
            # - should be more careful checking -  see if there are an any 'part' files
            # - or keep checking until there is only one new file
            print("sleeping to let file download")
            time.sleep(base_sleep)
            new_file_list = np.array(os.listdir(data_dir))

            new_file = new_file_list[~np.in1d(new_file_list, old_file_list)]
            new_file = new_file[0]

            # rename file
            src_file = os.path.join(data_dir, new_file)
            dst_file = os.path.join(data_dir, f"{ticker}_{page_info}.xlsx")
            print(f"renaming:\n{src_file}\nto\n{dst_file}")
            os.rename(src=src_file,
                      dst=dst_file)



