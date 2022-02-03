# example file of how to download excel workbook from refinitiv

import json
import os
import sys
import re
import time
import numpy as np

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

if __name__ == "__main__":
    # TODO: wrap up below into function(s) or maybe class
    # TODO: should read from file / table
    # TODO: should cross reference against already fetched
    # TODO: should have selection criteria of which to take next (market cap, date, or something)
    # TODO: should continue to loop over next companys
    # TODO: save files / write tables using company ID - will avoid special characters
    # TODO: have a file / table of with ID, company name
    # TODO: add noise to sleep amount
    # TODO: let sleep (base) amount be a parameter

    # ---
    # parameters
    # ---

    # starting ticker
    ticker = "AAPL"

    # load a json containing a path to firefox 'profile'
    with open(get_configs_path("profiles.json"), "r") as f:
        profiles = json.load(f)

    page_info = "VCHAINS"

    # need to determine which iframe to select
    page_long_name = {
        "VCHAINS": "ValueChains",
        "TREE": "CompanyTreeStructure"
    }

    # config containing refinitiv username and password
    # NOTE: found logging on with a profile that was already logged on did not require a password
    # with open(get_configs_path("refinitiv.json"), "r") as f:
    #     ref_dets = json.load(f)
    #
    # uname = ref_dets["username"]
    # pwd = ref_dets["password"]

    # --
    # location to download files to
    # --

    # store in data directory in package
    # TODO: review, ideally would be storing in a shared database
    data_dir = get_data_path("value_chain")
    os.makedirs(data_dir, exist_ok=True)

    # ---
    # fire up browser
    # ---

    # provide the ta
    browser = start_browser(profile_loc=profiles["firefox"], out_dir=data_dir)
    # browser = start_browser(profile_loc=None, out_dir=data_dir)

    # visit refinitiv workspace
    refin_url = "https://workspace.refinitiv.com/web"

    print(f"loading page: {refin_url}")
    browser.get(refin_url)
    time.sleep(10)

    # TODO: handle login
    # NOTE: found that the below if statement could be true, but then previously
    # used credential would work and
    # - check if login is in name?
    if re.search("login", browser.current_url):
        print("need to login")
        # NOTE: there are name="IDToken1"
        # NOTE: was having issuing with sending test to input, need to click on it?
        # uname_input = browser.find_element_by_name("IDToken1")
        # uname_input.click()
        # uname_input.send_keys(uname)

    # NOTE: can be very, very slow to load!
    print("having a quick nap while page loads")
    # TODO: change this to react to page - check for elements to be loaded
    time.sleep(50)
    print("awake")

    # base url for workspace - use to deal with src
    base_url = 'https://emea1.apps.cp.thomsonreuters.com'

    # set window size
    # TODO: review if this is needed
    # print("re-sizing window")
    # browser.set_window_size(2000, 600)


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
    print(f"entering ticker: {ticker} in search bar")
    tb_input.click()
    time.sleep(2)
    tb_input.send_keys(ticker + " " + page_info)
    time.sleep(2)
    tb_input.send_keys(Keys.RETURN)
    time.sleep(3)


    # wait to load - should be checking page
    # TODO: should be checking if page has loaded
    print("quick snooze")
    time.sleep(10)

    # TODO: add print statements
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

    assert len(iframe) > 0, "expected more than one iframe with 'Corp' in 'src' attribute"
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

    # REMOVE!
    # check files in download directory - used to determine new files
    old_file_list = np.array(os.listdir(data_dir))

    # NOTE: specifying not to be prompted to save xlsx file was
    # not working as expected - needed to change in
    # FireFox -> (hamburger, 3 horizontal lines, right side) -> (scroll down to) Applications ->
    # select action for Microsoft Excel Worksheet: Save File
    # - this needs to be done in the profile before running with selenium (profile gets copied?)

    # download button
    if page_info == "TREE":
        excel_btn = browser.find_element_by_tag_name("app-excel-export")
        excel_btn.click()
    elif page_info == "VCHAINS":
        print("downloading excel file")
        # find div with "excel" in class name
        divs = [d for d in browser.find_elements_by_tag_name("div") if re.search("excel", d.get_attribute("class"))]
        # HARDCODE: expect only one!
        btn = [b for b in divs[0].find_elements_by_tag_name("button") if b.get_attribute("class") == "icon"]
        btn[0].click()

    # HACK: sleep to allow for download
    # - should be more careful checking -  see if there are an any 'part' files
    # - or keep checking until there is only one new file
    time.sleep(10)
    new_file_list = np.array(os.listdir(data_dir))

    new_file = new_file_list[~np.in1d(new_file_list, old_file_list)]
    new_file = new_file[0]

    # rename file
    src_file = os.path.join(data_dir, new_file)
    dst_file = os.path.join(data_dir, f"{ticker}_{page_info}.xlsx")
    print(f"renaming:\n{src_file}\nto\n{dst_file}")
    os.rename(src=src_file,
              dst=dst_file)



