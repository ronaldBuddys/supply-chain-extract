import json
import os
import re
import pandas as pd
from supply_chain_extract.utils import get_database
from supply_chain_extract import get_configs_path, get_data_path

from supply_chain_extract.examples.knowledge_base_excel_download_w_autogui import read_fetched_file

if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)

    # get credentials
    with open(get_configs_path("mongo.json"), "r+") as f:
        mdb_cred = json.load(f)

    # get mongodb client - for connections
    client = get_database(username=mdb_cred["username"],
                          password=mdb_cred["password"],
                          clustername=mdb_cred["cluster_name"])

    # get files from page_info downloads
    page_info = "VCHAINS"

    # connect to knowledge_base database
    db = client["knowledge_base"]

    # directory where data is stored (locally)
    data_dir = get_data_path(page_info)

    # vchains database to contain
    # - collection per company vchain result
    # - bad_ticker collection

    # ---
    # read in previously results
    # ---

    fetched_files = [i for i in os.listdir(data_dir) if re.search(f"_{page_info}.xlsx$", i)]

    for ff in fetched_files:
        df = read_fetched_file(os.path.join(data_dir, ff))

        # check Parent Id matches file name
        pid = df['Parent Id'].unique()
        assert len(pid) == 1, f"got more than one parent id for: {ff}"
        pid = int(pid[0])

        if re.search(f"^{pid}", ff):
            # upload file as documents - if the Parent Id can't be found

            # check if parent ID is already in
            res = db[page_info].find_one({'Parent Id': pid})

            # add company if not found before
            if res is None:
                print(f"adding documents for: {pid}")
                print(f"company name: {df['Parent Name'].unique()[0]}")
                db[page_info].insert_many(df.to_dict('records'))
            else:
                # TODO: allow for adding if fetch_time is more recent - probably not needed in short term
                # res["fetch_time"] <= df["fetch_time"].unique()[0] - covert to datetime
                pass

        else:
            print(f"Parent Id: {pid} does not match file name: {ff}, not uploading to db\nwill remove file")
            os.remove(os.path.join(data_dir, ff))

    # -----
    # bad tickers
    # -----

    bad_tickers = [pd.read_csv(os.path.join(data_dir, i))
                   for i in os.listdir(data_dir)
                   if re.search("^bad_ticker", i)]
    bad_tickers = pd.concat(bad_tickers)

    bad_tickers = bad_tickers[["Company Name", "Identifier"]].drop_duplicates()
    # - reason for previous bad_tickers is unknown
    bad_tickers["reason"] = "unknown"

    for bt in bad_tickers.to_dict("records"):
        # check if bad ticker is already in collection
        res = db[page_info + "_bad_ticker"].find_one(filter=bt)
        if res is None:
            print(f"adding bad_ticker: {json.dumps(bt, indent=4)}")
            db[page_info + "_bad_ticker"].insert_one(bt)

    # ---
    # read back in as dataframe
    # ---

    # example: read it all back - by setting filter to empty all will be satisfied
    # vc = pd.DataFrame(list(db[page_info].find(filter={})))
    # vc.drop("_id", axis=1, inplace=True)

    # bdt = pd.DataFrame(list(db[page_info + "_bad_ticker"].find(filter={})))
    # bdt.drop("_id", axis=1, inplace=True)
