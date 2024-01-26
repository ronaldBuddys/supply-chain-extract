# use pyautogui to find image on page

import os
import sys
import re
import pyautogui
import pandas as pd
from PIL import Image

try:
    # python package (supply_chain_extract) location - two levels up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # add package to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None

from supply_chain_extract import get_image_path

if __name__ == "__main__":

    download_buttons = [i for i in os.listdir(get_image_path()) if re.search("^download_excel", i)]

    btn1 = 'download_excel3.png'
    print(f"using: {btn1}\nas reference image")

    # show an image
    image = Image.open(get_image_path(btn1))
    image.show()

    # find display image

    res = []
    # different confidence level
    conf = [1.0, 0.9, 0.8, 0.6, 0.7, 0.5]
    for c in conf:
        # res[c] = {}
        for db in download_buttons:
            if db == btn1:
                continue
            loc = pyautogui.locateOnScreen(get_image_path(db), confidence=c)
            if loc is not None:
                loc_xy = pyautogui.center(loc)
                print(f"found using - image: {db} c: {c} at location: {loc_xy}")
                res += [(c, db, loc_xy.x, loc_xy.y)]

    df = pd.DataFrame(res, columns=["conf", "file", "x", "y"])

    conf_count = pd.pivot_table(df,
                                index="conf",
                                values="file",
                                aggfunc='count')
    print("-" * 50)
    print("images matched at different confidence levels")
    print(conf_count)
