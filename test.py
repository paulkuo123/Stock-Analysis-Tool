# -*- coding: utf-8 -*-
# coding: utf-8
import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import math
import time

stockNo = 2330
url = "https://sjmain.esunsec.com.tw/z/zc/zcl/zcl.djhtm?a={}&b=4"

headers = {
            "user-agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.146 Safari/537.36",
            "referer": "https://sjmain.esunsec.com.tw/z/zc/zcl/zcl_{}.djhtm"
        }
headers["referer"] = headers["referer"].format(str(stockNo))
r = requests.get(url.format(str(stockNo)), headers=headers)
r.encoding = 'utf-8'
soup = BeautifulSoup(r.text, 'lxml')
data = soup.select_one("#SysJustIFRAMEDIV")
df = pd.read_html(data.prettify())
df = df[1]
df.columns = df.iloc[0]
df.columns = df.columns.str.replace(" ", "")
df = df.drop([0])
df = df.dropna(axis=0, how="all")
df = df.reset_index(drop=True)
pass