# -*- coding: utf-8 -*-
# coding: utf-8
import requests
from io import StringIO
import pandas as pd
import numpy as np
import datetime
import json
import time
import os
from argparse import ArgumentParser

# 三大法人買賣金額

class QFIICrawler:
    def __init__(self):
        self.wrkdir = "./三大法人買賣"

    def crawl_data(self, url):
        r = self._connect(url)
        df = pd.read_csv(StringIO(r.text.replace("=", "")),
                        header=[1],
                        converters={
                            u'證券代號': str,
                            u'證券名稱': str
                        })
        df = self._transform_data(df)
        print(df.head())
        return df

    def filter(self, df, column):
        df = df[pd.to_numeric(df[column], errors='coerce') > 0]
        return df

    def _transform_data(self, df):
        df["證券代號"] = df["證券代號"].str.replace(' ', '')
        df["證券名稱"] = df["證券名稱"].str.replace(' ', '')
        for column in df.keys()[2:]:
            df[str(column)] = df.apply(lambda s: pd.to_numeric(
                df[str(column)].astype(str).str.replace(",", ""), errors='coerce'))
        return df

    def _connect(self, url):
        r = requests.get(url)
        if r.status_code == requests.codes.ok:
            print("Connect OK")
        return r

    def load_series_data(self, n_days, date=datetime.datetime.now(), allow_continuous_fail_count=5, save=False):

        fail_count = 0
        dataset = {}

        while len(dataset) < n_days:

            print('parsing', date)
            try:
                date_str = date.strftime('%Y%m%d')
                url = "https://www.twse.com.tw/fund/T86?response=csv&date={}&selectType=ALL".format(date_str)
                data = self.crawl_data(url)
                dataset[date.date()] = data
                if save:
                    if not os.path.exists(self.wrkdir):
                        os.makedirs(self.wrkdir)
                    file_name = "{}_三大法人買賣金額統計表.csv".format(date_str)
                    data.to_csv(os.path.join(self.wrkdir, file_name), encoding='utf_8_sig', index=False)

                fail_count = 0
                print('success!')

            except:
                print('fail! check the date is holiday')
                fail_count += 1
                if fail_count == allow_continuous_fail_count:
                    raise
                    break
            
            date -= datetime.timedelta(days=1)
            time.sleep(10)
        return dataset

def get_date(date=None):
    """
    Args:
    date number (int): 20210131

    Return string
    """
    if date is None:
        weeks = [1, 2, 3, 4, 5]
        dt = datetime.date.today()
        current_day = dt.isoweekday()

        if not current_day in weeks:
            dt = dt - datetime.timedelta(days=current_day - 5)

        return dt
    else:
        return str(date)


def main():

    date = get_date()
    save_path = r"E:\stock\test.csv"
    date_str = date.strftime('%Y%m%d')
    savepath = "./{}_三大法人買賣金額統計表.csv".format(date_str)
    url = "https://www.twse.com.tw/fund/T86?response=csv&date={}&selectType=ALL".format(date_str)

    crawler = QFIICrawler()
    # crawler.load_series_data(5, save=True)
    data = crawler.crawl_data(url)
    # data.to_csv(save_path, encoding='utf_8_sig', index=False)
    pass


if __name__ == "__main__":
    main()
