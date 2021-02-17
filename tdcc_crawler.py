# -*- coding: utf-8 -*-
# coding: utf-8
import requests
from io import StringIO
import pandas as pd
import numpy as np
import datetime
import json
from argparse import ArgumentParser
from bs4 import BeautifulSoup
from decimal import Decimal, ROUND_HALF_UP

tdcc_url = 'https://www.tdcc.com.tw/smWeb/QryStockAjax.do' # 集保-大戶持股率

def get_current_csv():
    tdcc_url = 'https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5'
    data = pd.read_csv(tdcc_url)
    return data


def filter(data, number):
    fliter = (data["證券代號"] == str(number))
    data[filter]
    print(data[filter])


class TDCC_Crawler:
    def __init__(self):
        self.headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36",
            "Referer": "https://www.tdcc.com.tw/smWeb/QryStockAjax.do"
        }

    def create_post_data(self, date, stockNo):
        my_data = {
            'scaDates': str(date),
            'scaDate': str(date),
            'SqlMethod': 'StockNo',
            'StockNo': str(stockNo),
            'radioStockNo': str(stockNo),
            'StockName': '',
            'REQ_OPR': 'SELECT',
            'clkStockNo': str(stockNo),
            'clkStockName': ''
        }
        return my_data

    def get_dataset(self, date, stockNo):
        html = requests.post(tdcc_url,
                             data=self.create_post_data(date, stockNo),
                             headers=self.headers)
        if html.status_code == requests.codes.ok:
            print("Connect OK")
        dataset = self._parse_html(html)
        return dataset

    def get_800up_proportion(self, date, stockNo):
        dataset = self.get_dataset(date, stockNo)
        proportion = float(dataset["占集保庫存數比例(%)"][14]) + float(dataset["占集保庫存數比例(%)"][15])
        proportion = float(Decimal(str(proportion)).quantize(Decimal('.00'), ROUND_HALF_UP))
        return proportion

    def _parse_html(self, html):
        soup = BeautifulSoup(html.text, 'lxml')
        table = soup.find_all('table')[0]
        table = soup.select(".mt")[1]
        df = pd.read_html(str(table), encoding='utf-8')[0]
        df.iloc[0, :] = df.iloc[0, :].str.replace(' ', '')
        df = df.rename(columns=df.iloc[0, :])
        df = df.drop([0])
        return df


def main():
    date = "20210129"
    stockNo = "2330"

    crawler = TDCC_Crawler()
    df = crawler.get_dataset(date, stockNo)
    proportion = crawler.get_800up_proportion(date, stockNo)
    print(df)


if __name__ == '__main__':
    main()
