import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import time


class StockEvaluator():
    def __init__(self, callback_thread=None):
        # test123
        #TODO(Paul): 改成不是goodinfo的網站
        #TODO(Paul): 6770力積電 800張大戶bug
        self.headers = {
            "user-agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36",
            "referer": "https://goodinfo.tw/StockInfo/EquityDistributionClassHis.asp?STOCK_ID={}"
        }
        self.url = "https://goodinfo.tw/StockInfo/ShowBuySaleChart.asp?STOCK_ID={}&CHT_CAT=DATE"
        self.large_trader_url = "https://goodinfo.tw/StockInfo/EquityDistributionClassHis.asp?STOCK_ID={}&CHT_CAT=WEEK"
        self.qfii_ratio_url = "https://goodinfo.tw/StockInfo/ShowK_Chart.asp?STOCK_ID={}&CHT_CAT2=DATE"
        self.my_data = {
            "STOCK_ID": "{}",
            "CHT_CAT": "WEEK",
            "STEP": "DATA",
            "DISPLAY_CAT": "PEOPLE_NO"
        }
        self.qfii_ratio_headers = {
            "user-agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36",
            "referer": "https://goodinfo.tw/StockInfo/ShowK_Chart.asp?STOCK_ID=2923&CHT_CAT2=DATE"
        }
        self.qfii_ratio_data = {
            "STOCK_ID": "{}",
            "CHT_CAT2": "DATE"
        }
        self.scr_url = "https://norway.twsthr.info/StockHolders.aspx?stock={}"
        self.scr_headers = {
            "user-agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36",
            "referer": "https://www.google.com/"
        }
        self.info = {}
        self.callback_thread = callback_thread

    def _parse_html(self, stockNo):
        r = requests.get(self.url.format(stockNo), headers=self.headers)
        r.encoding = 'utf-8'
        
        if "您的瀏覽量異常" in r.text:
            raise RuntimeError
        elif "查無法人買賣相關資料!!" in r.text:
            raise ValueError
        else:
            soup = BeautifulSoup(r.text, 'lxml')
        
        self.stockNo = int(stockNo)

        # find stock name
        self.stock_name = soup.find_all("a", href=f"StockDetail.asp?STOCK_ID={stockNo}")[1].string.split("\xa0")[1]

        if self.callback_thread:
            self.callback_thread.callbackSignal.emit("抓取【{}】籌碼資訊".format(self.stock_name))
            self.callback_thread.callbackSignal.emit("抓取三大法人買賣資料...")
        else:
            print("抓取【{}】籌碼資訊".format(self.stock_name))
            print("抓取三大法人買賣資料...")

        # find qfii table
        data = soup.select_one("#divBuySaleDetailData")
        df = pd.read_html(data.prettify())
        df = df[2]

        col1 = df.columns.get_level_values(0).to_list()
        col2 = df.columns.get_level_values(1).to_list()
        merge = [i + j for i, j in zip(col1, col2)]
        merge = col1[:5] + merge[5:]
        df.columns = merge
        df.columns = df.columns.str.replace(" ", "")
        return df

    def get_info(self, stockNo):
        """
        date: 期別
        price: 成交
        qfii_net: 外資買賣超(張)
        qfii_net5: 外資買賣超(張) 5天淨值
        di_net: 投信買賣超(張)
        di_net5: 投信買賣超(張) 5天淨值
        """

        def _parse_date(df, first_index=0):
            try:
                date = df["期別"][first_index]
                year, month_day= date.split("'")
                month, day = month_day.split("/")
                date = "20" + year + "/" + month + "/" + day
                return date
            except:
                date = df["期別"][first_index]
                month, day = date.split("/")
                date = month + "/" +  day
                return date

        def _parse_qfii_net(df, first_index=0):
            net = df["外資買賣超(張)"][first_index]
            if isinstance(net, str) and "," in net:
                net = net.replace(",", "")
            net = float(net)
            return net

        def _parse_di_net(df, first_index=0):
            net = df["投信買賣超(張)"][first_index]
            if isinstance(net, str) and "," in net:
                net = net.replace(",", "")
            net = float(net)
            return net

        def _cal_qfii_net5(df):
            cal_list = []
            temp = df["外資買賣超(張)"].to_list()[:5]
            for i in temp:
                if isinstance(i, str) and "," in i:
                    i = i.replace(",", "")
                cal_list.append(i)
            cal_list = [float(i) for i in cal_list]
            return sum(cal_list)

        def _cal_di_net5(df):
            cal_list = []
            temp = df["投信買賣超(張)"].to_list()[:5]
            for i in temp:
                if isinstance(i, str) and "," in i:
                    i = i.replace(",", "")
                cal_list.append(i)
            cal_list = [float(i) for i in cal_list]
            return sum(cal_list)
            
        df = self._parse_html(stockNo)
        extract_column = ["期別", "成交", "外資買賣超(張)", "投信買賣超(張)"]
        df = df[extract_column]
        df = df.dropna(axis=0, how='any')


        self.info["股票代碼"] = self.stockNo
        self.info["股票名稱"] = self.stock_name
        try:
            self.info["日期"] = _parse_date(df)
        except:
            self.info["日期"] = _parse_date(df, first_index=1)
        try:
            self.info["成交價"] = df["成交"][0]
        except:
            self.info["成交價"] = df["成交"][1]
        try: 
            self.info["外資買超"] = _parse_qfii_net(df)
        except:
            self.info["外資買超"] = _parse_qfii_net(df, first_index=1)
        self.info["外資近5日買超"] = _cal_qfii_net5(df)
        try: 
            self.info["投信買超"] = _parse_di_net(df)
        except:
            self.info["投信買超"] = _parse_di_net(df, first_index=1)
        self.info["投信近5日買超"] = _cal_di_net5(df)

    def _parse_scr_html(self, stockNo):
        r = requests.get(self.scr_url.format(str(stockNo)), headers=self.scr_headers)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'lxml')
        data = soup.select_one("#D1")
        df = pd.read_html(data.prettify())
        df = df[1]
        df.columns = df.iloc[0]
        df.columns = df.columns.str.replace(" ", "")
        df = df.drop([0])
        df = df.dropna(axis=0, how="all")
        df = df.reset_index(drop=True)
        return df

    def get_scr_info(self, stockNo):
        def _is_published(df):
            #TODO(Paul): nan bug
            pass
        
        if self.callback_thread:
            self.callback_thread.callbackSignal.emit("抓取籌碼集中度資料...")
        else:
            print("抓取籌碼集中度資料...")
            
        df = self._parse_scr_html(stockNo)
        latest = 0
        lastest_week = float(df["平均張數/人"][latest])
        previous_week = float(df["平均張數/人"][latest+1])
        previous_week_20 = float(df["平均張數/人"][latest+4])

        self.info["籌碼集中度增加(周)"] = True if lastest_week > previous_week else False
        self.info["20日籌碼集中度增加"] = True if lastest_week > previous_week_20 else False

    def _parse_large_trader_html(self, r):
        if "您的瀏覽量異常" in r.text:
            raise RuntimeError
        elif "查無法人買賣相關資料!!" in r.text:
            raise ValueError
        else:
            soup = BeautifulSoup(r.text, 'lxml')
        
        data = soup.select_one("#divEquityDistributionClassHis")
        df = pd.read_html(data.prettify())
        df = df[2]
        col = df.columns.get_level_values(1).to_list()

        df.columns = col
        df.columns = df.columns.str.replace(" ", "")
        return df
    
    def get_large_trader_info(self, stockNo):
        def _is_published(latest):
            observed = ["＞800張≦1千張", "＞1千張"]
            for _, key in enumerate(observed):
                temp = df[key][latest]
                if temp == "-":
                    return False
            return True

        if self.callback_thread:
            self.callback_thread.callbackSignal.emit("抓取800張大戶持股比例資料...")
        else:
            print("抓取800張大戶持股比例資料...")

        r = requests.get(self.large_trader_url.format(str(self.stockNo)), headers=self.headers)
        r.encoding = 'utf-8'
        df = self._parse_large_trader_html(r)

        latest = 0
        while True:
            if _is_published(latest):
                break
            else:
                latest += 1
        month = latest + 4

        large_trader = float(df["＞800張≦1千張"][latest]) + float(df["＞1千張"][latest])
        large_trader_month = float(df["＞800張≦1千張"][month]) + float(df["＞1千張"][month])
        self.info["800張大戶"] = large_trader
        self.info["800大戶增加(月)"] = bool(np.maximum(large_trader - large_trader_month, 0))

    def crawl(self, stockNo):
        self.get_info(stockNo)
        self.get_scr_info(stockNo)
        self.get_large_trader_info(stockNo)
        return self.info

    # def crawl(self, stockNo):
    #     # pool = mp.Pool(4)
    #     #TODO(Paul): multiprocessing
    #     with mp.Pool(processes=4) as pool:
    #         pool.apply_async(self.get_info, args=(stockNo,))
    #         pool.apply_async(self.get_scr_info, args=(stockNo,))
    #         pool.apply_async(self.get_large_trader_info, args=(stockNo,))
    #     # pool.close()
    #     # pool.join()
    #     return self.info

    def evaluate(self, stockNo):
        self.crawl(stockNo)

        if self.callback_thread:
            self.callback_thread.callbackSignal.emit("分析數據...")

        score = int(self.info["800大戶增加(月)"]) * 1 + \
                int(self.info["籌碼集中度增加(周)"]) * 1 + \
                int(self.info["20日籌碼集中度增加"]) *  1+ \
                int(bool(np.maximum(self.info["外資買超"], 0))) * 0.5 + \
                int(bool(np.maximum(self.info["外資近5日買超"], 0))) * 0.5 + \
                int(bool(np.maximum(self.info["投信買超"], 0))) * 0.5 + \
                int(bool(np.maximum(self.info["投信近5日買超"], 0))) * 0.5

        return self.info, score
        # return pd.DataFrame([self.info]), score

def main():
    print("郭菲特的籌碼分析小工具v1.0")
    while True:
        try:
            stockNo = input("請輸入股票代碼 （輸入quit退出）: ")
            if stockNo == "quit":
                break
            elif not isinstance(int(stockNo), int):
                print("請輸入代碼數字")
                continue

            stockEvaluator = StockEvaluator()
            info, score = stockEvaluator.evaluate(int(stockNo))
            print("======================籌碼分析======================")
            for key, value in info.items():
                print("{}: {}".format(key, value))
            print("\n")
            print("籌碼評比: {} 顆星".format(score))
            print("===================================================")

            print("等待15秒...")
            time.sleep(15)
            pass

        except Exception as e:
            print(str(e))


if __name__ == '__main__':
    main()
