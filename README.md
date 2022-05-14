# 股票籌碼分析小工具

## 簡介
此 GUI 應用程式主要利用 PyQt5 和網絡爬蟲技術進行台股股票分析及評估。

針對每日外資買賣超、投信買賣超、籌碼集中度、800張大戶分析股票籌碼並給予當日評級。

## 數據爬取來源
* [Goodinfo!台灣股市資訊網](https://goodinfo.tw/tw/index.asp)
* [股權分散表 - 神秘金字塔](https://norway.twsthr.info/StockHolders.aspx)

## 環境搭建

* 新建 Anaconda 環境
```
conda create -n test python=3.8 -y
conda activate test
```

* 安裝套件
```
conda install -c anaconda pyqt
pip install requests
pip install pandas
pip install BeautifulSoup4
pip install lxml
pip install html5lib 
pip install QCandyUi
conda install pywin32
```


## 使用方法

* 下載此 Git 存放庫
```
git clone https://github.com/paulkuo123/Stock-Evaluator.git
```
* 切換至 Stock-Analysis-Tool 資料夾
```
cd Stock-Analysis-Tool
```
* 執行 main.py
```
python main.py
```

## Demo

![image](https://github.com/paulkuo123/Stock-Evaluator/blob/master/demo.gif)
