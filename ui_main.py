# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import Ui_stock_evaluator_ui as ui
from stock_evaluator import StockEvaluator
import numpy as np
from QCandyUi import CandyWindow
from thread import CallBackThread, WaitingThread

class Main(QMainWindow, ui.Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.callback = CallBackThread()
        self.waiting = WaitingThread()
        self.lineEdit.returnPressed.connect(self.evaluate)
        self.pushButton.clicked.connect(self.evaluate)
        # self.setWindowIcon(QtGui.QIcon(r"./refs/window_icon.jpg"))
        self.scoreLabel.setStyleSheet("color:rgb({},{},{})".format(255, 0, 0))
        # self.diLabel.setStyleSheet("color:rgb({},{},{},255)".format(255, 0, 0))
        self.execute_callback()


    def evaluate(self):
        try:
            self.clear_all()
            self.printf("Start evaluating...")
            stockNo = int(self.lineEdit.text())
            self.evaluator = StockEvaluator(self.callback)
            self.info, self.score = self.evaluator.evaluate(stockNo)
            self.display_img()
            self.display_info()
            # self.wait()
            
        except Exception as e:
            self.printf(str(e))

    def _transform(self, data):
        if isinstance(data, bool):
            return "V" if data == True else ""
        else:
            return "V" if bool(np.maximum(data, 0)) else ""

    def clear_all(self):
        self.stockNameLabel.clear()
        self.dateLabel.clear()
        self.priceLabel.clear()
        self.qfiiLabel.clear()
        self.qfiiLabel5.clear()
        self.diLabel.clear()
        self.diLabel5.clear()
        self.scrLabel.clear()
        self.scrLabel20.clear()
        self.largeTraderLabel.clear()
        self.scoreLabel.clear()
        self.imgLabel.setStyleSheet("")
        self.textBrowser.clear()

    def display_img(self):
        if self.score >= 4.0:
            self.pixmap = QPixmap("./refs/good.jpg")
        elif self.score >= 3.0 and self.score < 4.0:
            self.pixmap = QPixmap("./refs/ok.jpg")
        else:
            self.pixmap = QPixmap("./refs/bad.jpg")

        self.imgLabel.setPixmap(self.pixmap)
        self.imgLabel.setScaledContents(True)
        
    def printf(self, mes):
        self.textBrowser.append(mes)
        self.cursot = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursot.End)
        QtWidgets.QApplication.processEvents()

    def execute_callback(self):
        self.callback.start()
        self.callback.callbackSignal.connect(self.callback_display)

    def callback_display(self, strs):
        self.printf(strs)

    def wait(self):
        #TODO(Paul): bug
        self.printf("等待15秒，避免過度頻繁查詢...")
        self.pushButton.setEnabled(False)
        self.waiting.start()
        self.waiting.join()
        self.printf("完成")
        self.pushButton.setEnabled(True)

    def display_info(self):
        #TODO(Paul): bug 連查3次會出現重複
        self.scoreLabel.setText("{} 顆星".format(str(self.score)))
        self.stockNameLabel.setText(self.info["股票名稱"])
        self.dateLabel.setText(self.info["日期"])
        self.priceLabel.setText(str(self.info["成交價"]))
        self.qfiiLabel.setText(self._transform(self.info["外資買超"]))
        self.qfiiLabel5.setText(self._transform(self.info["外資近5日買超"]))
        self.diLabel.setText(self._transform(self.info["投信買超"]))
        self.diLabel5.setText(self._transform(self.info["投信近5日買超"]))
        self.scrLabel.setText(self._transform(self.info["籌碼集中度增加(周)"]))
        self.scrLabel20.setText(self._transform(self.info["20日籌碼集中度增加"]))
        self.largeTraderLabel.setText(self._transform(self.info["800大戶增加(月)"]))

        self.printf("=================詳細資訊=================")
        for key, value in self.info.items():
            if value == True:
                value = "是"
            elif value == False:
                value = "否"

            self.printf("{}: {}".format(key, value))
        self.printf("=========================================")
        self.printf("Success!")


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window = CandyWindow.createWindow(window, 'blueDeep', title="Stock Evaluator", ico_path=r"./refs/window_icon.jpg")
    window.show()
    sys.exit(app.exec_())