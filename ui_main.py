# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
import Ui_stock_evaluator_ui as ui
from stock_evaluator import StockEvaluator
import numpy as np
from QCandyUi import CandyWindow
from thread import CallBackThread
import images


class Main(QMainWindow, ui.Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.callback = CallBackThread()
        self.lineEdit.returnPressed.connect(self.evaluate)
        self.pushButton.clicked.connect(self.evaluate)
        self.scoreLabel.setStyleSheet("color: red")
        self.qfiiLabel.setStyleSheet("color: red")
        self.qfiiLabel5.setStyleSheet("color: red")
        self.diLabel.setStyleSheet("color: red")
        self.diLabel5.setStyleSheet("color: red")
        self.scrLabel.setStyleSheet("color: red")
        self.scrLabel20.setStyleSheet("color: red")
        self.largeTraderLabel.setStyleSheet("color: red")
        self.execute_callback()

    def evaluate(self):
        try:
            self.pushButton.setEnabled(False)
            self.lineEdit.setEnabled(False)
            self.clear_all()
            self.printf("開始分析...")
            stockNo = self.lineEdit.text()
            self.evaluator = StockEvaluator(self.callback)
            self.info, self.score = self.evaluator.evaluate(stockNo)
            self.display_img()
            self.display_info()
            self.wait()

        except ValueError:
            self.printf("查無股票代碼:{}的相關資料，請輸入正確股票代碼。".format(stockNo))
            self.pushButton.setEnabled(True)
            self.lineEdit.setEnabled(True)
        except RuntimeError:
            self.printf("您的瀏覽量異常, 已影響網站速度, 目前暫時關閉服務, 請稍後再重新使用並調降程式查詢頻率, 以維護一般使用者的權益。")
            self.pushButton.setEnabled(True)
            self.lineEdit.setEnabled(True)
        except Exception as e:
            self.printf(str(e))
            self.pushButton.setEnabled(True)
            self.lineEdit.setEnabled(True)

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
        self.imgLabel.setPixmap(QPixmap(""))
        self.textBrowser.clear()

    def display_img(self):
        if self.score >= 4.0:
            self.pixmap = QPixmap(":refs/good.jpg")
        elif self.score >= 3.0 and self.score < 4.0:
            self.pixmap = QPixmap(":refs/ok.jpg")
        else:
            self.pixmap = QPixmap(":refs/bad.jpg")

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
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerTick) 
        self.step = 15
        self.timer.start(1000) 
        self.printf("\n請等待15秒，避免頻繁查詢...")

    def timerTick(self):
        self.step -= 1 
        if self.step <= 0:
            self.timer.stop()
            self.printf("等待完畢，可繼續查詢")
            self.pushButton.setEnabled(True)
            self.lineEdit.setEnabled(True)

    def display_info(self):
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
        self.printf("分析完畢!")


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window = CandyWindow.createWindow(window, 'blueDeep', title="Stock Evaluator", ico_path=":refs/window_icon.jpg")
    window.show()
    sys.exit(app.exec_())