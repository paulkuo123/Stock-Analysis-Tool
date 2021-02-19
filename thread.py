from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
import time

class CallBackThread(QThread):
    callbackSignal = QtCore.pyqtSignal(str)
    def __init__(self):
        super(CallBackThread, self).__init__()

    def run(self):
        # self.callbackSignal.emit(str())
        pass
