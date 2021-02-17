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


class WaitingThread(QThread):  
  
    sec_changed_signal = pyqtSignal(int)
  
    def __init__(self, sec=15, parent=None):  
        super().__init__(parent)
        self.sec = sec
  
    def run(self):  
        # for i in range(self.sec):
        #     self.sec_changed_signal.emit(i)  #发射信号
            # time.sleep(1)
        time.sleep(self.sec)