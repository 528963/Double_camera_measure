import sys
import time

from PyQt5.Qt import QApplication, QWidget, QThread, QMainWindow
from window.Main import Ui_MainWindow

class MyThread(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        for i in range(10):
            print("执行...%d" % (i + 1))
            time.sleep(1)

class MyWin(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    def init_ui(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

    def start_thread(self):
        self.my_thread = MyThread()
        self.my_thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myshow = MyWin()
    myshow.show()
    app.exec_()