# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/31 8:40 下午
@Auth ： Robbie Deng
@File ：main.py
"""
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from show import Ui_MainWindow

if __name__ == "__main__":
    # Qt Application
    app = QApplication(sys.argv)

    MainWindow = QMainWindow()

    window = Ui_MainWindow()
    window.setupUi(MainWindow)
    window.retranslateUi(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())


