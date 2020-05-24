# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'show.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QProgressBar
from PyQt5.QtCore import Qt, QBasicTimer
import webbrowser

import load
import isone
import usa


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 450)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(130, 30, 180, 31))
        self.comboBox.setInputMethodHints(QtCore.Qt.ImhNone)
        self.comboBox.setEditable(False)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        # self.comboBox.activated.connect(self.choose)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 40, 71, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 100, 60, 16))
        self.label_2.setObjectName("label_2")
        self.label_mae = QtWidgets.QLabel(self.centralwidget)
        self.label_mae.setGeometry(QtCore.QRect(40, 160, 60, 16))
        self.label_mae.setObjectName("label_mae")
        self.mae_value = QtWidgets.QLabel(self.centralwidget)
        self.mae_value.setGeometry(QtCore.QRect(150, 160, 60, 16))
        self.mae_value.setText("")
        self.mae_value.setObjectName("mae_value")
        self.label_rmse = QtWidgets.QLabel(self.centralwidget)
        self.label_rmse.setGeometry(QtCore.QRect(40, 210, 60, 16))
        self.label_rmse.setObjectName("label_rmse")
        self.rmse_value = QtWidgets.QLabel(self.centralwidget)
        self.rmse_value.setGeometry(QtCore.QRect(150, 210, 60, 16))
        self.rmse_value.setText("")
        self.rmse_value.setObjectName("rmse_value")
        self.label_mape = QtWidgets.QLabel(self.centralwidget)
        self.label_mape.setGeometry(QtCore.QRect(40, 260, 60, 16))
        self.label_mape.setObjectName("label_mape")
        self.mape_value = QtWidgets.QLabel(self.centralwidget)
        self.mape_value.setGeometry(QtCore.QRect(150, 260, 60, 16))
        self.mape_value.setText("")
        self.mape_value.setObjectName("mape_value")
        self.select_data = QtWidgets.QPushButton(self.centralwidget)
        self.select_data.setGeometry(QtCore.QRect(340, 30, 113, 31))
        self.select_data.setObjectName("select_data")
        self.select_data.clicked.connect(self.OPenCLick)
        self.progressbar = QProgressBar(self.centralwidget)
        self.progressbar.setGeometry(QtCore.QRect(30, 300, 200, 30))
        self.progressbar.setObjectName("progressbar")
        self.progressbar.setRange(0, 100)
        self.progressbar.setValue(0)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "智能电网用电预测系统"))
        self.comboBox.setItemText(0, _translate("MainWindow", "--数据集--"))
        self.comboBox.setItemText(1, _translate("MainWindow", "英国电力数据集"))
        self.comboBox.setItemText(2, _translate("MainWindow", "北美公共事业数据集"))
        self.comboBox.setItemText(3, _translate("MainWindow", "加拿大家庭用电数据集"))
        self.label.setText(_translate("MainWindow", "选择数据集"))
        self.label_2.setText(_translate("MainWindow", "评价指标"))
        self.label_mae.setText(_translate("MainWindow", "MAE"))
        self.label_rmse.setText(_translate("MainWindow", "RMSE"))
        self.label_mape.setText(_translate("MainWindow", "MAPE"))
        self.select_data.setText(_translate("MainWindow", "确认"))


    def OPenCLick(self):
        if self.comboBox.currentIndex() == 3:
            self.progressbar.setValue(0)
            self.mape_value.setText("")
            self.rmse_value.setText("")
            self.mae_value.setText("")
            QApplication.processEvents()
            a = load.AMPDS(self.progressbar, self.statusbar)
            self.mape_value.setText(a[0])
            self.rmse_value.setText(a[1])
            self.mae_value.setText(a[2])
            webbrowser.open("file:///Users/dengyiran/PycharmProjects/data-test/load_forecast_Ampds.html", new=1)
            QApplication.processEvents()
        if self.comboBox.currentIndex() == 2:
            self.progressbar.setValue(0)
            self.mape_value.setText("")
            self.rmse_value.setText("")
            self.mae_value.setText("")
            QApplication.processEvents()
            b = usa.usa(self.progressbar, self.statusbar)
            self.mape_value.setText(b[0])
            self.rmse_value.setText(b[1])
            self.mae_value.setText(b[2])
            webbrowser.open("file:///Users/dengyiran/PycharmProjects/data-test/load_forecast_usa.html", new=1)
            QApplication.processEvents()
        if self.comboBox.currentIndex() == 1:
            self.progressbar.setValue(0)
            self.mape_value.setText("")
            self.rmse_value.setText("")
            self.mae_value.setText("")
            QApplication.processEvents()
            c = isone.isone(self.progressbar, self.statusbar)
            self.mape_value.setText(c[0])
            self.rmse_value.setText(c[1])
            self.mae_value.setText(c[2])
            webbrowser.open("file:///Users/dengyiran/PycharmProjects/data-test/load_forecast_isone.html", new=1)
            QApplication.processEvents()




