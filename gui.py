# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'newgui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1252, 682)
        Dialog.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(181, 238, 255, 255), stop:1 rgba(255, 255, 255, 255))")
        self.imgLabel = QtWidgets.QLabel(Dialog)
        self.imgLabel.setGeometry(QtCore.QRect(10, 10, 621, 661))
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(1069, 5, 171, 291))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.TrainButton = QtWidgets.QPushButton(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        self.TrainButton.setFont(font)
        self.TrainButton.setStyleSheet("background-color: rgb(103, 103, 103);\n"
"color: rgb(255, 255, 255);\n"
"border-radius:5px;\n"
"padding:5px;")
        self.TrainButton.setObjectName("TrainButton")
        self.verticalLayout.addWidget(self.TrainButton)
        self.BrowseButton = QtWidgets.QPushButton(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        self.BrowseButton.setFont(font)
        self.BrowseButton.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.BrowseButton.setStyleSheet("background-color: rgb(103, 103, 103);\n"
"color: rgb(255, 255, 255);\n"
"border-radius:5px;\n"
"padding:5px;")
        self.BrowseButton.setObjectName("BrowseButton")
        self.verticalLayout.addWidget(self.BrowseButton)
        self.DetectRecogniseButton = QtWidgets.QPushButton(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        self.DetectRecogniseButton.setFont(font)
        self.DetectRecogniseButton.setStyleSheet("background-color: rgb(103, 103, 103);\n"
"color: rgb(255, 255, 255);\n"
"border-radius:5px;\n"
"padding:5px;")
        self.DetectRecogniseButton.setObjectName("DetectRecogniseButton")
        self.verticalLayout.addWidget(self.DetectRecogniseButton)
        self.ExitButton = QtWidgets.QPushButton(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(11)
        self.ExitButton.setFont(font)
        self.ExitButton.setStyleSheet("background-color: rgb(103, 103, 103);\n"
"color: rgb(255, 255, 255);\n"
"border-radius:5px;\n"
"padding:5px;")
        self.ExitButton.setObjectName("ExitButton")
        self.verticalLayout.addWidget(self.ExitButton)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(1070, 310, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(640, 370, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        font.setItalic(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.barimgLabel = QtWidgets.QLabel(Dialog)
        self.barimgLabel.setGeometry(QtCore.QRect(640, 10, 421, 351))
        self.barimgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.barimgLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.barimgLabel.setText("")
        self.barimgLabel.setObjectName("barimgLabel")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(Dialog)
        self.plainTextEdit.setGeometry(QtCore.QRect(640, 400, 601, 271))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setObjectName("plainTextEdit")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Diabetic Retinopathy Detection using machine Learning"))
        self.groupBox.setTitle(_translate("Dialog", "Process"))
        self.TrainButton.setText(_translate("Dialog", "Training"))
        self.BrowseButton.setText(_translate("Dialog", "Browse Test Image"))
        self.DetectRecogniseButton.setText(_translate("Dialog", " Recognise"))
        self.ExitButton.setText(_translate("Dialog", "Exit"))
        self.label_2.setText(_translate("Dialog", "Prognosis & Medication"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

