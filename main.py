import copy
import os
import pickle
import sys
from itertools import chain
from subprocess import call

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication, QTimer, pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

prognosis = ["1. Get a comprehensive dilated eye exam once a year.\n2. Do exercise to maintain health\n3. Do not smoke\n4. Maintain normal blood pressure",
             "1. Get a comprehensive dilated eye exam in 2 to 3 months.\n2. Control Blood Sugar levels.\n3. Maintain normal blood pressure\n4. Do exercise to maintain health.\n5. 95% chances of vision Restoration.",
             "1. This stage is curable, 90% vision restoration possible.\n2. Regular screening in 20-30 days.\n3. Maintain blood Sugar levels & cholestrol.\n 4. Stop smoking if you are.\n 5. Consult doctor ASAP",
             "1. Patient has abnormal growth of new blood vessels & scar tissue, bleeding in form of clear, jelly-like substance i.e. viterous humour.\n2. 60%-70% chances of vision restoration.\n3. Regular screening 5-10 days.\n4. Control blood Pressure and Sugar levels",
             "1. Patient has abnormal growth of new blood vessels, scar tissue and some degree of deatchment of retina, leakages of fluid from blood vessels.\n2. Regular eye screening 10-15 days.\n3. Control blood Pressure and Sugar levels.\n4. In severe cases laser treatment for restoring eye vision."
             ]

DISEASE_CLASSES = {
                      0: 'Normal',
                      1: 'Mild NPDR',
                      2: 'Moderate NPDR',
                      3: 'Severe NPDR',
                      4: 'Proliferative DR'
                    }


class DiabeticRetinopathyMobilenet(QDialog):
    def __init__(self):
        super(DiabeticRetinopathyMobilenet, self).__init__()
        loadUi('gui.ui', self)
        self.TrainButton.clicked.connect(self.trainingClicked)
        self.BrowseButton.clicked.connect(self.browseClicked)
        self.DetectRecogniseButton.clicked.connect(self.detectRecogniseClicked)
        self.ExitButton.clicked.connect(self.exitClicked)

        self.font = cv2.FONT_HERSHEY_PLAIN
        self.numPoints = 24
        self.radius = 8

    @pyqtSlot()
    def trainingClicked(self):
        call(["python", "model_training.py"])

    @pyqtSlot()
    def browseClicked(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', '.\\', "Image Files (*.*)")
        if fname:
            self.LoadImageFunction(fname)
        else:
            print("Invalid Image")

    def LoadImageFunction(self, fname):
        self.image = cv2.imread(fname)
        self.DisplayImage(self.image, 1)

    def DisplayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImg = outImg.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImg))
            self.imgLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)

    def DetectRetinopathy(self, img):

        model = self.Load_Training_Model()

        # pre-processing the image for classification
        image = cv2.resize(img, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)

        print(predictions)

        final_prediction = predictions.argmax(axis=1)[0]

        print(final_prediction)

        disease_name = DISEASE_CLASSES[final_prediction]

        print(disease_name)

        print("Disease Detected is: ", disease_name)
        self.label.setText(disease_name)
        self.Show_Bar_Chart(predictions)
        self.draw_barchart(final_prediction)
        

    # Loading our trained model
    def Load_Training_Model(self):
        
        mobile = keras.applications.mobilenet.MobileNet()
        x = mobile.layers[-6].output
        x = Dropout(0.25)(x)
        predictions = Dense(5, activation='softmax')(x)
        model = Model(inputs=mobile.input, outputs=predictions)

        for layer in model.layers[:-23]:
            layer.trainable = False

        model.load_weights('model.h5')

        return model

    def draw_barchart(self, index):
        xdata = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
        ydata = [[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]
                 ]
        barlabels = [0, 1, 2, 3, 4]

        x = np.arange(len(barlabels)) 
        width = 0.35 
        fig, ax = plt.subplots()
        rects1 = ax.bar(x, ydata[index], width, label='Disease Detected', color=['black', 'red', 'green', 'blue', 'cyan'])

        ax.set_ylabel('Scores')
        ax.set_title('Diabetic Retinopathy Detection')
        ax.set_xticks(x)
        ax.set_xticklabels(xdata)
        ax.legend()

        fig.tight_layout()
        #plt.savefig("barplot.png")

        barimage = cv2.imread("barplot_probabilities.png")
        self.DisplayBarChart(barimage, 1)
        QMessageBox.information(self, 'Details', prognosis[index], QMessageBox.Ok,
                                QMessageBox.Ok)
        self.plainTextEdit.document().setPlainText(prognosis[index])

    def DisplayBarChart(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImg = outImg.rgbSwapped()

        if window == 1:
            self.barimgLabel.setPixmap(QPixmap.fromImage(outImg))
            self.barimgLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.barimgLabel.setScaledContents(True)
            
    def Show_Bar_Chart(self, predictions_list):
        
        objects = ('Normal', 'Mild','Moderate', 'Severe', 'PDR')
        y_pos = np.arange(len(objects))
        
        flatten_predictions_list = list(chain.from_iterable(predictions_list)) 
        flatten_predictions_percentages = map(lambda x: round(x * 100, 2), flatten_predictions_list)    
            
        performance = list(flatten_predictions_percentages)
        print(performance)
        
        plt.bar(y_pos, performance, align='center', color=['green', 'blue', 'yellow', 'red', 'black'])
        
        for index,data in enumerate(performance):
            plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
            
        plt.xticks(y_pos, objects)
        plt.ylabel('Percentage Probabilities')
        plt.title('Disease Predictions')
        plt.savefig("barplot_probabilities.png")
    

    @pyqtSlot()
    def detectRecogniseClicked(self):
        self.DetectRetinopathy(self.image)

    @pyqtSlot()
    def exitClicked(self):
        QApplication.instance().quit()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = DiabeticRetinopathyMobilenet()
    window.setWindowTitle('Diabetic Retinopathy Detection using Mobilenet CNN')
    window.show()
    sys.exit(app.exec_())

