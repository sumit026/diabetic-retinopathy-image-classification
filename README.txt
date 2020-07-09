Diabetic Retinopathy Image Classification by using Convolutional Neural Network
---------------------------------------------------------------------------------------

Tools Requirements:

1. anaconda3-5.2.0-windows-x86_64
2. Visual Studio Code Editor
3. Python 3.6.5
4. QT Designer

--------------------------------------------------------------------------------------------

Once anaconda tool install in the system, we need to install following below commands 

1. conda update setuptools
2. conda update wrapt
3. pip install tensorflow==2
4. pip install opencv-contrib-python
5. pip install msgpack
6. pip install keras

---------------------------------------------------------------------------------------------

Setup Visual Studio Code Editor
-------------------------------
For Visual Studio Code Editor, you need to use Python3.6.5 as code interpreter. You have to select and activate an environment. 
Follow below steps:-

Step 1: To select a specific environment go to in View toolbar and select Command Pallete and enter 'python: Select Inerpreter'. 
Step 2: Once you select interepreter, it will show you list of available global environments for python. 
Step 3: As your are using anaconda free and opensourse distribution of python, for this you have to select 'Python 3.6.5 32-bit ('base':conda)' as python interpreter.


Also, you have to enabled below extensions in Visual Studio Code:

1. Anaconda Extension Pack
2. Python for VSCode

--------------------------------------------------------------------------------------------

Dataset Requirement : -

Download the dataset from - https://www.kaggle.com/c/diabetic-retinopathy-detection/data

---------------------------------------------------------------------------------------------

Installation
------------
Once data is downloaded, extract train.zip.001 images to data/train and put the trainLabels.csv file into the data directory as well.


Usage
-----
Firstly, use convert.py script to resize the images. It is very important to resize the images into smaller size as original images are too heavy, smaller size of images will be helpful for your model when you will train. 

For using convert.py please follow below steps:--

1. Create new folder dataset, put convert.py script and data/train folder into it.
2. Open anaconda command prompt and go to your dataset directory.
3. Use below command for resizing images:

   python.exe .\convert.py --crop_size 512 --extension jpeg

   You can change the cropping size and image extension based on your requirement.

4. Once you run above command it will start creating resizing images from data/train to data/train_res directory.
5. Once the process is complete, delete the data/train folder and place the train_res directory into main direcotry 'dataset'.  
----------------------------------------------------------------------------------------------
6. Once above steps is successfully done, create new folder for your project and place 'dataset' directory into that and place below scripts as well.


Scripts:
--------
All below pythons scripts are meant to be executed in the following order.

1. preprocessing.py
2. model_training.py
3. main.py

Below files has been created using QT designer tool which will work with main.py file. 

1. gui.ui
2. gui.py

--------------------------------------------------------------------------------------------

QT designer tool creates the .ui format file, to work with gui file with your python code you need to convert that gui file into python code.

By below command you can convert your gui file into python code:-

pyuic5 -x gui.ui -o gui.py

Once you done with the convert gui file into python file, it will be ready to work with your maine file.

---------------------------------------------------------------------------------------------

Once you run the 'preprocessing.py' file it will start creating below folders from the dataset.

1. base_dir
2. aug_dir

----------------------------------------------------------------------------------------------

After preprocessing the dataset, you are ready to train the model based on 'base_dir' dataset by running 'model_training.py'.

Note: Model file is included in this repository. But, you can train the model based on your criteria.

----------------------------------------------------------------------------------------------

Run the main.py file And Enjoy :)

