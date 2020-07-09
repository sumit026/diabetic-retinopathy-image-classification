import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import shutil


# creating file directory for the images
base_dir = 'base_dir'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# creating folders in the training and validation directory for each of the classes
Normal = os.path.join(train_dir, '0')
os.mkdir(Normal)
Mild_NPDR = os.path.join(train_dir, '1')
os.mkdir(Mild_NPDR)
Moderate_NPDR = os.path.join(train_dir, '2')
os.mkdir(Moderate_NPDR)
Severe_NPDR = os.path.join(train_dir, '3')
os.mkdir(Severe_NPDR)
PDR = os.path.join(train_dir, '4')
os.mkdir(PDR)

Normal = os.path.join(val_dir, '0')
os.mkdir(Normal)
Mild_NPDR = os.path.join(val_dir, '1')
os.mkdir(Mild_NPDR)
Moderate_NPDR = os.path.join(val_dir, '2')
os.mkdir(Moderate_NPDR)
Severe_NPDR = os.path.join(val_dir, '3')
os.mkdir(Severe_NPDR)
PDR = os.path.join(val_dir, '4')
os.mkdir(PDR)

df = pd.read_csv('./dataset/trainLabels.csv')

print(df.head())

# setting y as the labels
y = df['level']

# spliting the metadata into training and validation
df_train, df_val = train_test_split(df, test_size=0.1, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)

# finding the number of values in the training and validation set
df_train['level'].value_counts()
df_val['level'].value_counts()

# setting the image id as the index
df.set_index('image', inplace=True)

folder_1 = os.listdir('./dataset/train_res')

train_list = list(df_train['image'])
val_list = list(df_val['image'])

# transfering the training and validation images
for image in train_list:
    fname = image + '.jpeg'
    label = df.loc[image, 'level']
    if fname in folder_1:
        src = os.path.join('./dataset/train_res/', fname)
        dst = os.path.join(train_dir, str(label), fname)
        shutil.copyfile(src, dst)
    
for image in val_list:
    fname = image + '.jpeg'
    label = df.loc[image, 'level']
    if fname in folder_1:
        src = os.path.join('./dataset/train_res/', fname)
        dst = os.path.join(val_dir, str(label), fname)
        shutil.copyfile(src, dst)
    

# shecking how many training and validation images are in each folder
print(len(os.listdir('base_dir/train_dir/0')))
print(len(os.listdir('base_dir/train_dir/1')))
print(len(os.listdir('base_dir/train_dir/2')))
print(len(os.listdir('base_dir/train_dir/3')))
print(len(os.listdir('base_dir/train_dir/4')))

print(len(os.listdir('base_dir/val_dir/0')))
print(len(os.listdir('base_dir/val_dir/1')))
print(len(os.listdir('base_dir/val_dir/2')))
print(len(os.listdir('base_dir/val_dir/3')))
print(len(os.listdir('base_dir/val_dir/4')))


# augmenting the image data
class_list = ['0', '1', '2', '3', '4']

for item in class_list:
    # creating a temporary directory for the augmented images
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    img_class = item

    img_list = os.listdir('base_dir/train_dir/' + img_class)

    # copying images from the class train dir to the img_dir
    
    for fname in img_list:
        src = os.path.join('base_dir/train_dir/' + img_class, fname)
        dst = os.path.join(img_dir, fname)
        shutil.copyfile(src, dst)

    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    # creating a image data generator to augment the images in real time
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='jpeg',
                                              target_size=(224, 224),
                                              batch_size=batch_size)

    # generating the augmented images and adding them to the training folders
    num_aug_images_wanted = 1000
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

    for i in range(0, num_batches):
        imgs, labels = next(aug_datagen)

    #shutil.rmtree('aug_dir')

# checking how many training and validation images are in each folder
print(len(os.listdir('base_dir/train_dir/0')))
print(len(os.listdir('base_dir/train_dir/1')))
print(len(os.listdir('base_dir/train_dir/2')))
print(len(os.listdir('base_dir/train_dir/3')))
print(len(os.listdir('base_dir/train_dir/4')))

print(len(os.listdir('base_dir/val_dir/0')))
print(len(os.listdir('base_dir/val_dir/1')))
print(len(os.listdir('base_dir/val_dir/2')))
print(len(os.listdir('base_dir/val_dir/3')))
print(len(os.listdir('base_dir/val_dir/4')))
