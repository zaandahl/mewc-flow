import pandas as pd
import numpy as np
import os
import os.path
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import imgaug as ia

ia.seed(42)

# This function creates a pandas dataframe with the image path and class label derived from the directory structure
def create_dataframe(ds_path, n=[1000], seed=42):
    # Selecting folder paths in dataset
    dir_ = Path(ds_path)
    ds_filepaths = list(dir_.glob(r'**/*.jpg'))
    class_labels = [Path(f).name for f in dir_.iterdir() if f.is_dir()]
    print(class_labels)
    # Mapping labels...
    ds_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], ds_filepaths))
    # Data set paths & labels
    ds_filepaths = pd.Series(ds_filepaths, name='File').astype(str)
    ds_labels = pd.Series(ds_labels, name='Label')
    # Concatenating...
    ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
    num_classes = len(ds_labels.unique())

    #Sampling 
    ds_samp = pd.DataFrame()
    if(n != None): 
        # if n is a single number, just sample that for each class
        for x in range(num_classes):
            group_df = ds_df.groupby('Label').get_group(class_labels[x])
            group_pop = len(group_df.index)
            if(x < len(n)):
                samp = n[x]
            else:
                samp = n[len(n)-1]
            if(samp > group_pop):
                rep_flag = True
            else:
                rep_flag = False
            print(class_labels[x])
            print(group_pop)
            print(samp)
            print("--")
            ds_samp = pd.concat([ds_samp, group_df.sample(n=samp, replace=rep_flag)])
    # Randomising
    ds_samp = ds_samp.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(ds_samp.groupby('Label').size())
    # n = ds_samp.shape[0]
    # print(n)
    return ds_samp

def split_dataframe(ds_path, seed=42, val_split=0.2, test_split=0, n=1000):
    insample_df = create_dataframe(ds_path, n=n, seed=seed)
    num_classes = insample_df['Label'].nunique()
    print(num_classes)
    valtest_size = val_split + test_split
    train_df, valtest_df = train_test_split(insample_df, test_size=valtest_size, stratify=insample_df['Label'])
    # # Randomise and reset indexes
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    valtest_df = valtest_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if test_split != 0 :
        test_size = test_split / valtest_size
        val_df, test_df = train_test_split(valtest_df, test_size=test_size, stratify=valtest_df['Label'])
        test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else :
        val_df = valtest_df
        test_df = None
    # # Randomise and reset indexes
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return(train_df, val_df, test_df, num_classes)

# This function takes a pandas df from create_dataframe and converts to a TensorFlow dataset
def create_dataset(in_df, img_size, batch_size, magnitude, ds_name="train"):
    # helper function to use with the lambda mapping
    def load(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)
        img = tf.image.resize(img, size=(img_size, img_size))
        return img
    in_path = in_df['File']
    in_class = LabelEncoder().fit_transform(in_df['Label'].values)

    in_class = in_class.reshape(len(in_class), 1)
    in_class = OneHotEncoder(sparse=False).fit_transform(in_class)
    print(in_path.shape)
    print(in_class.shape)

    rand_aug = iaa.RandAugment(n=3, m=magnitude)

    def augment(images):
        # Input to `augment()` is a TensorFlow tensor which
        # is not supported by `imgaug`. This is why we first
        # convert it to its `numpy` variant.
        images = tf.cast(images, tf.uint8)
        return rand_aug(images=images.numpy())
    # convert to dataset
    if ds_name == "train":
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load(img_path), img_class))
            .batch(batch_size)
            .map(lambda img, img_class: (tf.py_function(augment, [img], [tf.float32])[0], img_class), num_parallel_calls=tf.data.AUTOTUNE,)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load(img_path), img_class),)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )  
    return(ds)
