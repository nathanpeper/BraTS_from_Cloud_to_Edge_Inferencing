#
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from pathlib import Path

from argparser import args

import nibabel as nib


class DatasetGenerator:

    def __init__(self,
                 data_dir=args.data_dir,
                 dataset=args.dataset,
                 data_path=args.data_path,
                 train_test_split=args.train_test_split,
                 validate_test_split=args.validate_test_split,
                 batch_size_train=args.batch_size_train,
                 batch_size_validate=args.batch_size_validate,
                 batch_size_test=args.batch_size_test,
                 tile_height=args.tile_height,
                 tile_width=args.tile_width,
                 tile_depth=args.tile_depth,
                 number_input_channels=args.number_input_channels,
                 crop_dim = (args.tile_height, args.tile_width,
                             args.tile_depth, args.number_input_channels),
                 number_output_classes=args.number_output_classes,
                 random_seed=args.random_seed,
                 shard=0):

        self.data_dir = data_dir
        self.dataset = dataset
        self.data_path = data_path
        self.train_test_split = train_test_split
        self.validate_test_split = validate_test_split
        self.batch_size_train = batch_size_train
        self.batch_size_validate = batch_size_validate
        self.batch_size_test = batch_size_test
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.tile_depth = tile_depth
        self.crop_dim = crop_dim
        self.number_input_channels = number_input_channels
        self.number_output_classes = number_output_classes
        self.random_seed = random_seed
        self.shard = shard  # For Horovod, gives different shard per worker

        self.create_file_list()

        self.ds_train, self.ds_val, self.ds_test = self.get_dataset()

    def create_file_list(self):
        """
        Get list of the files from the BraTS raw data
        Split into training and testing sets.
        """
        import os
        import json

        json_filename = os.path.join(self.data_path, "dataset.json")

        try:
            with open(json_filename, "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            print(f"File {json_filename} doesn't exist. It should be part of the "
                  "Decathlon directory")

        self.output_channels = experiment_data["labels"]
        self.input_channels = experiment_data["modality"]
        self.description = experiment_data["description"]
        self.name = experiment_data["name"]
        self.release = experiment_data["release"]
        self.license = experiment_data["licence"]
        self.reference = experiment_data["reference"]
        self.tensor_image_size = experiment_data["tensorImageSize"]
        self.num_files = experiment_data["numTraining"]

        """
        Create a dictionary of tuples with image filename and label filename
        """
        self.filenames = {}
        for idx in range(self.num_files):
            self.filenames[idx] = [os.path.join(self.data_path,
                                                experiment_data["training"][idx]["image"]),
                                   os.path.join(self.data_path,
                                                experiment_data["training"][idx]["label"])]

    def print_info(self):
        """
        Print the dataset information
        """

        print("="*30)
        print("Dataset name:        ", self.name)
        print("Dataset description: ", self.description)
        print("Tensor image size:   ", self.tensor_image_size)
        print("Dataset release:     ", self.release)
        print("Dataset reference:   ", self.reference)
        print("Input channels:      ", self.input_channels)
        print("Output labels:       ", self.output_channels)
        print("Dataset license:     ", self.license)
        print("="*30)

    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[..., channel]
            img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            img[..., channel] = img_temp

        return img

    def crop(self, img, msk, randomize):
        """
        Randomly crop the image and mask
        """

        slices = []

        # Do we randomize?
        is_random = randomize and np.random.rand() > 0.5

        for idx in range(len(img.shape)-1):  # Go through each dimension

            crop_len = self.crop_dim[idx]
            img_len = img.shape[idx]

            start = (img_len-crop_len)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + crop_len) > img_len):  # Don't fall off the image
                        start = (img_len-crop_len)//2
            else:
                start = 0

            slices.append(slice(start, start+crop_len))

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """

        # Determine if axes are equal and can be rotated
        # If the axes aren't equal then we can't rotate them.
        equal_dim_axis = []
        for idx in range(0, len(self.crop_dim)):
            for jdx in range(idx+1, len(self.crop_dim)):
                if self.crop_dim[idx] == self.crop_dim[jdx]:
                    equal_dim_axis.append([idx, jdx])  # Valid rotation axes
        dim_to_rotate = equal_dim_axis

        if np.random.rand() > 0.5:
            # Random 0,1 (axes to flip)
            ax = np.random.choice(np.arange(len(self.crop_dim)-1))
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        elif (len(dim_to_rotate) > 0) and (np.random.rand() > 0.5):
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            # This will choose the axes to rotate
            # Axes must be equal in size
            random_axis = dim_to_rotate[np.random.choice(len(dim_to_rotate))]

            img = np.rot90(img, rot, axes=random_axis)  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=random_axis)  # Rotate axes 0 and 1

        return img, msk

    def read_nifti_file(self, idx, randomize=False):
        """
        Read Nifti file
        """

        idx = idx.numpy()
        img_file = self.filenames[idx][0]
        msk_file = self.filenames[idx][1]

        img = np.array(nib.load(img_file).dataobj)

        img = np.rot90(img[..., [0]])  # Just take the FLAIR channel (0)

        msk = np.rot90(np.array(nib.load(msk_file).dataobj))

        """
        "labels": {
             "0": "background",
             "1": "edema",
             "2": "non-enhancing tumor",
             "3": "enhancing tumour"}
         """
        # Combine all masks but background
        if self.number_output_classes == 1:
            msk[msk > 0] = 1.0
            msk = np.expand_dims(msk, -1)
        else:
            msk_temp = np.zeros(list(msk.shape) + [self.number_output_classes])
            for channel in range(self.number_output_classes):
                msk_temp[msk == channel, channel] = 1.0
            msk = msk_temp

        # Crop
        img, msk = self.crop(img, msk, randomize)

        # Normalize
        img = self.z_normalize_img(img)

        # Randomly rotate
        if randomize:
            img, msk = self.augment_data(img, msk)

        return img, msk

    def plot_images(self, ds, slice_num=None):
        """
        Plot images from dataset
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 20))

        num_cols = 2

        msk_channel = 0
        img_channel = 0

        for img, msk in ds.take(1):
            batch_size = img.shape[0]
            if slice_num == None:
                slice_num = int(img.shape[3] / 2)

            for idx in range(batch_size):
                plt.subplot(batch_size, num_cols, idx*num_cols + 1)
                plt.imshow(img[idx, :, :, slice_num, img_channel], cmap="gray")
                plt.title(f"MRI {ds.input_channels[str(img_channel)]}", fontsize=18)
                plt.subplot(batch_size, num_cols, idx*num_cols + 2)
                plt.imshow(msk[idx, :, :, slice_num, msk_channel], cmap="gray")
                plt.title("Tumor", fontsize=18)

        plt.show()

        print(f"Mean pixel value of image = {np.mean(img[0, :, :, :, 0])}")

    def display_train_images(self, slice_num=None):
        """
        Plots some training images
        """
        self.plot_images(self.ds_train, slice_num)

    def display_validation_images(self, slice_num=None):
        """
        Plots some validation images
        """
        self.plot_images(self.ds_val, slice_num)

    def display_test_images(self, slice_num=None):
        """
        Plots some test images
        """
        self.plot_images(self.ds_test, slice_num)

    def get_train(self):
        """
        Return train dataset
        """
        return self.ds_train

    def get_test(self):
        """
        Return test dataset
        """
        return self.ds_test

    def get_validate(self):
        """
        Return validation dataset
        """
        return self.ds_val

    def get_dataset(self):
        """
        Create a TensorFlow data loader
        """
        self.num_train = int(self.num_files * self.train_test_split)
        num_val_test = self.num_files - self.num_train

        ds = tf.data.Dataset.range(self.num_files).shuffle(
            self.num_files, self.random_seed)  # Shuffle the dataset

        """
        Horovod Sharding
        Here we are not actually dividing the dataset into shards
        but instead just reshuffling the training dataset for every
        shard. Then in the training loop we just go through the training
        dataset but the number of steps is divided by the number of shards.
        """
        ds_train = ds.take(self.num_train).shuffle(
            self.num_train, self.shard)  # Reshuffle based on shard
        ds_val_test = ds.skip(self.num_train)
        self.num_val = int(num_val_test * self.validate_test_split)
        self.num_test = self.num_train - self.num_val
        ds_val = ds_val_test.take(self.num_val)
        ds_test = ds_val_test.skip(self.num_val)

        ds_train = ds_train.map(lambda x: tf.py_function(self.read_nifti_file,
                                                         [x, True], [tf.float32, tf.float32]),
                                num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = ds_val.map(lambda x: tf.py_function(self.read_nifti_file,
                                                     [x, False], [tf.float32, tf.float32]),
                            num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(lambda x: tf.py_function(self.read_nifti_file,
                                                       [x, False], [tf.float32, tf.float32]),
                              num_parallel_calls=tf.data.AUTOTUNE)

        ds_train = ds_train.repeat()
        ds_train = ds_train.batch(self.batch_size_train)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_val = ds_val.batch(self.batch_size_validate)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

        ds_test = ds_test.batch(self.batch_size_test)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_val, ds_test


if __name__ == "__main__":

    print("Load the data and plot a few examples")

    
    crop_dim = (args.tile_height, args.tile_width,
                args.tile_depth, args.number_input_channels)

    """
    Load the dataset
    """
    brats_data = DatasetGenerator()

    brats_data.print_info()
