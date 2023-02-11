import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self,
        n_way,
        k_support,
        k_query,
        batch_type,
        config={},
        device=torch.device("cpu"),
        cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_size: size of meta batch size (e.g. number of functions)
            batch_type: train/val/test
        """
        self.k_support = k_support
        self.k_query = k_query
        self.n_way = n_way

        data_folder = config.get("data_folder", "../hw1_starter_code/omniglot_resized")
        # data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        # self.dim_input = np.prod(self.img_size)
        self.dim_output = self.n_way

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        # num_val = 100
        num_train = 1200
        self.metatrain_character_folders = character_folders[:num_train]
        # self.metaval_character_folders = character_folders[num_train : num_train + num_val]
        self.metatest_character_folders = character_folders[num_train :]
        self.device = device
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        # elif batch_type == "val":
        #     self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

        # eu
        self.taskCounter = 0

        # self.label = []
        # for path in self.folder:
        #     self.label.append(os.path.sep(path)[-1])

    def image_file_to_array(self, filename):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        # image = image.reshape([dim_input])
        image = image.astype(np.float32) / 255.0
        image = 1.0 - image
        image = np.expand_dims(image, 0)
        # print(image.shape)
        if self.image_caching:
            self.stored_images[filename] = image
        return image

    def _sample(self):

        # return x_support, y_support, x_query, y_query
        # x_support = (support_size, channels, height, width)
        # y_support = (support_size)
        # x_query = (query_size, channels, height, width)
        # y_query = (query_size)
        # where support_size = k_support * n_way
        # and query_size = k_query * n_way

        tasks = np.random.randint(0, len(self.folders), size=self.n_way)

        support_images = []
        support_labels = []

        for k in range(self.k_support):
            for i, task in enumerate(tasks):
                taskFolder = self.folders[task]
                files = os.listdir(taskFolder)
                path = os.path.join(self.folders[task], random.sample(files,1)[0])  
                image = self.image_file_to_array(filename=path)
                support_images.append(image)
                support_labels.append(i)

        c = list(zip(support_images, support_labels))
        random.shuffle(c)
        support_images, support_labels = zip(*c)

        query_images = []
        query_labels = []

        for k in range(self.k_query):
            for i, task in enumerate(tasks):
                taskFolder = self.folders[task]
                files = os.listdir(taskFolder)
                path = os.path.join(self.folders[task], random.sample(files,1)[0])  
                image = self.image_file_to_array(filename=path)
                query_images.append(image)
                query_labels.append(i)

        c = list(zip(query_images, query_labels))
        random.shuffle(c)
        query_images, query_labels = zip(*c)

        support_images_np = np.array(support_images)
        support_labels_np = np.array(support_labels)
        query_images_np = np.array(query_images)
        query_labels_np = np.array(query_labels)

        return support_images_np, support_labels_np, query_images_np, query_labels_np

    def __iter__(self):
        while True:
            yield self._sample()
