# coding: utf-8
"""
    Building the dataset trying to follow `glyphreader` repository
    as closely as possible; dataset is constructed and split using Pool there,
    so the exactly the same split as in `glyphreader` does not seem
    to be reproducible; hence we do our own splitting.
"""

import hashlib
import shutil
import logging
from collections import Counter
from os import listdir, makedirs
from os.path import isdir, isfile, join, dirname

import numpy as np
from sklearn.model_selection import train_test_split

UNKNOWN_LABEL = "UNKNOWN"

if __name__ == "__main__":

    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument("--data_path", default="/".join(("data", "Dataset", "Manual", "Preprocessed")))
    ap.add_argument("--prepared_data_path", default="prepared_data")
    ap.add_argument("--test_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=261)

    arguments = ap.parse_args()

    # prepare yourself for some hardcode
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    file_dir = dirname(__file__)
    stele_path = join(file_dir, arguments.data_path)
    steles = [join(stele_path, f) for f in listdir(stele_path) if isdir(join(stele_path, f))]

    res_image_paths, labels = [], []

    for stele in steles:

        image_paths = [join(stele, f) for f in listdir(stele) if isfile(join(stele, f))]

        for path in image_paths:
            res_image_paths.append(path)
            labels.append(path[(path.rfind("_") + 1): path.rfind(".")])

    list_of_paths = np.asarray(res_image_paths)
    labels = np.array(labels)

    logging.debug(f"Labels total: {len(set(labels))}")

    labels_just_once = np.array([l for (l, c) in Counter(labels).items() if c <= 1])
    logging.debug(f"Labels seen just once: {len(labels_just_once)}")

    # those hieroglyphs that were seen in data only once, go to TRAIN set
    to_be_added_to_train_only = np.nonzero(np.isin(labels, labels_just_once))[0]

    # the hieroglyphs that have NO label are to be removed
    to_be_deleted = np.nonzero(labels == UNKNOWN_LABEL)[0]

    # we remove all elements of these two sets
    to_be_deleted = np.concatenate([to_be_deleted, to_be_added_to_train_only])
    filtered_list_of_paths = np.delete(list_of_paths, to_be_deleted, 0)
    filtered_labels = np.delete(labels, to_be_deleted, 0)

    # we split the data
    train_paths, test_paths, y_train, y_test = train_test_split(filtered_list_of_paths,
                                                                filtered_labels,
                                                                stratify=filtered_labels,
                                                                test_size=arguments.test_fraction,
                                                                random_state=arguments.seed)

    # we add the 'single-occurence' folks to the train set
    train_paths = np.concatenate([train_paths, list_of_paths[to_be_added_to_train_only]])
    y_train = np.concatenate([y_train, labels[to_be_added_to_train_only]])

    # then we copy all
    makedirs(arguments.prepared_data_path, exist_ok=True)
    [makedirs(join(arguments.prepared_data_path, "train", + l), exist_ok=True) for l in set(y_train)]
    [makedirs(join(arguments.prepared_data_path, "test", + l), exist_ok=True) for l in set(y_test)]

    for fp, label in zip(train_paths, y_train):
        fn = join(arguments.prepared_data_path, "train", label, hashlib.md5(fp.encode('utf-8')).hexdigest() + ".png")
        shutil.copyfile(fp, fn)

    for fp, label in zip(test_paths, y_test):
        fn = join(arguments.prepared_data_path, "test", label, hashlib.md5(fp.encode('utf-8')).hexdigest() + ".png")
        shutil.copyfile(fp, fn)
