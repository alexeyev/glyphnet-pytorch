# coding: utf-8
"""
    Building the dataset trying to follow `glyphreader` repository
    as closely as possible; dataset is constructed and split using Pool there,
    so the exactly the same split as in `glyphreader` does not seem
    to be reproducible; hence we do our own splitting.
"""

import hashlib
import shutil
from collections import Counter
from os import listdir, makedirs
from os.path import isdir, isfile, join, dirname

import numpy as np
from sklearn.model_selection import train_test_split

file_dir = dirname(__file__)
stele_path = join(file_dir, "data", "Dataset", "Manual", "Preprocessed")
steles = [join(stele_path, f) for f in listdir(stele_path) if isdir(join(stele_path, f))]

res_image_paths, labels = [], []

for stele in steles:

    image_paths = [join(stele, f) for f in listdir(stele) if isfile(join(stele, f))]

    for path in image_paths:
        res_image_paths.append(path)
        labels.append(path[(path.rfind("_") + 1): path.rfind(".")])

list_of_paths = np.asarray(res_image_paths)
labels = np.array(labels)

print("Labels total:", len(set(labels)))

labels_just_once = np.array([l for (l, c) in Counter(labels).items() if c <= 1])
print("Labels seen just once:", len(labels_just_once), labels_just_once)

to_be_added_to_train_only = np.nonzero(np.isin(labels, labels_just_once))[0]
to_be_deleted = np.nonzero(labels=="UNKNOWN")[0]

to_be_deleted = np.concatenate([to_be_deleted, to_be_added_to_train_only])

filtered_list_of_paths = np.delete(list_of_paths, to_be_deleted, 0)
filtered_labels = np.delete(labels, to_be_deleted, 0)

train_paths, test_paths, y_train, y_test = train_test_split(filtered_list_of_paths, filtered_labels, stratify=filtered_labels,
                                                            test_size=0.20, random_state=42)

train_paths = np.concatenate([train_paths, list_of_paths[to_be_added_to_train_only]])
y_train = np.concatenate([y_train, labels[to_be_added_to_train_only]])

makedirs("prepared_data", exist_ok=True)
[makedirs("prepared_data/train/" + l, exist_ok=True) for l in set(y_train)]
[makedirs("prepared_data/test/" + l, exist_ok=True) for l in set(y_test)]

for fp, label in zip(train_paths, y_train):
    shutil.copyfile(fp, f"prepared_data/train/{label}/{hashlib.md5(fp.encode('utf-8')).hexdigest()}.png")

for fp, label in zip(test_paths, y_test):
    shutil.copyfile(fp, f"prepared_data/test/{label}/{hashlib.md5(fp.encode('utf-8')).hexdigest()}.png")
