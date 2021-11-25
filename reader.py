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

res_image_paths, labels = [], []
batch_size = 200

print("indexing images...")
steles = [join(stele_path, f) for f in listdir(stele_path) if isdir(join(stele_path, f))]

for stele in steles:

    image_paths = [join(stele, f) for f in listdir(stele) if isfile(join(stele, f))]

    for path in image_paths:
        res_image_paths.append(path)
        labels.append(path[(path.rfind("_") + 1): path.rfind(".")])

list_of_paths = np.asarray(res_image_paths)
labels = np.asarray(labels)

print(len(set(labels)))

print(Counter(labels).most_common(200))
quit()

to_be_deleted = np.nonzero(labels == "UNKNOWN")  # Remove the Unknown class from the database
list_of_paths = np.delete(list_of_paths, to_be_deleted, 0)
labels = np.delete(labels, to_be_deleted, 0)
train_paths, test_paths, y_train, y_test = train_test_split(list_of_paths, labels, stratify=labels,
                                                            test_size=0.20, random_state=42)

makedirs("prepared_data", exist_ok=True)
[makedirs("prepared_data/train/" + l) for l in set(y_train)]
[makedirs("prepared_data/test/" + l) for l in set(y_test)]

for fp, label in zip(train_paths, y_train):
    shutil.copyfile(fp, f"prepared_data/train/{label}/{hashlib.md5(fp.encode('utf-8')).hexdigest()}.png")

for fp, label in zip(test_paths, y_test):
    shutil.copyfile(fp, f"prepared_data/test/{label}/{hashlib.md5(fp.encode('utf-8')).hexdigest()}.png")
