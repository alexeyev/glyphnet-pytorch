# coding: utf-8

from typing import Dict, Tuple, List

from torchvision.datasets import ImageFolder


class GlyphData(ImageFolder):

    def __init__(self, class_to_idx: Dict[str, int], root: str = "prepared_data/train/", *args, **kwargs):
        self.classes_list = ["UNKNOWN" for _ in range(max(class_to_idx.values()) + 1)]
        self.classes_map = class_to_idx

        for k, v in class_to_idx.items():
            self.classes_list[v] = k

        super(GlyphData, self).__init__(root=root, *args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return self.classes_list, self.classes_map
