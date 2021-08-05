from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class UnrealPerson(BaseImageDataset):
    """
    UnrealPerson
    Reference:
    Zhang et al. UnrealPerson: An Adaptive Pipeline towards Costless Person Re-identification. CVPR 2021.
    URL: https://github.com/FlyHighest/UnrealPerson
    "list_unreal_train.txt" is from https://github.com/FlyHighest/UnrealPerson/tree/main/JVTC/list_unreal

    Dataset statistics:
    # identities: 3000
    # cameras: 34
    # images: 120,000
    """
    dataset_dir = ''

    def __init__(self, root, verbose=True, **kwargs):
        super(UnrealPerson, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_list = osp.join(self.dataset_dir, 'list_unreal_train.txt')

        self._check_before_run()

        train = self._process_dir(self.train_list)
        self.train = train
        self.query = []
        self.gallery = []
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        if verbose:
            print("=> UnrealPerson loaded")
            print("  subset   | # ids | # cams | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:5d} | {:8d}"
                  .format(self.num_train_pids, self.num_train_cams, self.num_train_imgs))

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))

    def _process_dir(self, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        dataset = []
        pid_container = set()
        for line in lines:
            line = line.strip()
            pid = line.split(' ')[1]
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        for line in lines:
            line = line.strip()
            fname, pid, camid = line.split(' ')[0], line.split(' ')[1], int(line.split(' ')[2])
            img_path = osp.join(self.dataset_dir, fname)
            dataset.append((img_path, pid2label[pid], camid))

        return dataset
