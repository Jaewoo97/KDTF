# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm
import sys
sys.path.append('/mnt/jaewoo4tb/textraj/')
from HiVT.utils_hivt import TemporalData
class jrdbDataset_poseViz(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None) -> None:
        print(f'Using jrdb dataset, split {split}')
        self._split = split
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val_testPose'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        # Instead of assigning to the property, assign to a private variable
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in os.listdir(self.processed_dir)]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(jrdbDataset_poseViz, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names  # Use the internal variable to return the value

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def len(self) -> int:
        return len(self.processed_paths)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])
        # return TemporalData(**torch.load(self.processed_paths[idx]))
