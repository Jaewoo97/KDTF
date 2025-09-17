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
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class FDE(Metric):

    def __init__(self,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(FDE, self).__init__(dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               reg_mask: torch.Tensor) -> None:
        reg_mask[:, :-1] = False
        reg_mask_count = reg_mask.sum(-1)
        reg_mask_count[reg_mask_count==0] = 1
        ade = torch.norm(pred - target, p=2, dim=-1)
        ade = ade * reg_mask    # N X T
        ade = ade.sum(-1) / reg_mask_count
        self.sum += ade.sum()
        self.count += (reg_mask.sum(-1) > 0).sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
