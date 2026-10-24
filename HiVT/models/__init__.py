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
from models.decoder import GRUDecoder
from models.decoder import MLPDecoder
from models.embedding import MultipleInputEmbedding
from models.embedding import SingleInputEmbedding
from models.embedding import TrajPoseEmbedding, MultipleInputEmbedding_mask, MultipleInputEmbedding_mask_v2, MultipleInputEmbedding_mask_v3, MultipleInputEmbedding_mask_v4, MultipleInputEmbedding_mask_v5
from models.global_interactor import GlobalInteractor
from models.global_interactor import GlobalInteractorLayer
from models.global_interactor_text import GlobalInteractor_text

from models.local_encoder import AAEncoder
from models.local_encoder import ALEncoder
from models.local_encoder import LocalEncoder
from models.local_encoder import TemporalEncoder
from models.local_encoder import TemporalEncoderLayer
from models.local_encoder_onlyTraj import LocalEncoder_onlyTraj
from models.local_encoder_onlyTraj_v2 import LocalEncoder_onlyTraj_v2
from models.local_encoder_onlyTraj_v3 import LocalEncoder_onlyTraj_v3
from models.local_encoder_wPose import LocalEncoder_wPose
from models.local_encoder_wPose_joint import LocalEncoder_wPose_joint
from models.local_encoder_wPose_withAngle import LocalEncoder_wPose_withAngle
from models.local_encoder_wText import LocalEncoder_wText
from models.local_encoder_wText_intra import LocalEncoder_wText_intra
from models.local_encoder_wPoseText import LocalEncoder_wPoseText
from models.local_encoder_wPoseText_intra import LocalEncoder_wPoseText_intra
from models.local_encoder_wPoseText_intra_joint import LocalEncoder_wPoseText_intra_joint
from models.local_encoder_wPoseText_embedMask import LocalEncoder_wPoseText_embedMask
from models.local_encoder_wPoseText_embedMask_teacher import LocalEncoder_wPoseText_embedMask_teacher
from models.local_encoder_wPoseText_v1 import LocalEncoder_wPoseText_v1
from models.local_encoder_wPoseText_v2 import LocalEncoder_wPoseText_v2
from models.local_encoder_wPoseText_v3 import LocalEncoder_wPoseText_v3
from models.local_encoder_wPose_8steps import LocalEncoder_wPose_8steps
from models.local_encoder_wPose_withoutRotation import LocalEncoder_wPose_withoutRotation
from models.local_encoder_wPoseText_woRotation import LocalEncoder_wPoseText_woRotation

# Text ablation
from models.local_encoder_wText_egoOther import LocalEncoder_wText_egoOther
from models.local_encoder_wText_egoOtherInteraction import LocalEncoder_wText_egoOtherInteraction
from models.local_encoder_wText_onlyEgo import LocalEncoder_wText_onlyEgo
from models.local_encoder_wPoseText_onlyEgo import LocalEncoder_wPoseText_onlyEgo
from models.local_encoder_wPoseText_egoOther import LocalEncoder_wPoseText_egoOther
from models.local_encoder_wPoseText_egoOtherInteraction import LocalEncoder_wPoseText_egoOtherInteraction

# ETH/UCY
from models.local_encoder_wPose_ethucy import LocalEncoder_wPose_ethucy
from models.local_encoder_wPoseText_ethucy import LocalEncoder_wPoseText_ethucy
from models.local_encoder_wPoseText_ethucy_v1 import LocalEncoder_wPoseText_ethucy_v1
from models.local_encoder_wText_ethucy import LocalEncoder_wText_ethucy
# from models.local_encoder_original import LocalEncoder_original

# SIT
from models.local_encoder_wPoseText_sit import LocalEncoder_wPoseText_sit