# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Space distortions which occur as a function of time."""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import (
    Encoding,
    NeRFEncoding,
    TensorVMEncoding,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.utils.math import de_casteljau


class TemporalDistortion(nn.Module):
    """Apply spatial distortions as a function of time"""

    def forward(self, positions: TensorType["bs":..., 3], times: Optional[TensorType[1]]) -> TensorType["bs":..., 3]:
        """
        Args:
            positions: Samples to translate as a function of time
            times: times for each sample

        Returns:
            Translated positions.
        """


class TemporalDistortionKind(Enum):
    """Possible temporal distortion names"""

    DNERF = "dnerf"
    DTENSORF = "dtensorf"

    def to_temporal_distortion(self, config: Dict[str, Any]) -> TemporalDistortion:
        """Converts this kind to a temporal distortion"""
        if self == TemporalDistortionKind.DNERF:
            return DNeRFDistortion(**config)
        if self == TemporalDistortionKind.DTENSORF:
            return DTensoRFDistortion(**config)
        raise NotImplementedError(f"Unknown temporal distortion kind {self}")


class DNeRFDistortion(TemporalDistortion):
    """Optimizable temporal deformation using an MLP.
    Args:
        position_encoding: An encoding for the XYZ of distortion
        temporal_encoding: An encoding for the time of distortion
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
        self,
        position_encoding: Encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ),
        temporal_encoding: Encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ),
        mlp_num_layers: int = 4,
        mlp_layer_width: int = 256,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding
        self.mlp_deform = MLP(
            in_dim=self.position_encoding.get_out_dim() + self.temporal_encoding.get_out_dim(),
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

    def forward(self, positions, times=None):
        if times is None:
            return None
        p = self.position_encoding(positions)
        t = self.temporal_encoding(times)
        return self.mlp_deform(torch.cat([p, t], dim=-1))


class DTensoRFDistortion(TemporalDistortion):
    """Optimizable temporal deformation using an MLP.
    Args:
        position_encoding: An encoding for the XYZ of distortion
        temporal_encoding: An encoding for the time of distortion
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
        self,
        aabb=torch.tensor([[-1.5] * 3, [1.5] * 3]),
        spline_encoding=TensorVMEncoding(resolution=128, num_components=16, init_scale=0.1),
        num_points=4,
    ):
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spline_encoding = spline_encoding

        self.num_points = num_points
        self.mlp_head = MLP(
            in_dim=3 + self.spline_encoding.get_out_dim(),
            num_layers=3,
            layer_width=256,
            out_dim=1 + num_points * 3,
        )

    def forward(self, positions, times=None):
        if times is None:
            return None
        norm_p = 2 * SceneBox.get_normalized_positions(positions, self.aabb) - 1
        splines = self.spline_encoding(norm_p)
        rigidity, splines = self.mlp_head(torch.cat([norm_p, splines], dim=-1))\
          .split([1, self.num_points * 3], dim=-1)

        splines = splines.reshape(*splines.shape[:-1], 3, self.num_points)
        offset = de_casteljau(splines, times.unsqueeze(-1), self.num_points)
        offset = offset * rigidity.sigmoid()

        return SceneBox.denormalize_positions(offset, self.aabb)

