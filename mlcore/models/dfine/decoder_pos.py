"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import math
import functools
from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .denoising import get_contrastive_denoising_training_group
from mlcore.models.layers.utils import (
    distance2bbox,
    weighting_function,
    bias_init_with_prob,
    deformable_attention_core_func_v2,
    get_activation,
    inverse_sigmoid,
)
from mlcore.core import COMPONENTS, ComponentType, Task

__all__ = ["DFINETransformer"]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention"""
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ""
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        # [3 x 1/3, 6 x 1/6, 3x1/3], равномерно распределяет предсказанные сдвиги
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )

        # 8 * sum([3, 6, 3]) = 96
        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Predicts 2D spatial offsets. Out num_points x 2 (devide num_points to xy)
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=self.method
        )

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile(
            [1, sum(self.num_points_list), 1]
        )
        scaling = torch.concat(
            [torch.arange(1, n + 1) for n in self.num_points_list]
        ).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: list[int],
    ) -> torch.Tensor:
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (list): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        # Linear layer hidden_dim x total_points * 2
        # B, numQ, hid_dim > B, numQ, total_points * 2
        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        # B, numQ, 8, 12, 2
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )

        # Linear layer hidden_dim x total_points
        # B, numQ, hid_dim -> B, numQ, total_points -> B, numQ, Heads, 12
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            # Changes [H, W] -> [W, H] -> [1, 1, 1, 3, 1, 2]
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2
            )
            # Potential dim mismatch num_heads = 8, num_levels = 3
            # [B, numQ, 1, 3, 1, 2] +  [B, numQ, num_heads, -1, 2] / [1, 1, 1, 3, 1, 2]
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets  # [B, numQ, 8, 12, 2]
                * num_points_scale  # [12, 1]
                * reference_points[:, :, None, :, 2:]  # [B, numQ, 1, 1, 2]
                * self.offset_scale  # [1]
            )
            # [B, numQ, 1, 1, 2] + [B, numQ, 1, 1, 2] -> cxcy + offsets
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        # value -> [B, numQ, hid_dim]
        # value_spatial_shapes -> [[80, 80], [40, 40], [20, 20]]
        # sampling_locations -> [B, numQ, 8, 12, 2]
        # attention_weights -> [B, numQ, Heads, 12]
        # self.num_points_list -> [3, 6, 3]
        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
    ):
        """_summary_

        Args:
            d_model (int, optional): _description_. Defaults to 256.
            n_head (int, optional): _description_. Defaults to 8.
            dim_feedforward (int, optional): _description_. Defaults to 1024.
            dropout (float, optional): _description_. Defaults to 0.0.
            activation (str, optional): _description_. Defaults to "relu".
            n_levels (int, optional): _description_. Defaults to 4.
            n_points (int, optional): _description_. Defaults to 4.
            cross_attn_method (str, optional): _description_. Defaults to "default".
            layer_scale (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        value: list[torch.Tensor],
        spatial_shapes: list,
        attn_mask: torch.Tensor | None = None,
        query_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transformer decoder layer forwad function

        Args:
            target (torch.Tensor): topK queries [B, numQ, hid_dim]
            reference_points (torch.Tensor): [B, numQ, 1, 4]
            value (list[torch.Tensor]): [B, num_heads, K, spatial_size], where spatial size in [6400, 1600, 400]
            spatial_shapes (list): [[80x80], [40x40], [20x20]]
            attn_mask (torch.Tensor, optional): _description_. Defaults to None.
            query_pos_embed (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )

        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


# Что если предиктить num_kps x 32 (kps hidden dim можно настроить)
# в такком случае hidden_dim ~ 17 x 32 = 544, что по идее не должно
# сильно замедлить инференс. Также можно уменьшить numQ, т.к. это
# основной bottleneck, (attn(numQ^2) * num_kps x 32)
# Можно будет попробовать использовать refinment DFINE, НО!
# нужен ли refinment в этой таске??


class BoxTransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_head: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_levels: int,
        num_points: int,
        cross_attn_method: str,
        num_layers: int,
        reg_max: int,
        reg_scale: float,
        up: torch.Tensor,
        eval_idx=-1,
        layer_scale=2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            num_head,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )

        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            num_head,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)]
        )

    def value_op(
        self,
        memory: torch.Tensor,
        value_proj: Callable | None,
        value_scale: Callable | None,
        memory_mask: torch.Tensor | None,
        memory_spatial_shapes: list,
    ) -> tuple[torch.Tensor]:
        """Preprocess values for MSDeformableAttention.

        Args:
            memory (torch.Tensor): flattened multi-scale feature maps
            value_proj (Callable | None): _description_
            value_scale (Callable | None): _description_
            memory_mask (torch.Tensor | None): Was already applied before?
            memory_spatial_shapes (list): fe

        Returns:
            tuple[torch.Tensor]: multi-scale feature maps for Transformer input.
                Tensor with shape [B, num_heads, N, F], where N * num_heads = hidden_dim,
                F - feature map resolution
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = (
            F.interpolate(memory, size=value_scale)
            if value_scale is not None
            else value
        )
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(
            self.reg_max, self.up, self.reg_scale, deploy=True
        )
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target: torch.Tensor,
        ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: list,
        bbox_head: nn.Module,
        score_head: nn.Module,
        query_pos_head: nn.Module,
        pre_bbox_head: nn.Module,
        integral: nn.Module,
        attn_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        dn_meta: dict | None = None,
    ):
        """Forward pass Fine-grained Distribution Refinement layer

        Args:
            target (torch.Tensor): topK quries with shape [B, num_queries, hid_dim]
            ref_points_unact (torch.Tensor): predicted boxes from encoder,
                based on `target` with shape [B, num_quries, 4]
            memory (torch.Tensor): flattened multi-scale feature maps [B, F, hid_dim],
                F denotes to features map resolitions projection
            spatial_shapes (list): multi-scale feature maps size
            bbox_head (nn.Module): FFN for box predictions
            score_head (nn.Module): FFN for score predictions
            query_pos_head (nn.Module): FFN Restores reference points latent dim back into hidden_dim.
                preprocess for deformable_attention
            pre_bbox_head (nn.Module): FFN for box predictions
            integral (nn.Module): A static layer that calculates integral results from a distribution.
            attn_mask (torch.Tensor, optional): attn_mask from contrastive denoising. Defaults to None.
            memory_mask (torch.Tensor, optional): _description_. Defaults to None.
            dn_meta (dict | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes, dec_out_logits = [], []
        dec_out_pred_corners, dec_out_refs = [], []

        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, self.up, self.reg_scale)
        else:
            project = self.project

        # predicted boxes from topK mem + enc_topK_anchors
        # Forces coordinates to be in [0..1], shape [B, num_quries, 4]
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            output = layer(
                output,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
            )

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(
                    pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            # pred corners shape B, num_quries, 132 (32 * 4)
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(
                ref_points_initial, integral(pred_corners, project), self.reg_scale
            )

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            output,
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class PoseLayer(nn.Module):
    def __init__(
        self, num_kps: int = 17, hidden_dim: int = 256, box_scale: float = 1.15
    ):
        super().__init__()
        self.num_kps = num_kps
        self.box_scale = box_scale
        self.kps_hidden_dim = hidden_dim // 4

        # 256 x 17*128 x 17*64 -> 256 -> 2176 -> 1088
        self.pre_pos_layer = MLP(
            hidden_dim,
            self.num_kps * hidden_dim // 2,
            self.num_kps * self.kps_hidden_dim,
            num_layers=2,
        )

        self.mha = nn.MultiheadAttention(self.kps_hidden_dim, 4, batch_first=True)
        self.gateway = Gate(d_model=self.kps_hidden_dim)

        self.pose_emb = nn.Parameter(
            torch.randn(1, self.num_kps, self.kps_hidden_dim), requires_grad=True
        )

        self.vis_head = nn.Linear(self.kps_hidden_dim, 1)

        self.x_bins = int(self.kps_hidden_dim * 2)
        self.x_sinemb = self._sincos_pe(
            d_model=self.kps_hidden_dim, max_len=self.x_bins
        )[None, None, :, :]

        self.y_bins = int(self.kps_hidden_dim * 3)
        self.y_sinemb = self._sincos_pe(
            d_model=self.kps_hidden_dim, max_len=self.y_bins
        )[None, None, :, :]

        self.bin_proj = nn.Linear(self.kps_hidden_dim, self.kps_hidden_dim)

    def with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def make_linspace(
        self, start: torch.Tensor, end: torch.Tensor, steps: int
    ) -> torch.Tensor:

        # start: [B, numQ], end: [B, numQ] | alpha shape [num_steps]
        alpha = torch.linspace(0, 1, steps)
        return start.unsqueeze(-1) + (end - start).unsqueeze(-1) * alpha

    def _encode_emb(self, bboxes: torch.Tensor) -> tuple:

        # boxes shape [B, numQ, 4]

        B = bboxes.shape[0]
        cxcy, wh = (
            bboxes[:, :, :2],
            bboxes[:, :, 2:] * self.box_scale,
        )

        x1y1 = torch.clamp(cxcy - wh / 2, min=0.0)
        x2y2 = torch.clamp(cxcy + wh / 2, max=1.0)

        x1, y1 = x1y1.unbind(-1)
        x2, y2 = x2y2.unbind(-1)
        zero = torch.zeros_like(x1)

        # B, numQ, bins
        x_coords = self.make_linspace(zero, x2 - x1, self.x_bins)
        y_coords = self.make_linspace(zero, y2 - y1, self.y_bins)

        # B, numQ, bins, kps_hid_dim
        x_emb = x_coords.unsqueeze(-1) * self.x_sinemb.repeat(B, 1, 1, 1)
        y_emb = y_coords.unsqueeze(-1) * self.y_sinemb.repeat(B, 1, 1, 1)

        return (x1, y1), (x_coords, y_coords), (x_emb, y_emb)

    @staticmethod
    def _sincos_pe(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, target: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:

        (xs, ys), (x_coords, y_coords), (xemb, yemb) = self._encode_emb(boxes)

        x_proj = self.bin_proj(xemb)
        y_proj = self.bin_proj(yemb)

        pre_pose = self.pre_pos_layer(target)
        B, numQ, _ = pre_pose.shape
        target = pre_pose.reshape(B * numQ, self.num_kps, self.kps_hidden_dim)

        q = k = self.with_pos_embed(target, self.pose_emb.repeat(B * numQ, 1, 1))

        attn, _ = self.mha(q, k, value=target)
        # B x numQ, num_kps, kps_hidden_dim
        pred = self.gateway(attn, target).reshape(
            B, numQ, self.num_kps, self.kps_hidden_dim
        )

        logits_x = pred @ x_proj.transpose(-1, -2)
        logits_y = pred @ y_proj.transpose(-1, -2)

        probs_x = F.softmax(logits_x, dim=-1)
        probs_y = F.softmax(logits_y, dim=-1)

        final_x = torch.sum(probs_x * x_coords.unsqueeze(-2), dim=-1) + xs.unsqueeze(-1)
        final_y = torch.sum(probs_y * y_coords.unsqueeze(-2), dim=-1) + ys.unsqueeze(-1)
        vis_kps = F.sigmoid(self.vis_head(pred))

        return torch.stack([final_x, final_y, vis_kps], dim=-1)


@COMPONENTS.register(
    name="DfinePosDecoder",
    ctype=ComponentType.MODEL,
    supported_tasks=Task.OBJECT_DETECTION,
)
class DFINETransformer(nn.Module):
    def __init__(
        self,
        num_kps: int = 17,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: list[int] = [512, 1024, 2048],
        feat_strides: list[int] = [8, 16, 32],
        num_levels: int = 3,
        num_points: list[int] = [3, 6, 3],
        nhead: int = 8,
        num_box_layers: int = 1,
        num_kps_layers: int = 1,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        grid_size: float = 0.05,
        activation: str = "relu",
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_spatial_size: tuple[int, int] = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
        box_scale: float = 1.15,
        aux_loss: bool = True,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        reg_max: int = 32,
        reg_scale=4.0,
        layer_scale=1,
    ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        assert isinstance(num_points, (tuple, list))

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.grid_size = grid_size
        self.hidden_dim = hidden_dim

        self.num_classes = 2
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.feat_channels = feat_channels
        self.num_levels = num_levels
        self.num_kps = num_kps
        self.num_queries = num_queries
        self.eps = eps
        self.num_box_layers = num_box_layers
        self.num_kps_layers = num_kps_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.activation = activation
        self.num_points = num_points
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        assert query_select_method in ("default", "one2many", "agnostic"), ""
        assert cross_attn_method in ("default", "discrete"), ""
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        self.box_decoder = BoxTransformerDecoder(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method,
            num_box_layers,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
        )

        self.pos_layer = PoseLayer(
            num_kps=self.num_kps, hidden_dim=hidden_dim, box_scale=box_scale
        )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(2, hidden_dim, padding_idx=1)
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    (
                        "norm",
                        nn.LayerNorm(
                            hidden_dim,
                        ),
                    ),
                ]
            )
        )

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, 2)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        scaled_dim = round(layer_scale * hidden_dim)
        self.eval_idx = eval_idx if eval_idx >= 0 else num_box_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, 2) for _ in range(self.eval_idx + 1)]
            + [
                nn.Linear(scaled_dim, 2)
                for _ in range(num_box_layers - self.eval_idx - 1)
            ]
        )

        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3)
                for _ in range(num_box_layers - self.eval_idx - 1)
            ]
        )
        self.integral = Integral(self.reg_max)

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self._reset_parameters(feat_channels)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"num_classes           {self.num_classes}\n"
            f"num_keypoints:        {self.num_kps}\n"
            f"hidden_dim:           {self.hidden_dim}\n"
            f"num_queries:          {self.num_queries}\n"
            f"feat_channels:        {self.feat_channels}\n"
            f"feat_strides:         {self.feat_strides}\n"
            f"num_levels:           {self.num_levels}\n"
            f"num_points:           {self.num_points}\n"
            f"nhead:                {self.nhead}\n"
            f"num_box_layers:       {self.num_box_layers}\n"
            f"num_kps_layers:       {self.num_kps_layers}\n"
            f"dim_feedforward:      {self.dim_feedforward}\n"
            f"dropout:              {self.dropout}\n"
            f"activation:           {self.activation}\n"
            f"num_denoising:        {self.num_denoising}\n"
            f"label_noise_ratio:    {self.label_noise_ratio}\n"
            f"box_noise_scale:      {self.box_noise_scale}\n"
            f"eval_spatial_size:    {self.eval_spatial_size}\n"
            f"eval_idx:             {self.eval_idx}\n"
            f"eps:                  {self.eps}\n"
            f"aux_loss:             {self.aux_loss}\n"
            f"cross_attn_method:    {self.cross_attn_method}\n"
            f"query_select_method:  {self.query_select_method}\n"
            f"reg_max:              {self.reg_max}\n"
            f"reg_scale:            {self.reg_scale}\n"
        )

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels, self.hidden_dim, 1, bias=False
                                    ),
                                ),
                                (
                                    "norm",
                                    nn.BatchNorm2d(
                                        self.hidden_dim,
                                    ),
                                ),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels,
                                        self.hidden_dim,
                                        3,
                                        2,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(
        self, feats: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[list]]:
        """Prepare encoder inputs for next processing

        Args:
            feats (list[torch.Tensor]): multi-scale feature maps

        Returns:
            tuple[torch.Tensor, list[list]]:
                feat_flatten - flattened multi-scale feature maps\n
                spatial_shapes: multi-scale feature maps size
        """
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(
        self, spatial_shapes=None, dtype=torch.float32, device="cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated anchors for input multi-scale feature maps

        Args:
            spatial_shapes (list[tuple], optional): multi-scale feature maps size. Defaults to None.
            dtype (torch.dtype, optional): Memory dtype. Defaults to torch.float32.
            device (str, optional): Memory device. Defaults to "cpu".

        Returns:
            tuple:
                anchors (torch.Tensor): cx, cy, w, h for each scale in feature maps\n
                valid_mask (torch.Tensor): valid w, h size for each scale
        """
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * self.grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: list[tuple],
        denoising_logits: torch.Tensor | None = None,
        denoising_bbox_unact: torch.Tensor | None = None,
    ) -> tuple:
        """Generates decoder input from flattened multi-scale feature maps.
            All shapes for standard tensor input resolution 640x640 \n
        Args:
            memory (torch.Tensor): Flattened memory with shape [B, 8400, C] \n
            spatial_shapes (list[tuple]): features map scale, 80x80, 40x40, 20x20 \n
            denoising_logits (torch.Tensor, optional): _description_. Defaults to None. \n
            denoising_bbox_unact (torch.Tensor, optional): _description_. Defaults to None. \n

        Returns:
            tuple:
                content (torch.Tensor), top K queries from memory [B, num_queries + num_denoising, C] +
                 denoising_logist if provided. Note, denoising part comes first \n
                enc_topk_bbox_unact (torch.Tensor) top K boxes predicted from top K queries \n
                enc_topk_bboxes_list (list), Sigmoid applyed to `enc_topk_bbox_unact` \n
                enc_topk_logits_list (list), Top K preidcted logits [B, num_queries, num_cls] \n

        """

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.num_queries
        )

        enc_topk_bbox_unact: torch.Tensor = (
            self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors
        )

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        content = enc_topk_memory.detach()
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
    ) -> tuple:
        """Select top K queries from Encoder output.
        All shapes for standard tensor input resolution 640x640 \n

        Args:
            memory (torch.Tensor): Flattened memory with shape [B, 8400, C]
            outputs_logits (torch.Tensor): Linear layer over `memory` [B, 8400, num_cls]
            outputs_anchors_unact (torch.Tensor): Anchors from `_generate_anchors`, with shape [B, 8400, 4]
            topk (int): topK quries number

        Returns:
            tuple: Selected topK memory, logits, anchors from input memory
        """
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]),
        )

        topk_logits = (
            outputs_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]),
            )
            if self.training
            else None
        )

        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        ) = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        # decoder
        (
            output,
            out_bboxes,
            out_logits,
            out_corners,
            out_refs,
            pre_bboxes,
            pre_logits,
        ) = self.box_decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
        )

        keypoints = self.pos_layer(output, out_bboxes[-1])

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(
                pre_logits, dn_meta["dn_num_split"], dim=1
            )
            dn_pre_bboxes, pre_bboxes = torch.split(
                pre_bboxes, dn_meta["dn_num_split"], dim=1
            )
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )

            dn_out_corners, out_corners = torch.split(
                out_corners, dn_meta["dn_num_split"], dim=2
            )
            dn_out_refs, out_refs = torch.split(
                out_refs, dn_meta["dn_num_split"], dim=2
            )

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_keypoints": keypoints,
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_keypoints": keypoints,
            }

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list, enc_topk_bboxes_list
            )
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                }
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class, outputs_coord)
        ]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(
                outputs_class, outputs_coord, outputs_corners, outputs_ref
            )
        ]


if __name__ == "__main__":
    # python3 -m mlcore.models.dfine.decoder_pos
    sslist = [[80, 80], [40, 40], [20, 20]]
    feats = [torch.randn(2, 256, *ss) for ss in sslist]

    layer = DFINETransformer(feat_channels=[256, 256, 256], num_denoising=0).eval()

    out = layer(feats)
    print(out.keys())
