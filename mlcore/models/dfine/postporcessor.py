"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from mlcore.core import COMPONENTS, ComponentType, Task

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@COMPONENTS.register(
    name="DfinePostprocessor",
    ctype=ComponentType.MODEL,
    supported_tasks=Task.OBJECT_DETECTION,
)
class DFINEPostProcessor(nn.Module):
    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(
        self, outputs: dict[str, torch.Tensor], orig_target_sizes: torch.Tensor
    ):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )

        # Legacy from original DETR, modern aproached such as DFine use focal loss by default.
        # Also focall loss not required additional class for background
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self


class DFINEPosePostPorcesor(nn.Module):
    def __init__(
        self,
        num_kps=17,
        num_person=100,
    ) -> None:
        super().__init__()
        self.num_kps = num_kps
        self.num_person = num_person

        self.deploy_mode = False

    def forward(
        self, outputs: dict[str, torch.Tensor], orig_target_sizes: torch.Tensor
    ):
        logits, boxes, keypoints = (
            outputs["pred_logits"],
            outputs["pred_boxes"],
            outputs["pred_keypoints"],
        )

        # [B, 1, 2]
        scale = orig_target_sizes.unsqueeze(1)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred = bbox_pred * scale.repeat(1, 1, 2)
        keypoints_pred = keypoints[:, :, :, :2] * scale.repeat(
            1, self.num_kps, 1
        ).unsqueeze(1)

        scores = F.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.num_person, dim=-1)

        boxes = bbox_pred.gather(
            dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
        )
        print(index.shape)
        keypoints = keypoints_pred.gather(dim=1, index=index.unsqueeze())

        # TODO for onnx export
        if self.deploy_mode:
            return scores, boxes, keypoints_pred

        # TODO доделать с индексом
        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results
