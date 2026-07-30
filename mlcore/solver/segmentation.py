import torch

import logging
from torch.nn.utils import clip_grad_norm_
from ._solver import BaseSolver, metrics

logger = logging.getLogger()


class SegmentationSolver(BaseSolver):
    @torch.no_grad()
    def _val_epoch(self) -> metrics.ValMetricsOutput:
        self._model.eval()
        self._metrics_tracker.reset()

        loss = 0.0
        for i, (img, mask) in enumerate(self._val_dataloader):
            img, mask = img.to(self.device), mask.to(self.device)
            mask = mask.squeeze(1)
            preds = self._model(img)
            upsampled = self._upscale_pred(preds, mask)

            loss += self._criterion(upsampled, mask).item()

            gt = mask.cpu().numpy()
            predict = upsampled.argmax(dim=1).cpu().numpy()

            self._metrics_tracker.update(predict, gt)

        self._metrics_tracker.compute()

        return metrics.ValMetricsOutput(
            loss=float(loss),
            average_metrics=self._metrics_tracker.average(),
            per_cls_metrics=self._metrics_tracker.per_cls_metrics(),
        )

    def _train_epoch(self):
        self._model.train()
        train_loss = 0.0
        for i, (img, mask) in enumerate(self._train_dataloader):
            global_step = self._last_epoch * len(self._train_dataloader) + i

            self._optim.zero_grad()
            img, mask = img.to(self.device), mask.to(self.device)
            mask = mask.squeeze(1)

            preds = self._model(img)
            upsampled = self._upscale_pred(preds, mask)

            loss = self._criterion(upsampled, mask)
            loss.backward()

            if self.grad_clip:
                clip_grad_norm_(self._model.parameters(), max_norm=self.grad_clip)
            self._optim.step()

            train_loss += loss.item()
            if self._scheduler:
                self._scheduler.step()

            if self.log_time(global_step):
                logger.info(f"Train step: {global_step}, loss: {loss / (i +1):4.4f}")

        return train_loss / (i + 1)
