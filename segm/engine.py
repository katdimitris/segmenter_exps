import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu

class OCCELoss(nn.Module):
    def __init__(self, ignore_index=None):
        super(OCCELoss, self).__init__()
        self.ignore_index=ignore_index

    def forward(self, inputs, targets):
        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(-1,C) # shape: (N*H*W,C)
        targets = targets.reshape(-1)

        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0)
        inputs = inputs[valid_mask].squeeze()
        targets = targets[valid_mask].squeeze()

        # multiply with N-1 for numerical stability, does not affect gradient
        ycomp = (C - 1) * F.softmax(-inputs, dim=1)
        y = torch.ones_like(ycomp, device=inputs.device)
        y.scatter_(1, targets.unsqueeze(1), 0.0)
        loss = - 1 / (C - 1) * torch.sum(y * torch.log(ycomp + 0.0000001), dim=1)

        return torch.mean(loss)


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    occe_criterion = OCCELoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for i, batch in enumerate(logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        with amp_autocast():
            seg_pred = model.forward(im)
            loss_ce = criterion(seg_pred, seg_gt)
            loss_occe = occe_criterion(seg_pred, seg_gt)
        loss = loss_ce + loss_occe
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)


        if i % 4 == 0:
            if loss_scaler is not None:
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                )
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            num_updates += 1
            lr_scheduler.step_update(num_updates=num_updates)

            torch.cuda.synchronize()

            logger.update(
                loss=loss.item(),
                loss_ce=loss_ce.item(),
                loss_occe=loss_occe.item(),
                learning_rate=optimizer.param_groups[0]["lr"],
            )

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    for i, batch in enumerate(logger.log_every(data_loader, print_freq, header)):
        if i > 100:
            break
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
