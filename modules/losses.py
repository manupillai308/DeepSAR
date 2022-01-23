import torch
from modules.utils import contour_extract
import numpy as np

def loss_fn_rpn(binary_pred, cls_true):
    
    losses = torch.nn.functional.binary_cross_entropy(binary_pred, cls_true, reduction='none')
    label_arr = cls_true.squeeze(1).detach().cpu().numpy()
    
    b, bg_i, bg_j = np.where(label_arr == 0)
    b, fg_i, fg_j = np.where(label_arr == 1)
    
    fg_l = len(fg_i)
    bg_l = len(bg_i)
    ixs = np.random.choice(np.arange(bg_l), fg_l)
    
    fg_loss = losses[b, :, fg_i, fg_j].mean()
    bg_loss = losses[b, :, bg_i[ixs], bg_j[ixs]].mean()
    
    return fg_loss + bg_loss


def loss_fn_dn(proposal_t, cls_pred, cls_true, class_weight=None, thresholds=[0.33, 0.67, 0.95], proposal_extract_fn=contour_extract):
        
    indexes = proposal_extract_fn(proposal_t, thresholds)
    cls, cnt = np.unique(cls_true.cpu().numpy(), return_counts=True)
    cnts = [10]
    for i in range(1, cls_pred.shape[1]):
        c = cnt[cls == i]
        if len(c) > 0:
            cnts.append(c*10)
        else:
            cnts.append(10)
        

    b, i, j = indexes[0]
    class_weight = torch.Tensor([0.1, len(i)/cnts[1], len(i)/cnts[2], len(i)/cnts[3]]).to(cls_pred.device)
    losses = torch.nn.functional.nll_loss(torch.log(cls_pred[b, :, i, j]), cls_true[b, i, j].long(), 
                                          weight=class_weight, reduction='none')
    indx_loss = losses.mean()

    for (b, i, j) in indexes[1:]:

        class_weight = torch.Tensor([0.1, len(i)/cnts[1], len(i)/cnts[2], len(i)/cnts[3]]).to(cls_pred.device)
        losses = torch.nn.functional.nll_loss(torch.log(cls_pred[b, :, i, j]), cls_true[b, i, j].long(),
                                              weight=class_weight, reduction='none')
        indx_loss += losses.mean()
        
    return indx_loss

def loss_fn_dn_2(cls_pred, cls_true):
    bp, ip, jp = np.where(np.argmax(cls_pred.detach().cpu().numpy(), axis=1) != 0)
    bt, it, jt = np.where(cls_true.detach().cpu().numpy() != 0)
    
    lp = torch.nn.functional.nll_loss(torch.log(cls_pred[bp, :, ip, jp]), cls_true[bp, ip, jp].long(), reduction='mean')
    lt = torch.nn.functional.nll_loss(torch.log(cls_pred[bt, :, it, jt]), cls_true[bt, it, jt].long(), reduction='mean')
    
    return lp+lt