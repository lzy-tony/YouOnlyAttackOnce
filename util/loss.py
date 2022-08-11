import torch
from torch import nn
from torch.nn import functional as F


class VanillaLoss:
    def __init__(self, model):
        h = model.hyp  # hyperparameters
        # Define criteria
        self.device = next(model.parameters()).device
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=self.device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=self.device))

        self.balance = {3: [4.0, 1.0, 0.4]}.get(269, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7

    def __call__(self, p, targets):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image, anchor, gridy, gridx

            tobj = torch.zeros_like(pi[..., 0], device=self.device)
            tcls = torch.zeros_like(pi[..., 5:], device=self.device)
            lobj += self.BCEobj(pi[..., 4], tobj)
            lcls += self.BCEcls(pi[..., 5:], tcls)

        bs = tobj.shape[0]  # batch size
        lcls *= 10
        lobj *= 10
        print("lcls: ", lcls)
        print("lobj: ", lobj)
        print((lcls + lobj) * bs)
        return (lcls + lobj) * bs


class OriginalLoss:
    def __init__(self, model):
        self.device = next(model.parameters()).device

    def __call__(self, p, targets):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        cnt = 0
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image, anchor, gridy, gridx

            pi = pi.view(-1, pi.shape[-1])
            for a in range(pi.shape[0]):
                p_t = pi[a, 5:].clone().cpu().detach().numpy()
                if p_t.argmax() in targets:
                    cnt += 1
                    conf = pi[a, 4].clone()
                    prob = pi[a, 5 + p_t.argmax()].clone()
                    lobj += torch.sigmoid(conf)
                    lcls += torch.sigmoid(prob)
        print(cnt)
        print(lobj)
        print(lcls)
        return lobj * 100


class Original_loss_gpu:
    def __init__(self, model):
        self.device = next(model.parameters()).device

    def __call__(self, p, targets):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        cnt = 0.0
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image, anchor, gridy, gridx

            pi = pi.view(-1, pi.shape[-1])
            best_class = pi[...,5:].max(dim=1).values
            mask = torch.zeros(pi.shape[0],device=self.device)
            for w in targets:
                m = (best_class==pi[...,5+w]).float()
                lcls += (torch.sigmoid(pi[...,5+w])*m).sum()
                mask += m
            conf = torch.sigmoid(pi[...,4])
            lobj += (mask*conf).sum()
            cnt += float(mask.sum())
        return lobj * 100, cnt

class Faster_RCNN_loss:
    def __init__(self):
        self.device = "cuda"

    def __call__(self, _bboxes, _labels, _scores):
        l = torch.zeros(1, device=self.device)
        for i, labels in enumerate(_labels):
            for j, label in enumerate(labels):
                if label == 5 or label == 6:
                    l += _scores[i][j]
        
        return l


class AttentionTransferLoss:
    def __init__(self):
        self.device = "cuda"

        im_height, im_width = 384, 640
        h1, h2, w1, w2 = 150, 300, 150, 500
        m = torch.ones((1, h2 - h1, w2 - w1))
        p2d = (w1, im_width - w2, h1, im_height - h2)
        mask = F.pad(m, p2d, "constant", -0.1)
        self.mask = mask.to(self.device)
    
    def __call__(self, grayscale_cam):
        loss = (grayscale_cam * self.mask).sum()
        print(loss)
        return loss