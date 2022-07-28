import torch
from torch import nn


class VanillaLoss:
    def __init__(self, model):
        h = model.hyp  # hyperparameters
        # Define criteria
        self.device = next(model.parameters()).device
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=self.device))

        self.balance = {3: [4.0, 1.0, 0.4]}.get(269, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7

    def __call__(self, p, targets):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)
        # lcls = torch.zeros(1, device=self.device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image, anchor, gridy, gridx

            tobj = torch.zeros_like(pi[..., 0], device=self.device)
            lobj += self.BCEobj(pi[..., 4], tobj) * self.balance[i]

        bs = tobj.shape[0]  # batch size
        lobj *= 10
        print(lobj * bs)
        return lobj * bs


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
        return lobj * 100 + lcls
