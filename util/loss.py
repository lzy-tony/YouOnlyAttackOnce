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

    def __call__(self, p, targets=[2, 5, 7]):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        cnt = 0.0
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image, anchor, gridy, gridx

            bs = pi.shape[0]  # batch size
            pi = pi.view(-1, pi.shape[-1])
            best_class = pi[...,5:].max(dim=1).values
            mask = torch.zeros(pi.shape[0],device=self.device)
            for w in targets:
                m = (best_class==pi[...,5+w]).float()
                # lcls += (torch.sigmoid(pi[...,5+w])*m).sum()
                mask += m
            conf = torch.sigmoid(pi[...,4])
            lobj += (mask*conf).sum()
            # cnt += float(mask.sum())
        
        # print("--yolov3 lobj: ", lobj)

        return lobj, lcls


class Faster_RCNN_loss:
    def __init__(self):
        self.device = "cuda"

    def __call__(self, _bboxes, _labels, _scores, targets=[5, 6]):
        l = torch.zeros(1, device=self.device)
        for i, labels in enumerate(_labels):
            # for j, label in enumerate(labels):
            #     if label == 5 or label == 6:
            #         l += _scores[i][j]
            for w in targets:
                m = (labels==w).float()
                l += (_scores[i] * m).sum()
        l *= 10
        print("--frcnn loss: ", l)    
        return l


class AttentionTransferLoss:
    def __init__(self):
        self.device = "cuda"

        im_height, im_width = 384, 640
        h1, h2, w1, w2 = 100, 300, 150, 500
        m = torch.ones((1, h2 - h1, w2 - w1))
        p2d = (w1, im_width - w2, h1, im_height - h2)
        mask = F.pad(m, p2d, "constant", -0.1)
        self.mask = mask.to(self.device)
    
    def __call__(self, grayscale_cam):
        loss = (grayscale_cam * self.mask).sum()
        return loss


class Faster_RCNN_COCO_loss:
    def __call__(self, result, targets=[2, 5, 7]):
        l = torch.zeros(1, device="cuda")
        for t in targets:
            for i in range(len(result[t])):
                l += result[t][i, 4]
        print("-faster r-cnn loss: ", l)
        return l


class TORCH_VISION_LOSS:
    def __call__(self, outputs, detection_threshold=0.01):
        l = torch.zeros(1, device="cuda")

        for index in range(len(outputs[0]['scores'])):
            if outputs[0]['scores'][index] >= detection_threshold and \
                (outputs[0]['labels'][index] == 3 or 
                 outputs[0]['labels'][index] == 6 or 
                 outputs[0]['labels'][index] == 8):
                l += outputs[0]['scores'][index]
        # print("-TORCH_VISION CONF LOSS: ", l)
        return l

class TV_loss:
    def __call__(self, patch):
        h = patch.shape[-2]
        w = patch.shape[-1]
        h_tv = torch.pow((patch[..., 1:, :] - patch[..., :h - 1, :]), 2).sum()
        w_tv = torch.pow((patch[..., 1:] - patch[..., :w - 1]), 2).sum()
        # h_tv = torch.pow((patch[..., 1:, int(w/2):-1] - patch[..., :h - 1, int(w/2):-1]), 2).sum()
        # w_tv = torch.pow((patch[..., int(w/2) + 1:] - patch[..., int(w/2):w - 1]), 2).sum()
        return h_tv + w_tv


class TV_loss_left:
    def __call__(self, patch):
        h = patch.shape[-2]
        w = patch.shape[-1]
        # h_tv = torch.pow((patch[..., 1:, :] - patch[..., :h - 1, :]), 2).sum()
        # w_tv = torch.pow((patch[..., 1:] - patch[..., :w - 1]), 2).sum()
        h_tv = torch.pow((patch[..., 1:, 0:int(w/2)] - patch[..., :h - 1, 0:int(w/2)]), 2).sum()
        w_tv = torch.pow((patch[..., 1:int(w/2)] - patch[..., :int(w/2) - 1]), 2).sum()
        return h_tv + w_tv


class TV_loss_right:
    def __call__(self, patch):
        h = patch.shape[-2]
        w = patch.shape[-1]
        # h_tv = torch.pow((patch[..., 1:, :] - patch[..., :h - 1, :]), 2).sum()
        # w_tv = torch.pow((patch[..., 1:] - patch[..., :w - 1]), 2).sum()
        h_tv = torch.pow((patch[..., 1:, int(w/2):-1] - patch[..., :h - 1, int(w/2):-1]), 2).sum()
        w_tv = torch.pow((patch[..., int(w/2) + 1:] - patch[..., int(w/2):w - 1]), 2).sum()
        return h_tv + w_tv