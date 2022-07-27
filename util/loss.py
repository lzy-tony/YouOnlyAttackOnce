import torch


class VanillaLoss:
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
        return lobj * 1000 + lcls