'''使用这个文件进行封装'''

import os
from PIL import Image

import torch
from torchvision import transforms
from model.implicit_neural.make_models import make  # 正式训练的时候用这个
import torch.nn as nn

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


class ImplicitNeural(nn.Module):
    def __init__(self, model_path):
        super(ImplicitNeural, self).__init__()
        self.path = model_path

    def forward(self, img):
        model = make(torch.load(self.path)['model'], load_sd=True).cuda()
        h, w = [img.shape[2] * 2, img.shape[3] * 2]
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        pred_all = torch.zeros((img.shape[0], 3, h, w)).cuda()
        for i in range(img.shape[0]):
            pred_line = \
            batched_predict(model, ((img[i] - 0.5) / 0.5).cuda().unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0),
                            bsize=30000)[0]
            pred = (pred_line * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1)
            pred_all[i] = pred

        return pred_all


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    img_path = '0_Raw.png'
    model_path = "/home/tian/code_gujia/gujia2_general_model/model/model_implicit_neural.pth"
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))  # img.max() = tensor(1.)
    model = make(torch.load(model_path)['model'], load_sd=True).cuda()
    h, w = [img.shape[1]*2, img.shape[2]*2]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save('./0_out.png')

