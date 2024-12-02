'''使用这个文件进行封装'''

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
from PIL import Image
import model.self_distillation.vision_transformer as vits

class SelfDistillation(nn.Module):
    def __init__(self, model_path):
        super(SelfDistillation, self).__init__()
        self.path = model_path

    def forward(self, img):
        # build model
        model = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.cuda()
        state_dict = torch.load(self.path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)

        w_featmap = img.shape[-2] // 8
        h_featmap = img.shape[-1] // 8

        attentions = model.get_last_selfattention(img)
        nh = attentions.shape[1] # number of head
        # we keep only the output patch attention
        attentions = attentions[:, :, 0, 1:].reshape(img.shape[0], nh, -1)
        attentions = attentions.reshape(img.shape[0], nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions, scale_factor=8, mode="nearest")
        return attentions




if __name__ == '__main__':
    device = torch.device("cuda")
    # build model
    model = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    state_dict = torch.load("/home/tian/code_gujia/gujia2_general_model/model/model_self_distillation.pth", map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)

    # open image
    img = Image.open("./img.png")
    img = img.convert('RGB')

    transform = pth_transforms.Compose([
        pth_transforms.Resize((512, 512)),
        pth_transforms.ToTensor()])
    img = transform(img)

    # img = torch.rand(10, 3, 1024, 1024)

    print(img.shape)
    print(type(img))
    print(img.max())
    # make the image divisible by the patch size
    # w, h = img.shape[1] - img.shape[1] % 8, img.shape[2] - img.shape[2] % 8
    # img = img[:, :w, :h].unsqueeze(0)
    img = img.unsqueeze(0)

    w_featmap = img.shape[-2] // 8
    h_featmap = img.shape[-1] // 8

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=8, mode="nearest")[0].cpu().numpy()
    print(attentions.shape)
    print(type(attentions))

    # save attentions heatmaps
    os.makedirs('.', exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join('.', "img.png"))
    for j in range(nh):
        fname = os.path.join('.', "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")


