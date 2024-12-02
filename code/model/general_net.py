from model.implicit_neural.implicit_neural_model import ImplicitNeural
from model.self_distillation.self_distillation_model import SelfDistillation
from model.initseg_mrin import SegNet, MultiResolutionNet

import torch
import torch.nn.functional as F
import copy

from model.conv_block import BasicBlock, Bottleneck, AdaptBlock
from model.position_encoding import PositionEmbeddingSine
from typing import Optional
from torch import nn, Tensor


class BasicBlock_upsample(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1):
        super(BasicBlock_upsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=inplanes,
            out_channels=inplanes,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(inplanes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.sigmod(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos, src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask, pos=pos, src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = tgt
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt,
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2, key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck, 'ADAPTIVE': AdaptBlock}

class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()

        d_model = 384
        dim_feedforward = 384
        encoder_layers_num = 3
        decoder_layers_num = 2
        n_head = 4

        self.patch_encoder = nn.Sequential(
            nn.Conv2d(18, 384, kernel_size=16, stride=16, bias=False),
            nn.BatchNorm2d(384, momentum=0.1),
            nn.ReLU(inplace=True))

        self.patch_decoder = nn.Sequential(
            nn.Conv2d(1, 384, kernel_size=16, stride=16, bias=False),
            nn.BatchNorm2d(384, momentum=0.1),
            nn.ReLU(inplace=True))

        # Attention
        self.pos_embed = PositionEmbeddingSine(d_model=384)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, activation='relu')
        self.global_encoder = TransformerEncoder(encoder_layer, encoder_layers_num)

        decoder_layer = TransformerDecoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=0.1, activation='relu')
        self.global_decoder = TransformerDecoder(decoder_layer, decoder_layers_num)

        # deconvolution layers
        self.deconv = nn.Sequential(BasicBlock_upsample(d_model))


    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, input_en, input_de):

        input_encoder = input_en

        input_encoder = self.patch_encoder(input_encoder)

        bs, c, h, w = input_encoder.shape

        pos_embedding = self.pos_embed(input_encoder).permute(2, 0, 1)

        mask = input_encoder.new_ones((bs, h, w))

        mask[:, :h, :w] = 0
        mask = F.interpolate(mask[None], size=input_encoder.shape[-2:]).to(torch.bool).squeeze(0)
        mask = mask.flatten(1)

        input_encoder = input_encoder.flatten(2).permute(2, 0, 1)
        output_encoder = self.global_encoder(input_encoder, src_key_padding_mask=mask, pos=pos_embedding)
        input_decoder = input_de
        input_decoder = self.patch_decoder(input_decoder)
        input_decoder = input_decoder.flatten(2).permute(2, 0, 1)
        output_decoder = self.global_decoder(input_decoder, output_encoder, memory_key_padding_mask=mask, pos=pos_embedding)
        x = output_decoder[0].permute(1, 2, 0).view(bs, c, h, w)

        return x


class General_Net(nn.Module):
    def __init__(self):
        super(General_Net, self).__init__()
        self.initseg = SegNet()
        self.implicit_neural = ImplicitNeural("../model/model_implicit_neural.pth")
        self.self_distillation = SelfDistillation("../model/model_self_distillation.pth")
        self.vit_model = Trans()
        self.multi_resolution = MultiResolutionNet()

        # 冻结权重
        self.implicit_neural.requires_grad_(False)
        self.self_distillation.requires_grad_(False)

    def forward(self, img):

        init_seg = self.initseg(img)
        # 不计算梯度
        with torch.no_grad():
            implicit = self.implicit_neural(img)
            implicit_split = implicit.unfold(2, 512, 512).unfold(3, 512, 512).reshape(img.shape[0], -1, 512, 512)
            distillation = self.self_distillation(img)

        encoder_input = torch.cat([implicit_split, distillation], dim=1)
        decoder_input = init_seg

        trans_output = self.vit_model(encoder_input, decoder_input)
        final_seg = self.multi_resolution(implicit_split, trans_output)

        return init_seg, implicit, implicit_split, distillation, final_seg

def make_model():
    return General_Net()
