import torch.nn as nn
import torch

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.conv1 = self.conv_block(3,32)
        self.conv2 = self.conv_block(32,64)
        self.conv3 = self.conv_block(64,128)
        self.conv4 = self.conv_block(128,128*2)
        self.conv5 = self.conv_block(128*2,128*4)
        self.pool = torch.nn.MaxPool2d(2)
        self.upconv1 = self.upconv(64,32)
        self.upconv2 = self.upconv(128,64)
        self.upconv3 = self.upconv(128*2,128)
        self.upconv4 = self.upconv(128*4,128*2)
        self.conv6 = self.conv_block(128*4,128*2)
        self.conv7 = self.conv_block(128*2,128)
        self.conv8 = self.conv_block(128,64)
        self.conv9 = self.conv_block(64,32)
        self.conv11 = self.conv_block(35,1)
        self.last_act = nn.PReLU()

    def conv_block(self, channel_in, channel_out):
        if channel_in==3:
            return nn.Sequential(
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )
        else:
            return nn.Sequential(
                nn.PReLU(),
                nn.BatchNorm2d(channel_in),
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in,channel_out,kernel_size=2,stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)
        u4 = self.upconv4(x5)
        u4 = torch.cat([u4, x4], 1)
        u4 = self.conv6(u4)
        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, x3], 1)
        u3 = self.conv7(u3)
        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, x2], 1)
        u2 = self.conv8(u2)
        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, x1], 1)
        u1 = self.conv9(u1)
        u1 = torch.cat([u1, x], 1)
        out_pred = torch.sigmoid(self.conv11(u1))
        return out_pred

class MultiResolutionNet(nn.Module):
    def __init__(self):
        super(MultiResolutionNet, self).__init__()
        self.inr_conv1 = self.conv_block(12,32)
        self.inr_conv2 = self.conv_block(32,64)
        self.inr_conv3 = self.conv_block(64,128)
        self.inr_conv4 = self.conv_block(128,256)
        self.inr_conv5 = self.conv_block(256,384)
        self.pool = torch.nn.MaxPool2d(2)

        self.trans_conv5 = self.conv_block(384, 384)
        self.trans_upconv4 = self.upconv(384,256)
        self.trans_conv4 = self.conv_block(256, 256)
        self.trans_upconv3 = self.upconv(256, 128)
        self.trans_conv3 = self.conv_block(128, 128)
        self.trans_upconv2 = self.upconv(128, 64)
        self.trans_conv2 = self.conv_block(64, 64)
        self.trans_upconv1 = self.upconv(64, 32)
        self.trans_conv1 = self.conv_block(32, 32)

        self.inr_conv_d5 = self.conv_block(384, 128)
        self.trans_conv_d5 = self.conv_block(384, 128)
        self.inr_conv_d4 = self.conv_block(256, 128)
        self.trans_conv_d4 = self.conv_block(256, 128)
        self.inr_conv_d3 = self.conv_block(128, 64)
        self.trans_conv_d3 = self.conv_block(128, 64)
        self.inr_conv_d2 = self.conv_block(64, 32)
        self.trans_conv_d2 = self.conv_block(64, 32)
        self.inr_conv_d1 = self.conv_block(32, 16)
        self.trans_conv_d1 = self.conv_block(32, 16)

        self.fuse_upconv4 = self.upconv(128*2,128)
        self.fuse_conv4 = self.conv_block(128, 128)
        self.fuse_upconv3 = self.upconv(128*3, 64)
        self.fuse_conv3 = self.conv_block(64, 64)
        self.fuse_upconv2 = self.upconv(64*3, 32)
        self.fuse_conv2 = self.conv_block(32, 32)
        self.fuse_upconv1 = self.upconv(32*3, 16)
        self.fuse_conv1 = self.conv_block(16, 16)
        self.fuse_conv0 = self.conv_block(16*3, 16)

        self.conv_p = self.conv_block(16, 1)

    def conv_block(self, channel_in, channel_out):
        if channel_in==12 or channel_in==384:
            return nn.Sequential(
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )
        else:
            return nn.Sequential(
                nn.PReLU(),
                nn.BatchNorm2d(channel_in),
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in,channel_out,kernel_size=2,stride=2)

    def conv_block_d2(self, channel_in):
        return nn.Sequential(
            nn.PReLU(),
            nn.BatchNorm2d(channel_in),
            nn.Conv2d(channel_in, channel_in/2, 3, 1, 1))

    def conv_block_d3(self, channel_in):
        return nn.Sequential(
            nn.PReLU(),
            nn.BatchNorm2d(channel_in),
            nn.Conv2d(channel_in, channel_in/3, 3, 1, 1))

    def forward(self, inr, trans):
        inr1 = self.inr_conv1(inr)
        inr2 = self.pool(inr1)
        inr2 = self.inr_conv2(inr2)
        inr3 = self.pool(inr2)
        inr3 = self.inr_conv3(inr3)
        inr4 = self.pool(inr3)
        inr4 = self.inr_conv4(inr4)
        inr5 = self.pool(inr4)
        inr5 = self.inr_conv5(inr5)

        trans5 = self.trans_conv5(trans)
        trans4 = self.trans_upconv4(trans5)
        trans4 = self.trans_conv4(trans4)
        trans3 = self.trans_upconv3(trans4)
        trans3 = self.trans_conv3(trans3)
        trans2 = self.trans_upconv2(trans3)
        trans2 = self.trans_conv2(trans2)
        trans1 = self.trans_upconv1(trans2)
        trans1 = self.trans_conv1(trans1)

        inr_d5 = self.inr_conv_d5(inr5)
        inr_d4 = self.inr_conv_d4(inr4)
        inr_d3 = self.inr_conv_d3(inr3)
        inr_d2 = self.inr_conv_d2(inr2)
        inr_d1 = self.inr_conv_d1(inr1)

        trans_d5 = self.trans_conv_d5(trans5)
        trans_d4 = self.trans_conv_d4(trans4)
        trans_d3 = self.trans_conv_d3(trans3)
        trans_d2 = self.trans_conv_d2(trans2)
        trans_d1 = self.trans_conv_d1(trans1)

        fuse_f5 = torch.cat([inr_d5, trans_d5], 1)
        fuse4 = self.fuse_upconv4(fuse_f5)
        fuse4 = self.fuse_conv4(fuse4)
        fuse_f4 = torch.cat([fuse4, inr_d4, trans_d4], 1)
        fuse3 = self.fuse_upconv3(fuse_f4)
        fuse3 = self.fuse_conv3(fuse3)
        fuse_f3 = torch.cat([fuse3, inr_d3, trans_d3], 1)
        fuse2 = self.fuse_upconv2(fuse_f3)
        fuse2 = self.fuse_conv2(fuse2)
        fuse_f2 = torch.cat([fuse2, inr_d2, trans_d2], 1)
        fuse1 = self.fuse_upconv1(fuse_f2)
        fuse1 = self.fuse_conv1(fuse1)

        fuse_f1 = torch.cat([fuse1, inr_d1, trans_d1], 1)
        fuse0 = self.fuse_conv0(fuse_f1)

        out_pred = torch.sigmoid(self.conv_p(fuse0))
        return out_pred