import torch
import torch.nn as nn
import numpy as np
class BasicBlock(nn.Module):
    def __init__(self, inplane, outplane, upsample=False):
        super(BasicBlock,self).__init__()
        if upsample:
            self.conv1 = nn.ConvTranspose3d( inplane, outplane,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        else:
            self.conv1 = nn.Conv3d( inplane, outplane,
                    kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d( outplane, outplane,
                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(outplane)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class BottomLevel(nn.Module):
    def __init__(self, inplane=128):
        super(BottomLevel,self).__init__()
        self.upsample = BasicBlock(inplane, inplane, upsample=True)
        self.block1 = BasicBlock(1, inplane)
        self.block2 = BasicBlock(inplane*2, 64)
        self.block3 = BasicBlock(64, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, TSDF, prev):
        assert prev.shape[2]==16
        assert TSDF.shape[2]==32
        x = self.upsample(prev)
        y = self.block1(TSDF)
        y = torch.cat((x,y),dim=1)
        y = self.block2(y)
        y = self.block3(y)
        pred = self.softmax(y)
        assert pred.shape==(1,3,32,32,32)
        return pred

class TopLevel(nn.Module):
    def __init__(self, inplane, outplane, sub_level):
        super(Level,self).__init__()
        self.sub_level=sub_level
        self.block1 = BasicBlock(inplane, outplane)

    def forward(TSDF, prev):
        assert np.all(prev.shape[1]==16)
        assert np.all(TSDF.shape[1])>=32
        assert TSDF.shape[1]>=64 or self.sub_level is None
        x = self.upsample(prev)
        tsdf = downsample(TSDF,32)
        y = self.tsdf_block(tsdf)
        out = torch.cat(x,y,dim=1)
        out = self.block1(out)
        assert out.shape==tsdf.shape
        pred = self.pred(out)
        assert np.all(pred.shape[1:]==2) or self.sub_level is None
        ret = torch.zeros_like(TSDF)
        half = ret.shape[1]/2
        subrange_ret = [range(half), range(half,2*half)]
        half = out.shape[1]/2
        subrange_out = [range(half), range(half,half*2)]
        for x,y,z in itertools.combinations(range(2),3):
            if pred[0,x,y,z]==1:
                ret[subrange_ret[x],subrange_ret[y],subrange_ret[z]]=self.sub_level(out[subrange_out[x],subrange_out[y],subrange_out[z]])
            else:
                ret[subrange_ret[x],subrange_ret[y],subrange_ret[z]]=pred[0,x,y,z]

class MidLevel(nn.Module):
    def __init__(self, inplane, outplane, sub_level):
        super(Level,self).__init__()
        self.sub_level=sub_level
        self.block1 = BasicBlock(inplane, outplane)

    def forward(TSDF, prev):
        assert np.all(prev.shape[1]==16)
        assert np.all(TSDF.shape[1])>=32
        assert TSDF.shape[1]>=64 or self.sub_level is None
        x = self.upsample(prev)
        tsdf = downsample(TSDF,32)
        y = self.tsdf_block(tsdf)
        out = torch.cat(x,y,dim=1)
        out = self.block1(out)
        pred = self.pred(out)
        assert np.all(pred.shape[1:]==2) or self.sub_level is None
        ret = torch.zeros_like(TSDF)
        half = ret.shape[1]/2
        subrange_ret = [range(half), range(half,2*half)]
        half = out.shape[1]/2
        subrange_out = [range(half), range(half,half*2)]
        for x,y,z in itertools.combinations(range(2),3):
            if pred[0,x,y,z]==1:
                ret[subrange_ret[x],subrange_ret[y],subrange_ret[z]]=self.sub_level(out[subrange_out[x],subrange_out[y],subrange_out[z]])
            else:
                ret[subrange_ret[x],subrange_ret[y],subrange_ret[z]]=pred[0,x,y,z]
