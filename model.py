import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class BasicBlock(nn.Module):
    def __init__(self, inplane, outplane, upsample=False):
        super(BasicBlock,self).__init__()
        if upsample:
            self.conv1 = nn.ConvTranspose3d( inplane, outplane,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        else:
            self.conv1 = nn.Sequential(nn.ReplicationPad3d(1), 
                    nn.Conv3d( inplane, outplane, kernel_size=3, stride=1,
                        padding=0, bias=False))
        self.pad = nn.ReplicationPad3d(2)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d( outplane, outplane, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(outplane)
        self.conv3 = nn.Conv3d( outplane, outplane, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(outplane)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        assert out.shape[-1] == x.shape[-1] or out.shape[-1]==x.shape[-1]*2, (x.shape, out.shape)
        return out

class PredictionBlock(nn.Module):
    '''output prediction for each of the 8 subsections'''
    def __init__(self, inplane, n_class):
        super(PredictionBlock,self).__init__()
        self.conv1 = nn.Conv3d(inplane, inplane, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(inplane)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=4)
        self.conv2 = nn.Conv3d( inplane, inplane/2, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(inplane/2)
        self.conv3 = nn.Conv3d( inplane/2, inplane/4, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(inplane/4)
        self.fc = nn.Linear(inplane*2, n_class) # = inplane/4 * 2^3
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = out.view(out.size(0),-1) #Flatten
        #TODO add dropout? 
        out = self.fc(out)
        out = self.softmax(out)
        return out

class BottomLevel(nn.Module):
    def __init__(self, inplane=128):
        super(BottomLevel,self).__init__()
        self.upsample = BasicBlock(inplane, inplane, upsample=True)
        self.tsdf_bloc = BasicBlock(1, inplane)
        self.block1 = BasicBlock(inplane*2, inplane)
        self.block2 = BasicBlock(inplane, inplane)
        self.pred = BasicBlock(inplane, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, TSDF, prev):
        x = self.upsample(prev)
        y = self.tsdf_bloc(TSDF)
        y = torch.cat((x,y),dim=1)
        y = self.block1(y)
        y = self.block2(y)
        y = self.pred(y)
        pred = self.softmax(y)
        return pred

class MidLevel(nn.Module):
    def __init__(self, tsdf_res, inplane, outplane, sub_level, thresh=0.1):
        super(MidLevel,self).__init__()
        self.sub_level=sub_level
        self.upsample = BasicBlock(inplane, inplane, upsample=True)
        self.tsdf_block = BasicBlock(1, inplane)
        self.block1 = BasicBlock(inplane*2, inplane)
        self.block2 = BasicBlock(inplane, outplane)
        #self.pred = PredictionBlock(outplane, 3)
        #self.dummy = nn.Conv3d( 1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.thresh = thresh
        self.down = tsdf_res/32
        self.tsdf_down = nn.AvgPool3d(kernel_size=self.down)

    def forward(self,TSDF, prev):
        x = self.upsample(prev)
        tsdf = self.tsdf_down(TSDF)
        y = self.tsdf_block(tsdf)
        out = torch.cat((x,y),dim=1)
        out = self.block1(out)
        out = self.block2(out)
        #ret = self.dummy(TSDF)
        #ret = torch.zeros(1,3,TSDF.shape[2], TSDF.shape[3], TSDF.shape[4], requires_grad=True)
        #pred = self.pred(out)
        #if pred[0][1]<0.1 : #self.thresh:
        #    ret = pred[:,:,None,None,None]
        #    return ret
        half = TSDF.shape[-1]/2
        assert out.shape[-1]==32
        assert len(torch.split(TSDF,half,2))==2

        # weve got to split along each dim and then cat them back. 
        # I wish I know a more readable way of doing this
        ret = torch.cat(
                [torch.cat([
                    torch.cat( [
                        self.sub_level(tsdf_z, out_z)
                        for tsdf_z, out_z in zip(torch.split(tsdf_y,half,4), torch.split(out_y,16,4))] ,4)
                    for tsdf_y, out_y in zip(torch.split(tsdf_x,half,3), torch.split(out_x,16,3))],3)
                    for tsdf_x, out_x in zip(torch.split(TSDF,half,2), torch.split(out,16,2))],2)

        return ret
