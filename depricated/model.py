import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, inplane, outplane, upsample=False, bn=True):
        super(BasicBlock,self).__init__()
        modules = []
        if upsample:
            modules.append(nn.ConvTranspose3d( inplane, outplane,
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                    bias=False))
        else:
            modules.append(nn.Sequential(nn.ReplicationPad3d(1),
                nn.Conv3d( inplane, outplane, kernel_size=3, stride=1,
                        padding=0, bias=False)))
        modules.append( nn.ReplicationPad3d(1))
        if bn:
            modules.append( nn.BatchNorm3d(outplane))
        modules.append( nn.ReLU(inplace=True))
        modules.append( nn.Conv3d( outplane, outplane, kernel_size=3, stride=1,
            padding=0, bias=False))
        if bn:
            modules.append( nn.BatchNorm3d(outplane))
        modules.append( nn.ReLU(inplace=True))
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        out = self.module(x)
        assert out.shape[-1] == x.shape[-1] or out.shape[-1]==x.shape[-1]*2, (x.shape, out.shape)
        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Debug(nn.Module):
    def __init__(self, str=""):
        super(Debug,self).__init__()
        self.str = str
    def forward(self, input):
        #import ipdb; ipdb.set_trace()
        print self.str, input.shape
        return input

class PredictionBlock(nn.Module):
    '''output prediction for this block  '''
    def __init__(self, inplane, n_class, bn=True):
        super(PredictionBlock,self).__init__()
        modules = []
        modules.append( nn.Conv3d(inplane, inplane/2, kernel_size=3, stride=1,
            padding=0, bias=False))
        modules.append( nn.BatchNorm3d(inplane/2))
        modules.append( nn.ReLU(inplace=True))
        modules.append( nn.MaxPool3d(kernel_size=3))
        modules.append( nn.Conv3d( inplane/2, inplane/4, kernel_size=3, stride=1,
            padding=0, bias=False))
        modules.append( nn.BatchNorm3d(inplane/4))
        modules.append( nn.ReLU(inplace=True))
        modules.append( nn.MaxPool3d(kernel_size=4))
        modules.append( Flatten())
        modules.append( nn.Linear(inplane*2, n_class)) # = inplane/4 * 2^3
        #add dropout? 
        modules.append( nn.Softmax(dim=1))
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        out = self.module(x)
        return out

class BottomLevel(nn.Module):
    def __init__(self, inplane=128):
        super(BottomLevel,self).__init__()
        self.upsample = BasicBlock(inplane, inplane, upsample=True)
        self.tsdf_bloc = BasicBlock(1, inplane)
        self.block1 = BasicBlock(inplane*2, inplane)
        self.block2 = BasicBlock(inplane, inplane)
        self.pred = BasicBlock(inplane, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, TSDF, prev):
        x = self.upsample(prev)
        y = self.tsdf_bloc(TSDF)
        y = torch.cat((x,y),dim=1)
        y = self.block1(y)
        y = self.block2(y)
        y = self.pred(y)
        pred = self.softmax(y)
        
        #the cost of the compute for this block is one. 
        tsdf_res = TSDF.shape[-1]
        compute = torch.ones(1,1,tsdf_res, tsdf_res,tsdf_res,
                requires_grad=True).cuda()/ tsdf_res**3
        pred = torch.cat((pred, compute),1)
        return pred

class MidLevel(nn.Module):
    def __init__(self, tsdf_res, inplane, outplane, sub_level, thresh=0.1):
        super(MidLevel,self).__init__()
        self.tsdf_res = tsdf_res
        self.sub_level=sub_level
        self.upsample = BasicBlock(inplane, inplane, upsample=True)
        self.tsdf_block = BasicBlock(1, 16)
        self.block1 = BasicBlock(inplane+16, outplane)
        self.pred = PredictionBlock(outplane, 3)
        self.thresh = thresh
        self.tsdf_down = nn.AvgPool3d(kernel_size=tsdf_res/32)

    def forward(self,TSDF, prev):
        assert TSDF.shape[-1] == self.tsdf_res
        x = self.upsample(prev)
        tsdf = self.tsdf_down(TSDF)
        y = self.tsdf_block(tsdf)
        out = torch.cat((x,y),dim=1)
        out = self.block1(out)
        pred = self.pred(out)
        if pred[0,-1]< self.thresh:
            ret = pred[None,:-2,None,None,None].repeat(1,1,self.tsdf_res,self.tsdf_res,self.tsdf_res)
            #the cost of the compute for this block is one. 
            cost = torch.ones(1,1,self.tsdf_res, self.tsdf_res,self.tsdf_res,
                    requires_grad=True).cuda()/ self.tsdf_res**3
            ret = torch.cat((ret, cost),2)

            return ret
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

class TopLevel(nn.Module):
    def __init__(self, tsdf_res, outplane, sub_level, thresh=0.1):
        super(TopLevel,self).__init__()
        self.tsdf_res = tsdf_res
        self.sub_level=sub_level
        self.tsdf_block = BasicBlock(1, 16)
        self.block1 = BasicBlock(16, outplane)
        self.pred = PredictionBlock(outplane, 3)
        self.thresh = thresh
        self.tsdf_down = nn.AvgPool3d(kernel_size=tsdf_res/32)

    def forward(self,TSDF):
        assert TSDF.shape[-1] == self.tsdf_res
        tsdf = self.tsdf_down(TSDF)
        y = self.tsdf_block(tsdf)
        out = self.block1(y)
        pred = self.pred(out)
        if pred[0,-1]< self.thresh:
            import ipdb; ipdb.set_trace()
            ret = pred[:,:-1,None,None,None].repeat(1,1,self.tsdf_res,self.tsdf_res,self.tsdf_res)
            #the cost of the compute for this block is one. 
            cost = torch.ones(1,1,self.tsdf_res, self.tsdf_res,self.tsdf_res,
                    requires_grad=True).cuda()/ self.tsdf_res**3
            ret = torch.cat((ret, cost),1)

            return ret
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
