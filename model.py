import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = 'cuda'
class BasicBlock(nn.Module):
    def __init__(self, inplane, outplane, upsample=False, bn=True):
        super(BasicBlock,self).__init__()
        modules = []
        if upsample:
            modules.append(nn.ConvTranspose3d( inplane, outplane,
                    kernel_size=4, stride=2, padding=0, output_padding=0,
                    bias=False))
        else:
            modules.append(nn.Sequential(nn.ReplicationPad3d(2),
                nn.Conv3d( inplane, outplane, kernel_size=3, stride=1,
                        padding=0, bias=False)))
        if bn:
            modules.append( nn.BatchNorm3d(outplane))
        modules.append( nn.ReLU(inplace=True))
        modules.append( nn.Conv3d( outplane, outplane, kernel_size=3, stride=1,
            padding=0, bias=False))
        if bn:
            modules.append( nn.BatchNorm3d(outplane))
        #modules.append( nn.ReLU(inplace=True))
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
        zeros = float(np.count_nonzero(input==0))/input.numel()
        print( self.str, input.shape, zeros)
        #if zeros>0.45:
        #    import ipdb; ipdb.set_trace()
        return input

class PredictionBlock(nn.Module):
    '''output prediction for this block  '''
    def __init__(self, inplane, n_class, bn=True):
        super(PredictionBlock,self).__init__()
        modules = []
        modules.append( nn.Conv3d(inplane, inplane//2, kernel_size=3, stride=1,
            padding=0, bias=False))
        modules.append( nn.BatchNorm3d(inplane//2))
        modules.append( nn.ReLU(inplace=True))
        modules.append( nn.MaxPool3d(kernel_size=3))
        modules.append( nn.Conv3d( inplane//2, inplane//4, kernel_size=3, stride=1,
            padding=0, bias=False))
        modules.append( nn.BatchNorm3d(inplane//4))
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
    def __init__(self, inplane, block_size):
        super(BottomLevel,self).__init__()
        self.upsample = BasicBlock(inplane, inplane, upsample=True)
        self.tsdf_block = BasicBlock(1, inplane)
        self.block1 = BasicBlock(inplane*2, inplane)
        self.block2 = BasicBlock(inplane, inplane)
        self.pred = BasicBlock(inplane, 2)
        self.block_size=block_size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tsdf_pyramid, prev):
        assert len(tsdf_pyramid)==1
        assert tsdf_pyramid[-1].shape[0]==1
        assert type(tsdf_pyramid) == list
        prevs = {X:self.upsample(p) for X,p in prev.items()}
        tsdfs_in = {(x,y,z):
                tsdf_pyramid[-1][:,:,
                    x*self.block_size:(x+1)*self.block_size,
                    y*self.block_size:(y+1)*self.block_size,
                    z*self.block_size:(z+1)*self.block_size]
                for (x,y,z) in prevs.keys()}
        tsdfs = {X:self.tsdf_block(T) for X,T in tsdfs_in.items()}
        out   = {X:torch.cat((prevs[X], tsdfs[X]) ,dim=1) for X in prevs.keys()}
        feat1  = {X:self.block1(T) for X,T in out.items()}
        feat  = {X:self.block2(T) for X,T in feat1.items()}
        pred1  = {X:self.pred(T) for X,T in feat.items()}
        pred  = {X:self.softmax(T) for X,T in pred1.items()}
        return [pred]

class MidLevel(nn.Module):
    def __init__(self, inplane, outplane, sub_level, block_size, thresh=0.1):
        super(MidLevel,self).__init__()
        self.sub_level=sub_level
        self.upsample = BasicBlock(inplane, inplane, upsample=True)
        self.tsdf_block = BasicBlock(1, 16)
        self.block1 = BasicBlock(inplane+16, outplane)
        self.pred = PredictionBlock(outplane, 3)
        self.thresh = thresh
        self.block_size=block_size

    def forward(self,tsdf_pyramid, prev):
        assert type(tsdf_pyramid)==list
        assert type(prev)==dict
        prevs = {X:self.upsample(p) for X,p in prev.items()}
        tsdfs = {(x,y,z): tsdf_pyramid[-1][:,:,
            x*self.block_size:(x+1)*self.block_size,
            y*self.block_size:(y+1)*self.block_size,
            z*self.block_size:(z+1)*self.block_size]
            for (x,y,z) in prevs.keys()}
        tsdfs = {X:self.tsdf_block(T) for X,T in tsdfs.items()}
        out = {X:torch.cat((prevs[X], tsdfs[X]) ,dim=1) for X in prevs.keys()}
        feat = {X:self.block1(T) for X,T in out.items()}
        pred = {X:self.pred(T) for X,T in feat.items()}

        #durring training continue down the octree randomly (sampled toward
        #cells with boundaries
        if self.training:
            p_tot = torch.Tensor([p[0,-1] for p in
                pred.values()]).to(device).sum()
            mixed = [X for X,p in pred.items() if p[0,-1]/p_tot > np.random.rand()]
        else:
            #durring test time continue to refine only boundary cells
            mixed = [X for X,p in pred.items() if p[0,-1] > self.thresh]

        refine = {}
        for x,y,z in mixed:
            refine.update(split_tree(feat[(x,y,z)],x,y,z))
        return self.sub_level(tsdf_pyramid[:-1], refine) +[pred]

def split_tree(feat, parent_x=0, parent_y=0, parent_z=0):
    block_size=feat.shape[-1]
    assert feat.shape[-1] == feat.shape[-2] == feat.shape[-3]
    subtree = {}
    for x,feat_x in enumerate(torch.split(feat,block_size//2,2)):
        for y,feat_y in enumerate(torch.split(feat_x,block_size//2,3)):
            for z,feat_z in enumerate(torch.split(feat_y,block_size//2,4)):
                subtree[(parent_x*2+x,parent_y*2+y,parent_z*2+z)]=feat_z
    return subtree

class TopLevel(nn.Module):
    def __init__(self, outplane, sub_level, block_size):
        super(TopLevel,self).__init__()
        self.sub_level=sub_level
        self.tsdf_block = BasicBlock(1, 16)
        self.block1 = BasicBlock(16, outplane)
        self.pred = PredictionBlock(outplane, 3)
        self.block_size=block_size
        self.tsdf_down = nn.AvgPool3d(kernel_size=2)

    def forward(self,TSDF):
        pyramid = [TSDF]
        while pyramid[-1].shape[-1]>self.block_size:
            down = self.tsdf_down(pyramid[-1])
            pyramid.append(down)
        y = self.tsdf_block(pyramid[-1])
        feat = self.block1(y)
        pred = self.pred(feat)
        #always split top level. No early termination yet
        assert feat.shape[-1]==self.block_size, (feat.shape[-1], self.block_size)

        subtree = split_tree(feat)
        return self.sub_level(pyramid[:-1], subtree) + [{(0,0,0):pred}]


class OctreeCrossEntropyLoss(nn.Module):
    def __init__(self, gt_label, block_size):
        super(OctreeCrossEntropyLoss, self).__init__()
        '''gt_label is the GT signed binary distance function'''
        self.block_size=block_size
        self.max_level = int(np.log2(gt_label.shape[-1]/block_size))
        assert 2**self.max_level*block_size==gt_label.shape[-1]
        self.criteria = nn.CrossEntropyLoss()
        self.gt_octree = [{} for _ in range(self.max_level+1)] #each level is a dictionary
        for level in range(self.max_level,-1,-1):
            bs = block_size*np.power(2,level)
            num_blocks = gt_label.shape[-1]//bs
            for x in range(num_blocks):
                for y in range(num_blocks):
                    for z in range(num_blocks):
                        label = gt_label[:,x*bs:(x+1)*bs,
                                y*bs:(y+1)*bs,
                                z*bs:(z+1)*bs]
                        mixed = torch.max(label)!=torch.min(label)
                        if mixed:
                            if level==0:
                                self.gt_octree[level][(x,y,z)]=label
                            else:
                                self.gt_octree[level][(x,y,z)]=torch.ones([1],device=device).long()*2
                        else:
                            #all labels are the same, and are either -1 or 1.
                            #so (label+1)/2 is 0 or 1
                            self.gt_octree[level][(x,y,z)]=torch.ones([1],device=device).long()*(label[0,0,0,0]+1)//2

    def loss_singles(self, l, gt, bs):
        assert gt.numel()>0, l.numel()>0
        try:
            ret = self.criteria(l, gt)*bs**3
            return ret
        except:
            import ipdb; ipdb.set_trace()
            print( 'something is wrong')

    def loss_full_single(self, l, gt):
        try:
            return self.criteria(l,torch.ones_like(l[:,0]).long()*gt)
        except:
            print( 'something is wrong')
            import ipdb; ipdb.set_trace()

    def loss_single_full(self, l, gt, bs):
        try:
            loss = self.criteria(l,torch.ones(1,device=device).long()*2)
            return loss
        except:
            print( 'something is wrong')
            import ipdb; ipdb.set_trace()

    def loss_fulls(self, l, gt):
        try:
            return self.criteria(l, gt)
        except:
            print( 'something is wrong')
            import ipdb; ipdb.set_trace()

    def forward(self, octree):
        assert len(octree)<=len(self.gt_octree), (len(octree), len(self.gt_octree))
        ret = torch.zeros(1,device=device).squeeze()
        for level in range(len(octree)-1,-1,-1):
            bs = self.block_size*np.power(2,level)
            assert type(octree[level])==dict, (level, octree[level])
            for X,label in octree[level].items():
                gt = self.gt_octree[level][X]
                assert type(gt)==torch.Tensor
                if gt.numel()==1:
                    if label.numel()==3:
                        assert level!=0 #level 0 should be full res
                        ret+=self.loss_singles(label, gt, bs)
                    else:
                        ret+=self.loss_full_single(label, gt)
                else:
                    if label.numel()==3:
                        ret+=self.loss_single_full(label, gt, bs)
                    else:
                        ret += self.loss_fulls(label, gt)
        return ret


def octree_to_sdf(octree, block_size):
    dim = block_size*2**(len(octree)-1)
    sdf = np.zeros((dim, dim, dim))
    for level in range(len(octree)-1,-1,-1):
        bs = block_size*np.power(2,level)
        for (x,y,z),label in octree[level].items():
            label=label.cpu()
            if label.numel()==1:
                sdf[x*bs:(x+1)*bs, y*bs:(y+1)*bs, z*bs:(z+1)*bs] = label
            if label.numel()==3 and torch.argmax(label)!=2:
                sdf[x*bs:(x+1)*bs, y*bs:(y+1)*bs, z*bs:(z+1)*bs] = torch.argmax(label)*2-1
            if label.shape == (1,2,block_size,block_size,block_size):
                assert level == 0
                sdf[x*bs:(x+1)*bs, y*bs:(y+1)*bs, z*bs:(z+1)*bs] = torch.argmax(label[0],dim=0)*2-1
            if label.shape == (1,1,block_size,block_size,block_size):
                assert level == 0
                sdf[x*bs:(x+1)*bs, y*bs:(y+1)*bs, z*bs:(z+1)*bs] = label
    return sdf



