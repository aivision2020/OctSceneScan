import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger



#device = 'cpu'
device = 'cuda'

INSIDE=0
OUTSIDE=1
MIXED=2


class Conv3d(nn.Module):
    def __init__(self, inplane, outplane, bn=True, relu=True):
        super(Conv3d, self).__init__()
        modules = []
        modules.append(nn.Conv3d(inplane, outplane, kernel_size=3, stride=1,
            padding=0, bias=False))
        if bn:
            modules.append(nn.BatchNorm3d(outplane))
        if relu:
            modules.append(nn.ReLU(inplace=True))
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        out = self.module(x)
        return out




class Debug(nn.Module):
    def __init__(self, str=""):
        super(Debug, self).__init__()
        self.str = str

    def forward(self, input):
        #import ipdb; ipdb.set_trace()
        logger.debug("{} {}, {}, {}".format(self.str, input.shape, input.min(), input.max()))
        assert torch.all(torch.isfinite(input))
        return input




def _split_overlap(a, dim, pad):
    n = (a.shape[dim]-2*pad)//2
    return a.narrow(dim, 0, n+2*pad), a.narrow(dim, n, n+2*pad)


def split_tree(feat, parent_x=0, parent_y=0, parent_z=0, padding=2):
    #pad is one sided (always even)
    block_size = feat.shape[-1]
    assert feat.shape[-1] == feat.shape[-2] == feat.shape[-3]
    subtree = {}
    #feat_pad = F.pad(feat, pad=[padding]*6+[0]*2)
    feat_pad = feat
    for x, feat_x in enumerate(_split_overlap(feat_pad, 2, padding)):
        for y, feat_y in enumerate(_split_overlap(feat_x, 3, padding)):
            for z, feat_z in enumerate(_split_overlap(feat_y, 4, padding)):
                subtree[(parent_x*2+x, parent_y*2+y, parent_z*2+z)] = feat_z
    return subtree

class BottomLevel(nn.Module):
    def __init__(self, inplane, block_size):
        super(BottomLevel, self).__init__()
        self.upsample = nn.ConvTranspose3d(
                inplane, inplane, kernel_size=4,
                stride=2, padding=0,
                output_padding=0, bias=False)
        self.n_conv=3
        self.pad = self.n_conv-1
        modules=[]
        modules.append(Debug('input Bottom Level'))
        for i in range(self.n_conv):
            #first conv has 1 extra input channel
            #dims are 40,38,36 (assuming block size 32)
            # 36 = block size + 2*pad
            modules.append(Conv3d(inplane+(i==0), inplane))
            modules.append(Debug('conv%d'%i))
        #2 more convs for prediction. final size == block size (no overlap)
        modules.append(Conv3d(inplane, inplane))
        modules.append(Debug('botttom level pred 1'))
        modules.append(Conv3d(inplane, 2, False, False))
        modules.append(Debug('botttom level pred 2'))

        # modules.append(nn.Softmax(dim=1))
        #modules.append(Debug('botttom level pred softmax'))
        self.convs = nn.Sequential(*modules)
        self.block_size = block_size
        self.debug = Debug('generic bottom debug')

    def forward(self, tsdf_pyramid, prev):
        assert len(tsdf_pyramid) == 1
        assert tsdf_pyramid[-1].shape[0] == 1
        assert type(tsdf_pyramid) == list
        logger.debug("before upsample")
        logger.debug([p.shape for p in prev.values()])
        prevs = {X: self.upsample(p) for X, p in prev.items()}
        logger.debug("after upsample")
        logger.debug([p.shape for p in prevs.values()])
        padded_size = self.block_size+2*self.pad+2*self.n_conv
        tsdfs = {(x, y, z):
                tsdf_pyramid[-1][:, :,
                    x*self.block_size:x*self.block_size+padded_size,
                    y*self.block_size:y*self.block_size+padded_size,
                    z*self.block_size:z*self.block_size+padded_size]
            for (x, y, z) in prevs.keys()}
        logger.debug("tsdfs")
        logger.debug([p.shape for p in tsdfs.values()])
        cat = {X: torch.cat((prevs[X], tsdfs[X]), dim=1) for X in prevs.keys()}
        pred = {X: self.convs(T) for X, T in cat.items()}
        assert np.all([p.shape[-1]==self.block_size for p in pred.values()])
        return [pred]

class MidLevel(nn.Module):
    def __init__(self, inplane, outplane, sub_level, block_size, thresh=0.5, budget=100):
        super(MidLevel, self).__init__()
        self.sub_level = sub_level
        self.thresh = thresh
        self.block_size = block_size
        self.n_conv=3
        self.branch_factor=2
        self.pad = self.n_conv-1
        self.pred = PredictionBlock(inplane, 3, self.block_size//2 +2*self.pad)
        self.upsample = nn.ConvTranspose3d( inplane, inplane, kernel_size=4,
                stride=2, padding=0, output_padding=0, bias=False)
        self.convs = nn.Sequential(*[Conv3d(inplane+1 if i==0 else outplane, outplane)
                for i in range(self.n_conv)])
        self.budget = budget
        self.debug = Debug('Midlevel debug')


    def forward(self, tsdf_pyramid, prev):
        assert type(tsdf_pyramid) == list
        assert type(prev) == dict
        pred = {X: self.pred(T) for X, T in prev.items()}
        sm = nn.Softmax(dim=1)
        mixed = [X for X, p in pred.items() if sm(p)[0, -1] > self.thresh]
        if self.training:
            pass
            # durring test time continue to refine only boundary cells
            # TODO: add additional test that mixed is not largest
            #if len(mixed)>self.budget:
            #    mixed = [mixed[i] for i in np.random.choice(range(len(mixed)), self.budget)]

        prevs = {X: self.upsample(prev[X]) for X in mixed}
        padded_size = self.block_size+2*self.pad+2*self.n_conv
        tsdfs = {(x, y, z):
                tsdf_pyramid[-1][:, :,
                    x*self.block_size:(x*self.block_size +padded_size),
                    y*self.block_size:(y*self.block_size +padded_size),
                    z*self.block_size:(z*self.block_size +padded_size)]
            for (x, y, z) in prevs.keys()}

        if  np.any([prevs[X].shape[-3:]!=tsdfs[X].shape[-3:] for X in prevs.keys()]):
            import ipdb; ipdb.set_trace()
        cat = {X: torch.cat((prevs[X], tsdfs[X]), dim=1) for X in prevs.keys()}

        features = {X: self.convs(T) for X, T in cat.items()}
        subtree = {}
        for (x,y,z), feat in features.items():
            subtree.update(split_tree(feat,x,y,z))

        return self.sub_level(tsdf_pyramid[:-1], subtree) + [pred]

class TopLevel(nn.Module):
    def __init__(self, outplane, sub_level, block_size):
        super(TopLevel, self).__init__()
        self.sub_level = sub_level
        self.n_conv = 3
        self.pad = self.n_conv-1
        self.block_size = block_size

        self.pool = nn.AvgPool3d(kernel_size=2)
        self.rep_pad = nn.ReplicationPad3d(self.pad+self.n_conv)

        modules=[]
        modules.append(Conv3d(1, outplane))
        for i in range(self.n_conv-1):
            #dims are 40,38,36 (assuming block size 32)
            # 36 = block size + 2*pad
            modules.append(Conv3d(outplane, outplane))
        self.convs = nn.Sequential(*modules)


    def forward(self, TSDF):
        last = TSDF
        pyramid = [self.rep_pad(last)]
        while last.shape[-1] > self.block_size:
            last = self.pool(last)
            pyramid.append(self.rep_pad(last))
        feat = self.convs(pyramid[-1])
        logger.debug(feat.shape)
        assert feat.shape[-1] == self.block_size+2*self.pad, (feat.shape[-1], self.block_size)

        # always split top level. No early termination yet
        subtree = split_tree(feat,padding=2)

        #label as mixed [inside, mixed, outside] . negative tsdf is inside object
        mixed = torch.zeros(1,3).float().to(device)
        mixed[0,MIXED]=1
        return self.sub_level(pyramid[:-1], subtree) + [
                {(0, 0, 0): mixed}]


class OctreeCrossEntropyLoss(nn.Module):
    def __init__(self, gt_label, block_size):
        super(OctreeCrossEntropyLoss, self).__init__()
        '''
        gt_label is the GT signed binary distance function
        0 <= gt_label <=1. while 0 is "inside" and 1 is "outside"
        '''

        self.block_size = block_size
        self.max_level = int(np.log2(gt_label.shape[-1]/block_size))
        assert 2**self.max_level*block_size == gt_label.shape[-1]
        ratio = torch.sum(gt_label==1)/torch.sum(gt_label==0)
        #self.weights=torch.Tensor([64, 1, 64]) #TODO, this should be proportianate to the input data
        self.sm = nn.Softmax(dim=1)
        #self.weights=torch.Tensor([1, 1, 1]) #TODO, this should be proportianate to the input data
        self.weights=torch.Tensor([ratio, 1, ratio/2]) #TODO, this should be proportianate to the input data
        self.full_loss = nn.CrossEntropyLoss(self.weights[:2])#, reduction='sum')
        #self.full_loss = nn.NLLLoss(weight=self.weights[:2], reduction='sum')
        self.gt_octree = [{} for _ in range(self.max_level+1)]  # each level is a dictionary
        for level in range(self.max_level, -1, -1):
            bs = block_size*np.power(2, level)
            num_blocks = gt_label.shape[-1]//bs
            for (x,y,z) in itertools.product(range(num_blocks), range(num_blocks), range(num_blocks)):
                label = gt_label[:, x*bs:(x+1)*bs,
                        y*bs:(y+1)*bs,
                        z*bs:(z+1)*bs]
                self.gt_octree[level][(x,y,z)] = self._encode_block(label, level)

    def _encode_block(self, label, level):
        """ encode a dense label tensor into octree format
        label - the dense label block
        level - current level of octree (only level 0 will return a dense label
        """
        mixed = torch.max(label) != torch.min(label)
        if mixed:
            if level == 0:
                #finest level. keep all dense data
                return label
            else:
                # encode current level as "mixed"
                return torch.ones([1], device=device).long()*MIXED
        else:
            # all labels are the same, encode only once
            return torch.ones([1], device=device).long()*label[0, 0, 0, 0]

    def loss_singles(self, l, gt, bs):
        assert gt.numel() > 0, l.numel() > 0
        try:
            l = self.sm(l)
            return -torch.log(l[0,gt[0]])*self.weights[gt[0]]
            #ret = self.singles_loss(l, gt)*bs**3
            #assert ret>0
            #return ret
        except:
            import ipdb; ipdb.set_trace()
            logger.info('something is wrong')

    def loss_full_single(self, l, gt):
        try:
            logger.debug('full accuracy', torch.sum(torch.argmax(l,dim=1)==gt).float()/l.numel())
            loss= self.full_loss(l, torch.ones_like(l[:, 0]).long()*gt)
            assert loss >=0
            logger.debug('full loss    ', loss)
            return loss
        except:
            logger.info('something is wrong')
            import ipdb; ipdb.set_trace()

    def loss_single_full(self, l, gt, bs):
        try:
            import ipdb; ipdb.set_trace()
            loss = self.criteria(l, torch.ones(1, device=device).long()*gt)
            return loss
        except:
            logger.info('something is wrong')
            import ipdb; ipdb.set_trace()

    def loss_fulls(self, l, gt):
        assert torch.all(torch.isfinite(l))
        try:
            logger.debug('full accuracy', torch.sum(torch.argmax(l, dim=1)==gt).float()/l.numel())
            loss = self.full_loss(l, gt)
            assert loss >=0
            logger.debug('full loss    ', loss)
            return loss
        except:
            logger.info('something is wrong')
            import ipdb; ipdb.set_trace()

    def forward(self, octree):
        #import ipdb;ipdb.set_trace()
        assert len(octree) <= len(self.gt_octree), (len(octree), len(self.gt_octree))
        ret = torch.zeros(1, device=device).squeeze()
        for level in range(len(octree)-1, -1, -1):
            bs = self.block_size*np.power(2, level)
            assert type(octree[level]) == dict, (level, octree[level])
            for X, label in octree[level].items():
                gt = self.gt_octree[level][X]
                assert type(gt) == torch.Tensor
                if gt.numel() == 1:
                    if label.numel() == 3:
                        assert level != 0  # level 0 should be full res
                        ret += self.loss_singles(label, gt, bs)
                    else:
                        ret += self.loss_full_single(label, gt)
                else:
                    assert gt.shape[-1]==bs
                    if label.numel() == 3:
                        ret += self.loss_single_full(label, gt, bs)
                    else:
                        ret += self.loss_fulls(label, gt)
                if ret<0:
                    import ipdb; ipdb.set_trace()
                assert ret>=0, (ret, X, level)
        return ret


def octree_to_sdf(octree, block_size):
    #remember octree holds actual activations (not onehot)
    dim = block_size*2**(len(octree)-1)
    sdf = np.zeros((dim, dim, dim))
    def label_to_dist(l):
        assert np.all(l.numpy().ravel()<2)
        return l*2-1

    for level in range(len(octree)-1, -1, -1):
        bs = block_size*np.power(2, level)
        for (x, y, z), label in octree[level].items():
            assert label.shape[0]==1, 'dont support batch size more then one'
            label = label.cpu()
            if label.numel() == 1:
                sdf[x*bs:(x+1)*bs, y*bs:(y+1)*bs, z*bs:(z+1)*bs] = label_to_dist(label)
            elif label.numel() == 3:
                label = torch.argmax(label.squeeze()[:2])
                sdf[x*bs:(x+1)*bs, y*bs:(y+1)*bs, z*bs:(z+1)*bs] = label_to_dist(label)
            elif label.shape == (1, 2, block_size, block_size, block_size):
                assert level==0, "only last level should have full labels"
                assert block_size==bs
                label = torch.argmax(label[0], dim=0)
                sdf[x*bs:(x+1)*bs, y*bs:(y+1)*bs, z*bs:(z+1)*bs] = label_to_dist(label)
            elif label.shape == (1, 1, block_size, block_size, block_size):
                #this doesnt happen in current architecture
                assert False
                sdf[x*bs:(x+1)*bs, y*bs:(y+1)*bs, z*bs:(z+1)*bs] = label_to_dist(label)
            else:
                import ipdb; ipdb.set_trace()
                assert False
    return sdf


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class PredictionBlock(nn.Module):
    '''output prediction for this block  '''

    def __init__(self, inplane, n_class, input_res, bn=True):
        super(PredictionBlock, self).__init__()
        modules = []
        modules.append(Debug('input prediction'))
        lastdim = inplane

        res = input_res
        while res > 3:
            dim = lastdim
            if lastdim > 8:
                dim//=2
            modules.append(Conv3d(lastdim, dim, bn=bn))
            modules.append(Debug('input prediction'))
            modules.append(nn.MaxPool3d(kernel_size=2))
            modules.append(Debug('input prediction'))
            res = (res-2)//2
            lastdim=dim

        modules.append(Flatten())
        modules.append(nn.Linear(dim, n_class))  # 2^3
        # add dropout?
        # modules.append(nn.Softmax(dim=1))
        modules.append(Debug('output prediction'))
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        out = self.module(x)
        return out


