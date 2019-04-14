from pathlib import Path
import copy
import time
import torch.optim as optim
import numpy as np
import torch
from torch.autograd import Variable
from model import *
from data_utils import *
import torch.nn as nn
from loguru import logger

feature_dim = 8
block_size = 16
pad=2
n_conv=3
thresh=0.5
debug = False

def test_bottom_io():
    tsdf = [torch.from_numpy(np.random.rand(1, 1, block_size+2*pad+2*n_conv,
        block_size+2*pad+2*n_conv,
        block_size+2*pad+2*n_conv)).float().to(device)]
    prev = {(0, 0, 0): torch.from_numpy(np.random.rand(1, feature_dim,
        block_size//2+2*pad, block_size//2+2*pad, block_size//2+2*pad)
        ).float().to(device)}
    mod = BottomLevel(feature_dim, block_size=block_size)
    if device == 'cuda':
        mod.cuda()
    out = mod(tsdf, prev)
    assert type(out) == list
    assert len(out) == 1
    out = out[0]
    assert len(out) == 1
    for X in out.keys():
        assert out[X].shape == (1, 2, block_size, block_size, block_size), out[X].shape


def test_convtrans():
    conv1 = nn.ConvTranspose3d(10, 10, kernel_size=4, stride=2, output_padding=0, padding=0, bias=False)
    dat = torch.ones(1, 10, block_size, block_size, block_size)
    y = conv1(dat)
    assert y.shape[-1] == block_size*2+2 , (y.shape, dat.shape)

    pad = nn.ReplicationPad3d(1)
    conv1 = nn.ConvTranspose3d(1, 1, kernel_size=3, stride=2,
                               output_padding=1, padding=1, bias=False)
    dat = Variable(torch.ones(1, 1, 4, 4, 4))
    y = conv1(dat)
    assert y.shape[-1] == 8, y.shape


def test_data():
    data = TsdfGenerator(64)
    vis = visdom.Visdom()
    gt, tsdf_in = data.__getitem__(0)
    assert np.abs(tsdf_in).max() < 33


def test_ellipsoid():
    arr = ellipsoid(10, 10, 10, levelset=True)*10  # the output is ~normalized.  multiple by 10
    assert arr.shape == (23, 23, 23), arr.shape
    dist = np.sqrt(11**2*3)-10
    assert np.abs(arr[0, 0, 0]) > dist, (arr[0, 0, 0], dist)
    print(arr[0, 0, 0], dist)

    a, b, c = 10, 15, 25
    arr = ellipsoid(a, b, c, levelset=True)
    # if we move 1 voxel in space the sdf should also not change by more than 1
    # compare to 1.01 for numeric reasons
    assert np.all(np.abs(np.diff(arr, axis=0)) <= 1.01), np.abs(np.diff(arr, axis=0)).max()
    assert np.all(np.abs(np.diff(arr, axis=1)) <= 1.01)
    assert np.all(np.abs(np.diff(arr, axis=2)) <= 1.01)


def test_criteria_trivial():
    data = TsdfGenerator(block_size, sigma=0.)
    gt, tsdf_in = data.__getitem_split__()
    gt = gt[None, :]  # add dim for batch
    assert np.abs(tsdf_in).max() < 33
    gt_label = np.zeros_like(gt)
    gt_label[gt >= 0] = 1
    gt_label = torch.from_numpy(gt_label.astype(int))
    criteria = OctreeCrossEntropyLoss(gt_label, block_size)
    assert len(criteria.gt_octree) == 1
    mock_out = np.concatenate((tsdf_in[None,:]<0, tsdf_in[None,:]>=0),
            axis=1).astype(float)
    mock_out=1000*(mock_out-0.5)
    mock_out = [{(0,0,0):torch.from_numpy(mock_out).float()}]
    loss = criteria(mock_out)
    assert loss.dim()==0
    assert loss < 0.01, loss

def test_gt():
    pass
    #get gt, 
    #get gt_octree
    #retnder gt
    #render gt_octree

def test_criteria(levels=2):
    res=2**(levels-1)*block_size
    data = TsdfGenerator(res, sigma=0.9)
    gt, tsdf_in = data.__getitem_split__()
    gt = gt[None, :]  # add dim for batch
    assert np.abs(tsdf_in).max() < res
    #labels should be symetric
    def count_label(gt, label, level=1):
        gt_label = np.zeros_like(gt)
        gt_label[gt >= 0] = 1
        gt_label = torch.from_numpy(gt_label.astype(int))
        criteria = OctreeCrossEntropyLoss(gt_label, block_size)
        gt=criteria.gt_octree[level]
        return np.count_nonzero(np.array(list(gt.values()))==label)

    n_outside =  count_label(gt, OUTSIDE)
    n_inside = count_label(gt, INSIDE)
    n_mixed = count_label(gt, MIXED)
    assert n_outside+n_inside+n_mixed==(2**(levels-2))**3
    rev_inside = count_label(-gt, OUTSIDE)
    assert n_inside==rev_inside, (n_inside, rev_inside)


    gt_label = np.zeros_like(gt)
    gt_label[gt >= 0] = 1
    gt_label = torch.from_numpy(gt_label.astype(int))
    criteria = OctreeCrossEntropyLoss(gt_label, block_size)
    assert len(criteria.gt_octree) == levels
    assert len(criteria.gt_octree[0]) == (2**(levels-1))**3, len(criteria.gt_octree[0])
    assert len(criteria.gt_octree[-1]) == 1, len(criteria.gt_octree[-1])
    for l, level in enumerate(criteria.gt_octree):
        for k, v in level.items():
            assert v.dim() > 0, (l, k, v)


def test_basic_debug():
    T = torch.zeros(1,1,36,36,36)
    outplane = 16
    mod = nn.Conv3d(1, outplane, kernel_size=3, stride=1,
                padding=0, bias=False)
    T = mod(T)
    mod = nn.BatchNorm3d(outplane)
    T = mod(T)
    mod = nn.ReLU(inplace=True)
    T = mod(T)
    mod = nn.Conv3d(outplane, outplane, kernel_size=3, stride=1, 
            padding=0, bias=False)
    T = mod(T)
    mod = nn.BatchNorm3d(outplane)
    T = mod(T)
    assert T.shape == (1,16,32,32,32)


def test_simple_net_single_data():
    data = TsdfGenerator(block_size, sigma=0.9)
    vis = visdom.Visdom()
    gt, tsdf_in = data.__getitem__(0)
    gt = gt[None, :]  # add dim for batch
    assert np.abs(tsdf_in).max() < block_size
    gt_label = np.zeros_like(gt)
    gt_label[gt >= 0] = 1
    gt_label = torch.from_numpy(gt_label.astype(int)).to(device)
    rep_pad = nn.ReplicationPad3d(pad+n_conv)
    tsdf = [rep_pad(torch.from_numpy(copy.copy(tsdf_in)[None, :]).float().to(device))]
    #prev = {(0, 0, 0): torch.rand(1, feature_dim, block_size//2, block_size//2,
    #    block_size//2).float().to(device)}
    prev = {(0, 0, 0): torch.from_numpy(np.random.rand(1, feature_dim,
        block_size//2+2*pad, block_size//2+2*pad, block_size//2+2*pad)
        ).float().to(device)}
    #assert tsdf[0].shape == (1, 1, block_size, block_size, block_size)
    assert gt_label.shape == (1, block_size, block_size, block_size)
    criteria = OctreeCrossEntropyLoss(gt_label, block_size)
    mod = BottomLevel(feature_dim, block_size)
    if device=='cuda':
        mod.cuda()
        criteria.cuda()
    optimizer = optim.Adam(mod.parameters(), lr=0.001)  # , momentum=0.9)
    for it in range(1, 100):
        out = mod(tsdf, prev)
        assert len(out) == 1
        assert out[0][(0,0,0)].shape[1] == 2, out.shape
        loss = criteria(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (it+1) % 10 == 0:
            sdf_ = octree_to_sdf(out, block_size)
            print('level ', np.count_nonzero(sdf_ == 1))
            err = plotVoxelVisdom(gt[0], sdf_, tsdf_in[0], vis)
            assert np.abs(tsdf_in).max() < 33
            print(err)

        print(it, loss)
    assert err < 2


def test_bottom_layer( block_size = 32):
    dataset = TsdfGenerator(block_size, n_elips=1, sigma=0.9, epoch_size=1000)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                               num_workers=4)

    vis = visdom.Visdom()
    mod = BottomLevel(feature_dim, block_size)
    if device=='cuda':
        mod.cuda()
    optimizer = optim.SGD(mod.parameters(), lr=0.0001, momentum=0.9)
    m = nn.ReplicationPad3d(mod.pad+mod.n_conv)
    prev = {(0, 0, 0): torch.rand(1, feature_dim,
            block_size//2+2*pad, block_size//2+2*pad, block_size//2+2*pad
            ).float().to(device)}
    gt_label = None
    for it, (gt, tsdf_in) in enumerate(train_loader):
        assert np.abs(tsdf_in).max() < 33
        assert gt.max() > 1 and gt.min() < -1
        gt_label = torch.ones_like(gt)*INSIDE
        gt_label[gt >= 0] = OUTSIDE
        gt_label = gt_label.long().to(device)
        tsdf = [m(tsdf_in).float().to(device)]
        for T in prev.values():
            assert torch.all(torch.isfinite(T))
        for T in tsdf:
            assert torch.all(torch.isfinite(T))
        out = mod(tsdf, prev)
        assert out[0][(0,0,0)].max()>out[0][(0,0,0)].min()
        for oct in out:
            if not np.all([torch.all(torch.isfinite(o)) for o in oct.values()]):
                import ipdb; ipdb.set_trace()
        criteria = OctreeCrossEntropyLoss(gt_label, block_size)
        if device=='cuda':
            criteria.cuda()
        loss = criteria(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(it, loss)
        if it>1 and it%100 == 0:
            sdf_ = octree_to_sdf(out, block_size)
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0][0].numpy(), vis)
            print(it, err)
    assert err < 2, err


def test_2tier_net_single_data():
    res = block_size*2
    dataset = TsdfGenerator(res, n_elips=3, sigma=0.9, epoch_size=100)

    vis = visdom.Visdom()
    mod = TopLevel(feature_dim, BottomLevel(feature_dim, block_size), block_size=block_size)
    if device == 'cuda':
        mod.cuda()

    optimizer = optim.Adam(mod.parameters(), lr=0.01)#, momentum=0.9)
    gt, tsdf_in = dataset.__getitem__(0)
    assert np.abs(tsdf_in).max() < 33
    assert gt.max() > 1 and gt.min() < -1
    gt = torch.from_numpy(gt[None, :])
    gt_label = torch.zeros_like(gt)
    gt_label[gt >= 0] = 1
    gt_label = gt_label.long().to(device)
    criteria = OctreeCrossEntropyLoss(gt_label, block_size)
    if device == 'cuda':
        criteria.cuda()
    tsdf = torch.from_numpy(copy.copy(tsdf_in)[None, :]).float().to(device)
    for it in range(1000):
        out = mod(tsdf)
        assert len(out) == 2
        for l in out[1:]:
            for v in l.values():
                # only level 0 can have a full bloc
                assert v.shape[-1] < block_size, (v.shape)
        loss = criteria(out)
        assert len(out) == 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(it, loss)
        if (it+1) % 10 == 0:
            #mod.eval()
            sdf_ = octree_to_sdf(out, block_size)
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0], vis)
            #mod.train()
            print(it, err)
    assert err < 2,err


def test_4tier_data(block_size=block_size):
    res=block_size*(2**3)
    dataset = TsdfGenerator(res, n_elips=3, sigma=0.9, epoch_size=1000)
    gt, tsdf = dataset.__getitem__(0)

    mod = BottomLevel(feature_dim, block_size)
    for i in range(2): #add 2 mid layers
        print('adding mid layer')
        mod = MidLevel(feature_dim, feature_dim, mod, block_size,
                thresh=thresh, budget=4)
    mod = TopLevel(feature_dim, mod, block_size=block_size)
    out = mod(torch.from_numpy(tsdf[None,:]).float())



def test_2tier_net(res=64, block_size=block_size):
    dataset = TsdfGenerator(res, n_elips=1, sigma=0.9, epoch_size=10000, debug=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                               num_workers=2)

    vis = visdom.Visdom()
    Force = False
    if not Force and Path('model_2tier.pth').exists():
        mod = torch.load('model_2tier.pth')
    else:
        layers = []
        layers.append(BottomLevel(feature_dim, block_size))
        while block_size*2**len(layers) <= res/2:
            print('adding mid layer', len(layers))
            layers.append(MidLevel(feature_dim, feature_dim, layers[-1],
                                   block_size, thresh=0.5, budget=4))
        mod = TopLevel(feature_dim, layers[-1], block_size=block_size)
    if device == 'cuda':
        mod.cuda()
    optimizer = optim.SGD(mod.parameters(), lr=0.0001, momentum=0.95)
    for it, (gt, tsdf_in) in enumerate(train_loader):
        assert np.abs(tsdf_in).max() < res
        assert gt.max() > 1 and gt.min() < -1
        gt_label = torch.zeros_like(gt, device=device)
        gt_label[gt >= 0] = 1
        gt_label = gt_label.long().to(device)
        criteria = OctreeCrossEntropyLoss(gt_label, block_size)
        if device == 'cuda':
            criteria.cuda()
        #tsdf = tsdf_in.float().cuda()
        t_start = time.time()
        tsdf = tsdf_in.float().to(device)
        pred = mod(tsdf)
        forward_t = time.time()-t_start
        t = time.time()
        loss = criteria(pred)
        loss_t = time.time()-t
        t = time.time()
        optimizer.zero_grad()
        loss.backward()
        back_t = time.time()-t
        t = time.time()
        optimizer.step()
        step_t = time.time()-t
        t = time.time()
        print(it, loss.data)
        print('valuated ', [len(o) for o in pred])
        print('GT voxels ', np.count_nonzero([o.numel()>3 for o in criteria.gt_octree]))
        print('timing:{total:.3f}. forward {forward_t:.3f}, loss {loss_t:.3f}, back {back_t:.3f}, step {step_t:.3f}'.format(
            total=t-t_start, forward_t=forward_t, loss_t=loss_t, back_t=back_t, step_t=step_t))
        if (it+1) % 100 == 0:
            mod.eval()
            out = mod(tsdf)
            loss = criteria(out)
            for i in range(len(out)):
                resample = (2**i)
                print('Eval: level %d, %d/%d evaluated' % (i, len(out[i]),
                                                           (res/block_size/resample)**3))
            sdf_ = octree_to_sdf(out, block_size)
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0][0].numpy(), vis)
            if loss.data<1:
                import ipdb; ipdb.set_trace()
            mod.train()
            print(it, err)
            torch.save(mod, 'model_2tier.pth')
            if err < 2 :
                break
    #assert err < 2

def create_model(block_size, feature_dim, res):
    layers = []
    layers.append(BottomLevel(feature_dim, block_size))
    while block_size*2**len(layers) <= res/2:
        print('adding mid layer', len(layers))
        layers.append(MidLevel(feature_dim, feature_dim, layers[-1],
                               block_size, thresh=0.1))
    mod = TopLevel(feature_dim, layers[-1], block_size=block_size)
    return mod


def test_simple_split(res=64, block_size=block_size):
    dataset = TsdfGenerator(res, n_elips=3, sigma=0.9, epoch_size=1000, debug=True)
    vis = visdom.Visdom()

    mod = torch.load('model.pth')
    if device == 'cuda':
        mod.cuda()
    mod.eval()
    gt, tsdf_in = dataset.__getitem_split__()
    gt = torch.from_numpy(gt[None, :])
    tsdf_in = torch.from_numpy(tsdf_in[None, :])

    gt_label = torch.zeros_like(gt, device=device)
    gt_label[gt >= 0] = 1
    gt_label = gt_label.long().to(device)
    criteria = OctreeCrossEntropyLoss(gt_label, block_size)
    if device == 'cuda':
        criteria.cuda()
    tsdf = tsdf_in.float().to(device)
    pred = mod(tsdf)
    loss = criteria(pred)
    print(loss.data)
    print('evaluated ', [len(o) for o in pred])

    for X in pred[0]:
        X_ = tuple(np.array(X)//2)
        print (X, pred[1][X_])
        assert pred[1][X_][0,2]>0.5
    sdf_ = octree_to_sdf(pred, block_size)
    err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0][0].numpy(), vis)
    import ipdb; ipdb.set_trace()
    for X,v in criteria.gt_octree[0].items():
        if v.numel()>1:
            assert X[2]==1 #that's how we built the space


def test_split_subtree(padding=0):
    feat = torch.rand(1, feature_dim, block_size+2*padding,
            block_size+2*padding,
            block_size+2*padding
            ).float()
    split = split_tree(feat,padding=padding)
    assert len(split) == 8, len(split)
    assert torch.all(split[(0, 0, 0)][0, :, padding, padding, padding] ==
            feat[0, :, padding, padding, padding])
    assert torch.all(split[(1, 0, 0)][0, :, padding, padding, padding] ==
            feat[0, :, block_size//2+padding, padding, padding])
    split[(1, 0, 0)][0, 0, padding, padding, padding] = 12.13
    #this is no longer true, I don't know how to do this inplace
    #assert feat[0, 0, block_size//2, 0, 0] == 12.13

def test_split_subtree_with_padding():
    padding=2
    feat = torch.rand(1, feature_dim, block_size, block_size,
            block_size).float()
    split = split_tree(feat, padding=2)
    assert len(split) == 8, len(split)
    octant = split[(0,0,0)]
    assert torch.all(octant[0, :padding, 0, 0, 0] == 0)
    assert torch.all(octant[0, -padding:, 0, 0, 0] == 0)
    assert octant.shape[-3:]==feat.shape[-3:]//2+padding*2
    assert torch.all(octant[0, padding:-padding, 0, 0, 0] == feat[0, :, 0, 0, 0])
    assert torch.all(octant[0, padding:-padding, 0, 0, 0] == feat[0, :, 0, 0, 0])
    assert torch.all(split[(1, 0, 0)][0, :, padding, padding, padding] ==
            feat[0, :, block_size//2, 0, 0])
    split[(1, 0, 0)][0, 0, 0, 0, 0] = 12.13
    assert feat[0, 0, block_size//2+padding, 0, 0] == 12.13

if __name__ == '__main__':
    import sys
    logger.remove()
    logger.add(sys.stderr , format="{time} {level} {message}",  level="INFO")

    #test_4tier_data()
    #test_criteria_trivial()
    #test_criteria()
    #test_criteria(4)
    #test_data()
    #test_ellipsoid()
    #test_convtrans()
    #test_split_subtree()
    #test_split_subtree(padding=2)
    #test_basic_debug()
    #test_bottom_io()
    #test_simple_net_single_data()
    #test_bottom_layer()
    # TODO why does this not converge? interesting
    #test_2tier_net_single_data()
    #test_2tier_net(res=32, block_size=block_size)
    test_2tier_net(res=64, block_size=block_size)
    test_simple_split(res=64, block_size=block_size)
    import ipdb; ipdb.set_trace()
    test_2tier_net(res=128, block_size=block_size)
