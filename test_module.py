import copy
import time
import torch.optim as optim
import numpy as np
import torch
from torch.autograd import Variable
from model import *
from data_utils import *

#device = torch.device('cuda')
device = 'cuda'
feature_dim = 16
block_size = 32


def test_bottom_level():
    tsdf = [torch.from_numpy(np.random.rand(1, 1, block_size, block_size,
        block_size)).float().to(device)]
    prev = {(0, 0, 0): torch.from_numpy(np.random.rand(1, feature_dim,
        block_size//2, block_size//2, block_size//2)).float().to(device)}
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
        assert out[X].sum(dim=1).max() < 1.01
        assert out[X].sum(dim=1).max() > .99


def test_mid_level():
    tsdf = [torch.rand(1, 1, block_size*2, block_size*2, block_size*2).float().to(device)]
    tsdf_down = nn.AvgPool3d(kernel_size=2)
    while tsdf[-1].shape[-1] > block_size:
        tsdf.append(tsdf_down(tsdf[-1]))
    prev = {(0, 0, 0): torch.from_numpy(np.random.rand(1, feature_dim,
        block_size//2, block_size//2, block_size//2)).float().to(device)}
    mod = MidLevel(feature_dim, feature_dim,
                   BottomLevel(feature_dim, block_size=block_size),
                   block_size)
    if device == 'cuda':
        mod.cuda()
    out = mod(tsdf, prev)
    assert type(out) == list
    assert type(out[0]) == dict
    assert len(out) == 2
    for X, pred in out[0].items():
        assert pred.shape == (1, 2, block_size, block_size, block_size), pred.shape
        assert 0.99 < pred.sum(dim=1).max() < 1.01
    for X, pred in out[1].items():
        assert pred.shape == (1, 3), pred.shape
        assert 0.99 < pred.sum(dim=1).max() < 1.01


def test_convtrans():
    conv1 = nn.ConvTranspose3d(10, 10, kernel_size=4, stride=2,
                               output_padding=0, padding=0, bias=False)
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
    data = TsdfGenerator(block_size)
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


def test_criteria():
    data = TsdfGenerator(2*block_size, sigma=0.9)
    gt, tsdf_in = data.__getitem__(0)
    gt = gt[None, :]  # add dim for batch
    assert np.abs(tsdf_in).max() < 33
    gt_label = np.zeros_like(gt)
    gt_label[gt >= 0] = 1
    gt_label = torch.from_numpy(gt_label.astype(int)).to(device)
    criteria = OctreeCrossEntropyLoss(gt_label, block_size)
    assert len(criteria.gt_octree) == 2
    assert len(criteria.gt_octree[0]) == 8, len(criteria.gt_octree[0])
    assert len(criteria.gt_octree[1]) == 1, len(criteria.gt_octree[1])
    for l, level in enumerate(criteria.gt_octree):
        for k, v in level.items():
            assert v.dim() > 0, (l, k, v)


def test_simple_net_single_data():
    data = TsdfGenerator(block_size, sigma=0.9)
    vis = visdom.Visdom()
    gt, tsdf_in = data.__getitem__(0)
    gt = gt[None, :]  # add dim for batch
    assert np.abs(tsdf_in).max() < 33
    gt_label = np.zeros_like(gt)
    gt_label[gt >= 0] = 1
    gt_label = torch.from_numpy(gt_label.astype(int)).device(device)
    tsdf = [torch.from_numpy(copy.copy(tsdf_in)[None, :]).float().device(device)]
    prev = {(0, 0, 0): torch.rand(1, feature_dim, block_size/2, block_size/2, block_size/2).float().device(device)}
    assert tsdf[0].shape == (1, 1, block_size, block_size, block_size)
    assert gt_label.shape == (1, block_size, block_size, block_size)
    criteria = OctreeCrossEntropyLoss(gt_label, block_size)
    mod = BottomLevel(feature_dim, block_size)
    mod.device(device)
    optimizer = optim.Adam(mod.parameters(), lr=0.001)  # , momentum=0.9)
    for it in range(1, 100):
        out = mod(tsdf, prev)
        assert len(out) == 1
        assert out[0].values()[0].shape[1] == 2, out.shape
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
    assert err < 1


def test_bottom_layer():
    dataset = TsdfGenerator(block_size, n_elips=3, sigma=0.9, epoch_size=1000)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                               num_workers=4)

    vis = visdom.Visdom()
    mod = BottomLevel(feature_dim, block_size)
    if device=='cuda':
        mod.cuda()
    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
    for it, (gt, tsdf_in) in enumerate(train_loader):
        assert np.abs(tsdf_in).max() < 33
        assert gt.max() > 1 and gt.min() < -1
        gt_label = torch.zeros_like(gt)
        gt_label[gt >= 0] = 1
        gt_label = gt_label.long().to(device)
        tsdf = [tsdf_in.float().to(device)]
        prev = {(0, 0, 0):
                torch.zeros(1, feature_dim, block_size//2, block_size//2,
                    block_size//2).float().to(device)}
        out = mod(tsdf, prev)
        criteria = OctreeCrossEntropyLoss(gt_label, block_size)
        loss = criteria(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(it, loss)
        if it % 50 == 0:
            sdf_ = octree_to_sdf(out, block_size)
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0][0].numpy(), vis)
            print(it, err)
    assert err < 1


def test_2tier_net_single_data():
    res = block_size*2
    dataset = TsdfGenerator(res, n_elips=2, sigma=0.9, epoch_size=100)

    vis = visdom.Visdom()
    mod = TopLevel(feature_dim, BottomLevel(feature_dim, block_size), block_size=block_size)
    if device == 'cuda':
        mod.cuda()
    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
    gt, tsdf_in = dataset.__getitem__(0)
    assert np.abs(tsdf_in).max() < 33
    assert gt.max() > 1 and gt.min() < -1
    gt = torch.from_numpy(gt[None, :])
    gt_label = torch.zeros_like(gt)
    gt_label[gt >= 0] = 1
    gt_label = gt_label.long().to(device)
    criteria = OctreeCrossEntropyLoss(gt_label, block_size)
    tsdf = torch.from_numpy(copy.copy(tsdf_in)[None, :]).float().to(device)
    last_out = None
    for it in range(500):
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
            mod.eval()
            out = mod(tsdf)
            sdf_ = octree_to_sdf(out, block_size)
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0], vis)
            mod.train()
            print(it, err)
    assert err < 1


def test_2tier_net(res=64, block_size=block_size):
    dataset = TsdfGenerator(res, n_elips=3, sigma=0.9, epoch_size=10000)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                               num_workers=2)

    vis = visdom.Visdom()
    layers = []
    layers.append(BottomLevel(feature_dim, block_size).cuda())
    while block_size*2**len(layers) <= res/2:
        print('adding mid layer', len(layers))
        layers.append(MidLevel(feature_dim, feature_dim, layers[-1],
                               block_size, thresh=0.1))
    mod = TopLevel(feature_dim, layers[-1], block_size=block_size)
    if device == 'cuda':
        mod.cuda()
    optimizer = optim.Adam(mod.parameters(), lr=0.001)  # , momentum=0.9)
    last_t = time.time()
    for it, (gt, tsdf_in) in enumerate(train_loader):
        assert np.abs(tsdf_in).max() < 33
        assert gt.max() > 1 and gt.min() < -1
        gt_label = torch.zeros_like(gt, device=device)
        gt_label[gt >= 0] = 1
        gt_label = gt_label.long()
        criteria = OctreeCrossEntropyLoss(gt_label, block_size)
        tsdf = tsdf_in.float().cuda()
        t = time.time()
        pred = mod(tsdf)
        forward_t = time.time()-t
        t = time.time()
        loss = criteria(pred)
        loss_t = time.time()-t
        t = time.time()
        for i, l in enumerate(pred):
            resample = (2**i)
            print('level %d, %d/%d trained' % (i, len(l),
                                               (res/block_size/resample)**3))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        back_t = time.time()-t
        t = time.time()
        print(it, loss.data)
        print('timming:{}. forward {}, loss {}, back {}'.format(t-last_t,
                                                                forward_t, loss_t, back_t))
        last_t = t
        if (it+1) % 2 == 0:
            mod.eval()
            out = mod(tsdf)
            for i, l in enumerate(pred):
                resample = (2**i)
                print('Eval: level %d, %d/%d evaluated' % (i, len(l),
                                                           (res/block_size/resample)**3))
            sdf_ = octree_to_sdf(out, block_size)
            print(gt[0].shape)
            print(sdf_.shape)
            print(tsdf_in[0][0].shape)
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0][0].numpy(), vis)
            mod.train()
            print(it, err)
    assert err < 1


# def split_tree(feat):
#    subtree = {}
#    for x,feat_x in enumerate(torch.split(feat,8,2)):
#        for y,feat_y in enumerate(torch.split(feat_x,8,3)):
#            for z,feat_z in enumerate(torch.split(feat_y,8,4)):
#                subtree[(x,y,z)]=feat_z
#    return subtree

def test_split_subtree():
    feat = torch.rand(1, feature_dim, block_size, block_size,
            block_size).float()
    split = split_tree(feat)
    assert len(split) == 8, len(split)
    assert torch.all(split[(0, 0, 0)][0, :, 0, 0, 0] == feat[0, :, 0, 0, 0])
    assert torch.all(split[(1, 0, 0)][0, :, 0, 0, 0] == feat[0, :,
        block_size//2, 0, 0])
    split[(1, 0, 0)][0, 0, 0, 0, 0] = 12.13
    assert feat[0, 0, block_size//2, 0, 0] == 12.13


if __name__ == '__main__':
    #test_bottom_layer()
    ##test_bottom_level()
    #test_mid_level()
    #test_convtrans()
    #test_data()
    #test_ellipsoid()
    #test_criteria()
    #test_split_subtree()
    # TODO why does this not converge? interesting
    test_2tier_net_single_data()
    #test_criteria()
    #test_2tier_net(res=64, block_size=block_size)
    #test_2tier_net(res=128, block_size=block_size)
