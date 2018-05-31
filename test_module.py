import copy
import torch.optim as optim
import numpy as np
import torch
from torch.autograd import Variable
from model import *
from data_utils import *

def test_bottom_level():
    tsdf = torch.from_numpy(np.random.rand(1,1,32,32,32)).float()
    prev = torch.from_numpy(np.random.rand(1,128,16,16,16)).float()
    mod = BottomLevel(128)
    out = mod(tsdf,prev)
    assert out.shape==(1,3,32,32,32)
    assert out.sum(dim=1).max()<1.01
    assert out.sum(dim=1).max()>.99

def test_convtrans():
    conv1 = nn.ConvTranspose3d( 16, 16, kernel_size=3, stride=2,
            output_padding=1, padding=1, bias=False)
    dat = Variable(torch.ones(1,16,10,10,10))
    y = conv1(dat)
    assert y.shape[-1]==20


def test_data():
    data = TsdfGenerator(32)
    vis = visdom.Visdom()
    gt, tsdf_in = data.__getitem__(0)
    assert np.abs(tsdf_in).max()<33

def test_ellipsoid():
    arr = ellipsoid(10,10,10,levelset=True)
    assert arr.shape==(23,23,23), arr.shape
    dist =  np.sqrt(10**2*3)-10
    assert np.abs(arr[0,0,0]) > dist , (dist, arr[0,0,0])


    a,b,c = 10,15,25
    arr = ellipsoid(a,b,c,levelset=True)
    vis = visdom.Visdom()
    vis.heatmap(arr[10],win=7)
    #if we move 1 voxel in space the sdf should also not change by more than 1
    # compare to 1.01 for numeric reasons
    assert np.all(np.abs(np.diff(arr,axis=0))<=1.01), np.abs(np.diff(arr,axis=0)).max()
    assert np.all(np.abs(np.diff(arr,axis=1))<=1.01)
    assert np.all(np.abs(np.diff(arr,axis=2))<=1.01)

def test_simple_net_single_data():
    data = TsdfGenerator(32, sigma=0.9)
    criteria = nn.CrossEntropyLoss()
    vis = visdom.Visdom()
    gt, tsdf_in = data.__getitem__(0)
    assert np.abs(tsdf_in).max()<33
    gt_label = np.zeros_like(gt)
    gt_label[gt>0.5]=1
    gt_label[gt<-0.5]=-1
    gt_label+=1
    gt_label = torch.from_numpy(gt_label.astype(int)).cuda()
    tsdf = torch.from_numpy(copy.copy(tsdf_in)[None,:]).float().cuda()
    prev = torch.zeros(1,128,16,16,16).float().cuda()
    assert tsdf.shape == (1,1,32,32,32)
    mod = BottomLevel(128)
    mod.cuda()
    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
    for it in range(1,1000):
        out = mod(tsdf,prev)
        loss = criteria(out, gt_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it%10==0:
            sdf_=np.argmax(out[0].cpu().detach().numpy(),axis=0)-1
            print 'level ', np.count_nonzero(sdf_==1)
            err = plotVoxelVisdom(gt[0], sdf_, tsdf_in[0], vis,title='3d')
            assert np.abs(tsdf_in).max()<33
            print err

        print it, loss
    assert err < 1

def test_simple_net():
    dataset = TsdfGenerator(32, n_elips=3, sigma=0.9)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
            num_workers=1)

    criteria = nn.CrossEntropyLoss()
    vis = visdom.Visdom()
    mod = BottomLevel(128)
    mod.cuda()
    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
    for it, (gt, tsdf_in) in enumerate(train_loader):
        assert np.abs(tsdf_in).max()<33
        assert gt.max()>1 and gt.min()<-1
        gt_label = torch.zeros_like(gt)
        gt_label[gt>0.5]=1
        gt_label[gt<-0.5]=-1
        gt_label+=1
        gt_label = gt_label.long().cuda()
        tsdf = tsdf_in.float().cuda()
        prev = torch.zeros(1,128,16,16,16).float().cuda()
        assert tsdf.shape == (1,1,32,32,32)
        out = mod(tsdf,prev)
        loss = criteria(out, gt_label)
        print it, loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it%10==0:
            sdf_=np.argmax(out[0].cpu().detach().numpy(),axis=0)-1
            print 'level ', np.count_nonzero(sdf_==1)
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0][0].numpy(), vis)
            assert np.abs(tsdf_in).max()<33
            print err

if __name__=='__main__':
    #test_ellipsoid()
    #exit()
    test_simple_net()
    exit()#
    test_data()
    test_convtrans()
    test_bottom_level()
