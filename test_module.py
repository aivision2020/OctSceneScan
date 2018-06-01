import copy
import torch.optim as optim
import numpy as np
import torch
from torch.autograd import Variable
from model import *
from data_utils import *


feature_dim = 32
def test_bottom_level():
    tsdf = torch.from_numpy(np.random.rand(1,1,32,32,32)).float()
    prev = torch.from_numpy(np.random.rand(1,feature_dim,16,16,16)).float()
    mod = BottomLevel(feature_dim)
    out = mod(tsdf,prev)
    assert out.shape==(1,3,32,32,32)
    assert out.sum(dim=1).max()<1.01
    assert out.sum(dim=1).max()>.99

def test_mid_level():
    tsdf = torch.rand(1,1,64,64,64).float().cuda()
    print tsdf.shape
    prev = torch.rand(1,32,16,16,16).float().cuda()
    mod = MidLevel(64, 32,32,BottomLevel(32)).cuda()
    out = mod(tsdf,prev)
    assert out.shape==(1,3,64,64,64)
    assert out.sum(dim=1).max()<1.01
    assert out.sum(dim=1).max()>.99

def test_convtrans():
    conv1 = nn.ConvTranspose3d( 16, 16, kernel_size=3, stride=2,
            output_padding=1, padding=1, bias=False)
    dat = Variable(torch.ones(1,16,10,10,10))
    y = conv1(dat)
    assert y.shape[-1]==20

    pad = nn.ReplicationPad3d(1)
    conv1 = nn.ConvTranspose3d( 1, 1, kernel_size=3, stride=2,
            output_padding=1, padding=1, bias=False)
    dat = Variable(torch.ones(1,1,4,4,4))
    y = conv1(dat)
    assert y.shape[-1]==8, y.shape


def test_data():
    data = TsdfGenerator(32)
    vis = visdom.Visdom()
    gt, tsdf_in = data.__getitem__(0)
    assert np.abs(tsdf_in).max()<33

def test_ellipsoid():
    arr = ellipsoid(10,10,10,levelset=True)*10 #the output is ~normalized.  multiple by 10
    assert arr.shape==(23,23,23), arr.shape
    dist =  np.sqrt(11**2*3)-10
    assert np.abs(arr[0,0,0]) > dist , ( arr[0,0,0], dist)
    print arr[0,0,0], dist


    a,b,c = 10,15,25
    arr = ellipsoid(a,b,c,levelset=True)
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
    gt = gt[None,:] #add dim for batch
    assert np.abs(tsdf_in).max()<33
    gt_label = np.zeros_like(gt)
    gt_label[gt>0.5]=1
    gt_label[gt<-0.5]=-1
    gt_label+=1
    gt_label = torch.from_numpy(gt_label.astype(int)).cuda()
    tsdf = torch.from_numpy(copy.copy(tsdf_in)[None,:]).float().cuda()
    prev = torch.zeros(1,feature_dim,16,16,16).float().cuda()
    assert tsdf.shape == (1,1,32,32,32)
    assert gt_label.shape==(1,32,32,32)
    mod = BottomLevel(feature_dim)
    mod.cuda()
    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
    for it in range(1,100):
        out = mod(tsdf,prev)
        assert out.shape[1]==3, out.shape
        loss = criteria(out, gt_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (it+1)%10==0:
            sdf_=np.argmax(out[0].cpu().detach().numpy(),axis=0)-1
            print 'level ', np.count_nonzero(sdf_==1)
            err = plotVoxelVisdom(gt[0], sdf_, tsdf_in[0], vis)
            assert np.abs(tsdf_in).max()<33
            print err

        print it, loss
    assert err < 1

def test_simple_net():
    dataset = TsdfGenerator(32, n_elips=3, sigma=0.9,epoch_size=100)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
            num_workers=4)

    criteria = nn.CrossEntropyLoss()
    vis = visdom.Visdom()
    mod = BottomLevel(feature_dim)
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
        prev = torch.zeros(1,feature_dim,16,16,16).float().cuda()
        assert tsdf.shape == (1,1,32,32,32)
        out = mod(tsdf,prev)
        loss = criteria(out, gt_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it%50==0:
            sdf_=np.argmax(out[0].cpu().detach().numpy(),axis=0)-1
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0][0].numpy(), vis)
            print it, err
    assert err<1

def test_dynamic_graph():
    x = torch.ones(5)
    w = torch.ones(5, requires_grad=True)
    y = x.mul(w)
    if y.max()>4:
        y = nn.sum(y)
    else:
        y = y.max()
    y.backward()

def test_2tier_net():
    res = 64
    dataset = TsdfGenerator(res, n_elips=3, sigma=0.9,epoch_size=100)

    criteria = nn.CrossEntropyLoss()
    vis = visdom.Visdom()
    mod = MidLevel(res, feature_dim,feature_dim,BottomLevel(feature_dim))
    mod.cuda()
    optimizer = optim.Adam(mod.parameters(), lr=0.1)#, momentum=0.9)
    gt, tsdf_in = dataset.__getitem__(0)
    assert np.abs(tsdf_in).max()<33
    assert gt.max()>1 and gt.min()<-1
    gt = torch.from_numpy(gt[None,:])
    tsdf_in = torch.from_numpy(tsdf_in[None,:])
    gt_label = torch.zeros_like(gt)
    gt_label[gt>0.5]=1
    gt_label[gt<-0.5]=-1
    gt_label+=1
    gt_label = gt_label.long().cuda()
    tsdf = tsdf_in.float().cuda()
    prev = torch.rand(1,feature_dim,16,16,16).float().cuda()
    assert tsdf.shape == (1,1,res,res,res)
    last_out=None
    for it in range(100):
        out = mod(tsdf,prev)
        #if last_out is None:
        #    last_out = out.cpu().detach().numpy()
        #else:
        #    assert not np.all(last_out==out.cpu().detach().numpy())
        #    #last_out = out.cpu().detach().numpy()
        loss = criteria(out, gt_label)
        optimizer.zero_grad()
        loss.backward()
        for param in mod.parameters():
            assert param.grad is not None, param
        optimizer.step()
        print it, loss
        if (it+1)%40==0:
            sdf_=np.argmax(out[0].cpu().detach().numpy(),axis=0)-1
            err = plotVoxelVisdom(gt[0].numpy(), sdf_, tsdf_in[0][0].numpy(), vis)
            print it, err
    assert err<1
if __name__=='__main__':
    test_convtrans()
