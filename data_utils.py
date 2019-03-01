import torch.utils.data
import torchvision.datasets as datasets

import numpy as np
from skimage import measure
from skimage.draw import ellipsoid
from scipy.spatial import KDTree
import visdom

def plotVoxelVisdom(GT_voxels, voxels, tsdf_in, visdom):
    v_gt,f_gt,_,_ =  measure.marching_cubes_lewiner(GT_voxels, level=0., allow_degenerate=False)
    if visdom is not None:
        visdom.mesh(X=v_gt, Y=f_gt, win=2,opts=dict(opacity=1., title='gt_tsdf'))

    res = voxels.shape[-1]
    if voxels.min()<0 and voxels.max()>0:
        v,f,_,_ =  measure.marching_cubes_lewiner(voxels, level=0., allow_degenerate=False)
        kd = KDTree(v_gt)
        d,ind = kd.query(v)
        assert len(d)==len(v)
        if visdom is not None:
            visdom.histogram(d, win=0, opts=dict(title='hist errors'))
            visdom.mesh(X=v, Y=f, win=1,opts=dict(opacity=1., title='deep tsdf denoising'))
            v,f,_,_ =  measure.marching_cubes_lewiner(tsdf_in, level=0., allow_degenerate=False)
            visdom.mesh(X=v, Y=f, win=3,opts=dict(opacity=1., title='input tsdf'))
            visdom.heatmap(tsdf_in[res//2,:,:], win=4,opts=dict(title='mid slice input tsdf'))
            visdom.heatmap(voxels[res//2,:,:], win=5, opts=dict(title='mid slice output tsdf x'))
            visdom.heatmap(voxels[:,res//2,:], win=6, opts=dict(title='mid slice output tsdf y'))

        return np.mean(d)
    else:
        print( 'visdom cant render empy tsdf')
    return None

class TsdfGenerator(torch.utils.data.Dataset):
    def __init__(self, res=512, n_elips=1, sigma=0.1, epoch_size=1000):
        super(TsdfGenerator, self).__init__()
        self.res=res
        self.n_elips=n_elips
        self.epoch_size=epoch_size
        self.sigma=sigma

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _):
        max_elips_size = self.res/5
        TSDF = np.ones((self.res,self.res,self.res))*max_elips_size
        for i in range(self.n_elips):
            x,y,z = np.random.randint(3,max_elips_size),np.random.randint(3,max_elips_size),np.random.randint(3,max_elips_size)

            el = ellipsoid(x,y,z,levelset=True)
            el*=np.min([x,y,z]) #scikit returns values vaguely [-1:1]. 
            assert np.abs(el).max()<self.res
            x,y,z = el.shape
            x0,y0,z0 = (np.random.rand(3)*(self.res-np.array(el.shape)-1)).astype(int)
            tmp = np.minimum(TSDF[x0:x0+x, y0:y0+y,z0:z0+z], el)
            assert tmp.shape == TSDF[x0:x0+x, y0:y0+y,z0:z0+z].shape, (tmp.shape ,  TSDF[x0:x0+x, y0:y0+y,z0:z0+z].shape)

            TSDF[x0:x0+x, y0:y0+y,z0:z0+z] = tmp
        return TSDF, (TSDF+np.random.randn(self.res,self.res,self.res)*self.sigma)[None,:]

