import numpy as np
from skimage import measure
from skimage.draw import ellipsoid
from scipy.spatial import KDTree
import visdom

def plotVoxelVisdom(GT_voxels, voxels, visdom, title):
    v,f,_,_ =  measure.marching_cubes_lewiner(voxels, level=0., allow_degenerate=False)
    v_gt,f_gt,_,_ =  measure.marching_cubes_lewiner(GT_voxels, level=0., allow_degenerate=False)
    kd = KDTree(v_gt)
    d,ind = kd.query(v)
    assert len(d)==len(v)
    visdom.histogram(d, win=0, opts=dict(title='hist errors'))
    visdom.mesh(X=v, Y=f, win=1,opts=dict(opacity=1., title=title))

def generate_tsdf(vox_size=512, n_elips=10, sigma=0.):
    TSDF = np.ones((vox_size,vox_size,vox_size))*20
    for i in range(n_elips):
        el = ellipsoid(np.random.randint(1,40),np.random.randint(1,40), np.random.randint(1,40),levelset=True)
        x,y,z = el.shape
        x0,y0,z0 = (np.random.rand(3)*(vox_size-np.array(el.shape))).astype(int)
        TSDF[x0:x0+x, y0:y0+y,z0:z0+z] = np.minimum(TSDF[x0:x0+x, y0:y0+y,z0:z0+z], el)
    return TSDF, TSDF+np.random.randn(vox_size,vox_size,vox_size)*sigma

if __name__=='__main__':
    vis = visdom.Visdom()
    GT, TSDF = generate_tsdf(vox_size=128, n_elips=3, sigma=0.05)
    plotVoxelVisdom(GT, TSDF,vis,'ellipsoid')
