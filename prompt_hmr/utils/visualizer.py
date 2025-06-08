import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk 


def plot_10_kpts(kpts, c='pink', markersize=6, markeredgewidth=0.5, 
                   linewidth=3, thresh=0.2, ax=None):
    x, y, v = kpts.t().numpy()
    sks = [[0,1],[2,3],[4,5],[6,7],[8,9]]
    
    if ax is None:
        plot = plt
    else:
        plot = ax

    for sk in sks:
        if np.all(v[sk]>thresh):
            plot.plot(x[sk],y[sk], linewidth=linewidth, color=c)

    plot.plot(x[v>thresh], y[v>thresh],'o',markersize=markersize, markerfacecolor=c, 
            markeredgecolor='k',markeredgewidth=markeredgewidth)
    

def plot_coco_kpts(kpts, c='pink', markersize=6, markeredgewidth=0.5, 
                   linewidth=3, thresh=0.2, ax=None):
    x, y, v = kpts.t().numpy()
    sks = [[0,13],[0,14],[13,14],[13,15],[14,16],
           [1,15],[1,2],[2,3],[4,16],[4,5],[5,6],
           [1,4],[1,7],[4,10],[7,10],
           [7,8],[8,9],[10,11],[11,12]]
    
    if ax is None:
        plot = plt
    else:
        plot = ax

    for sk in sks:
        if np.all(v[sk]>thresh):
            plot.plot(x[sk],y[sk], linewidth=linewidth, color=c)

    plot.plot(x[v>thresh], y[v>thresh],'o',markersize=markersize, markerfacecolor=c, 
            markeredgecolor='k',markeredgewidth=markeredgewidth)


def draw_coco_kpts(img, kpts, r=5, markersize=5, color=0, confidence=1e-6,
                   linewidth=1, edgewidth=5):
    if isinstance(img, np.ndarray):
        img = img.copy().astype(np.uint8)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        img = img.copy().astype(np.uint8)

    if isinstance(kpts, torch.Tensor):
        kpts = kpts.numpy()

    if type(color) == int:
        colors = [[245,195,203]]
        color = colors[color]

    # Skeleton
    x, y, v = kpts.transpose()
    x, y = x.astype(int), y.astype(int)
    sks = [[0,13],[0,14],[13,14],[13,15],[14,16],
           [1,15],[1,2],[2,3],[4,16],[4,5],[5,6],
           [1,4],[1,7],[4,10],[7,10],
           [7,8],[8,9],[10,11],[11,12]]
    for sk in sks:
        if np.all(v[sk]>confidence):
            cv2.line(img, (x[sk[0]], y[sk[0]]), 
                    (x[sk[1]], y[sk[1]]), color=color, thickness=linewidth)
    
    # Marker
    for kpt in kpts:
        x, y, v = kpt
        if v > confidence:
            cv2.circle(img, (int(x), int(y)), r, [30,30,30], markersize+edgewidth)
            cv2.circle(img, (int(x), int(y)), r, color, markersize)

    return img

def draw_kpts(img, kpts, r=5, thickness=5, color=(255,0,0), confidence=1e-6):
    if isinstance(img, np.ndarray):
        img = img.copy().astype(np.uint8)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        img = img.copy().astype(np.uint8)
        
    for kpt in kpts:
        if len(kpt)>2:
            x, y, c = kpt
        else:
            x, y = kpt
            c = 1

        if c >= confidence:
            cv2.circle(img, (int(x), int(y)), r, color, thickness)

    return img

def draw_boxes(img, boxes, thickness=5, color=(0,255,0), numbered=False):
    img_box = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        img_box = cv2.rectangle(img_box, (int(x1),int(y1)), (int(x2),int(y2)), 
                                color, thickness)
        
        if numbered:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            img_box = cv2.putText(img_box, f'{i}', [int(x1),int(y1)-5], font, font_scale, 
                                  color=(255,0,0), thickness=thickness+2)
    return img_box

def draw_masks(img, masks):
    a = 0.6
    colors = [[30, 255, 80], [30, 144, 255], [80, 80, 80]]
    img_msk = img.copy()

    for i, msk in enumerate(masks):
        try:
            color = colors[i]
        except Exception:
            color = colors[-1]

        msking = np.zeros_like(img_msk)
        msking[:,:] = color
        img_msk[msk] = (img_msk[msk] * (1-a) + msking[msk] * a)
        img_msk = img_msk.astype(np.uint8)

        boundary = find_boundaries(msk, mode='thick')[...,0] 
        boundary = dilation(boundary, disk(1.8))
        img_msk[boundary] = np.ones(3) * 80
       
    return img_msk

def to_rgb(grey, cmap='YlGnBu', resize=[224, 224], normalize=True):
    # cmap_list = ['YlGnBu', 'coolwarm', 'RdBu']
    g = np.array(grey)
    cmap = cm.get_cmap(cmap)

    if normalize:
        norm = Normalize(vmin=g.min(), vmax=g.max())
        g = norm(g)
    rgb = cmap(g)[:,:,:3]

    if resize is not None:
        rgb = cv2.resize(rgb, resize)

    rgb = (rgb * 255).astype(int)
    return rgb


def to_rgb_norm(grey, cmap='YlGnBu', resize=[224, 224], min_v=0.0, max_v=1.0, normalize=True):
    # cmap_list = ['YlGnBu', 'coolwarm', 'RdBu']
    g = np.array(grey)
    cmap = cm.get_cmap(cmap)

    if normalize:
        norm = Normalize(vmin=min_v, vmax=max_v)
        g = norm(g)
    rgb = cmap(g)[:,:,:3]

    if resize is not None:
        rgb = cv2.resize(rgb, resize)

    rgb = (rgb * 255).astype(int)
    return rgb

    
### Save for visualization
def save_ply(vert, face=None, color=None, filename='file.ply'):

    # Colors
    if color is None:
        vtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    else:
        vtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        vert = np.concatenate([vert, color], axis=-1)

    # Vertices
    if isinstance(vert, np.ndarray):
        vert = vert.tolist()
    vert = [tuple(v) for v in vert]
    vert = np.array(vert, dtype=vtype)
    vert = PlyElement.describe(vert, 'vertex')

    # Faces
    if face is not None:
        if isinstance(face, np.ndarray):
            face = face.tolist()
        face = [(face[i], 255, 255, 255) for i in range(len(face))]
        face = np.array(face, dtype=[('vertex_indices', 'i4', (3,)),
                                     ('red', 'u1'),
                                     ('green', 'u1'),
                                     ('blue', 'u1')])
        face = PlyElement.describe(face, 'face')
    
    # Save
    if face is not None:
        with open(filename, 'wb') as f:
            PlyData([vert, face]).write(f)
    else:
        with open(filename, 'wb') as f:
            PlyData([vert]).write(f)


def read_ply(plyfile):
    plydata = PlyData.read(plyfile)
    v = plydata['vertex'].data
    v = [list(i) for i in v]
    v = np.array(v)
    f = plydata['face'].data
    f = [list(i) for i in f]
    f = np.array(f).squeeze()
    return v, f


# from transforms3d.euler import euler2mat

# def novel_view(s_out, angle=-0.25*np.pi, axis='y', trans=8):
#     vertices = s_out.vertices.clone().cpu()
#     joints = s_out.joints.clone().cpu()

#     j3d = joints[:, 25:]
#     pelvis3d = j3d[:, [14]]
#     verts = vertices - pelvis3d

#     if axis=='x':
#         rot = euler2mat(angle, 0, 0, "sxyz")
#     elif axis=='y':
#         rot = euler2mat(0, angle, 0, "sxyz")
#     else:
#         rot = euler2mat(0, 0, angle, "sxyz")

#     rot = torch.from_numpy(rot).float()
#     verts = torch.einsum('ij, bvj->bvi', rot, verts)

#     # verts[:,:,1] -= 0.1
#     verts[:,:,2] += trans

#     return verts
        

