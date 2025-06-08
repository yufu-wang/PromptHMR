# Useful rendering functions from WHAM (some modification)
import os
import cv2
import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.renderer.camera_conversions import _cameras_from_opencv_projection

from prompt_hmr.utils.rotation_conversions import rotation_about_x, rotation_about_y
from prompt_hmr.vis.tools import get_colors, checkerboard_geometry
from prompt_hmr.vis.traj import align_meshes_to_ground, align_meshes_to_gravity


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    roi_image[mask] = image[mask]
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype
    
    K = torch.zeros((K_org.shape[0], 4, 4)
    ).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    
    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = (right - left)
    height = (bottom - top)

    new_left = torch.clamp(cx - width/2 * scaleFactor, min=0, max=img_w-1)
    new_right = torch.clamp(cx + width/2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h-1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    
    return bbox


class Renderer():
    def __init__(self, width, height, focal_length, device='cuda', 
                 bin_size=None, max_faces_per_bin=None, img_cx=None, img_cy=None):

        self.width = width
        self.height = height
        self.img_cx = img_cx
        self.img_cy = img_cy
        self.focal_length = focal_length
        self.device = device

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer(bin_size, max_faces_per_bin)

        color_file = os.path.abspath(os.path.join(__file__, "../colors.txt"))
        self.colors = np.loadtxt(color_file)/255
        self.colors = torch.from_numpy(self.colors).float().to(device)
        self.default_color = torch.tensor([[1.0,  1.0,  1.0, 1.0]]).to(device)

    def create_renderer(self, bin_size, max_faces_per_bin):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=0, bin_size=bin_size, 
                    max_faces_per_bin=max_faces_per_bin),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            )
        )

    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = torch.diag(
            torch.tensor([1, 1, 1])
        ).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor(
            [0, 0, 0]
        ).unsqueeze(0).float().to(self.device)

        # Intrinsics
        if self.img_cx is None:
            self.img_cx = self.width/2
        if self.img_cy is None:
            self.img_cy = self.height/2
            
        self.K = torch.tensor(
            [[self.focal_length, 0, self.img_cx],
            [0, self.focal_length, self.img_cy],
            [0, 0, 1]]
        ).unsqueeze(0).float().to(self.device)
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)

        # self.K_full = self.K  # test
        self.cameras = self.create_camera()

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R, #.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)
    
    def create_camera_from_cv(self, R=None, T=None, K=None, image_size=None):
        # R: [1, 3, 3] Tensor
        # T: [1, 3] Tensor
        # K: [1, 3, 3] Tensor
        # image_size: [1, 2] Tensor in HW
        if R is None:
            R = torch.eye(3)[None].float().to(self.device)
        if T is None:
            T = torch.zeros([1,3]).float().to(self.device)
        if K is None:
            K = self.K
        if image_size is None:
            image_size = torch.tensor(self.image_sizes)

        cameras = _cameras_from_opencv_projection(R, T, K, image_size)
        lights = PointLights(device=K.device, location=T)

        return cameras, lights
               
    def set_ground(self, length, center_x, center_z):
        device = self.device
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """ Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self,):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, faces, background, colors=[0.8, 0.8, 0.8], verts_colors=None, shininess=0):
        B = vertices.shape[0]
        F = faces.shape[0]
        faces = faces.unsqueeze(0).expand(B, F, -1)

        if colors[0] > 1: 
            colors = [c / 255. for c in colors]

        if verts_colors is None:
            verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
            verts_features = verts_features.repeat(1, vertices.shape[1], 1)
            textures = TexturesVertex(verts_features=verts_features)
        else:
            textures = TexturesVertex(verts_features=verts_colors)
        
        mesh = Meshes(verts=vertices,
                      faces=faces,
                      textures=textures,)
        
        materials = Materials(
            device=self.device,
            shininess=shininess
            )

        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3
        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
      
        return image

    def render_meshes(self, verts_list, faces, background, default_color=True,
                      specular=(0.0, 0.5, 1.2)):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts_, faces_, colors_ = [], [], []
        for i, verts in enumerate(verts_list):
            if default_color:
                colors = self.default_color
            else:
                colors = self.colors[[i]]

            verts_i, faces_i, colors_i = prep_shared_geometry(verts, faces, colors)
            if i == 0:
                verts_ = list(torch.unbind(verts_i, dim=0))
                faces_ = list(torch.unbind(faces_i, dim=0)) 
                colors_ = list(torch.unbind(colors_i, dim=0))
            else:
                verts_ += list(torch.unbind(verts_i, dim=0))
                faces_ += list(torch.unbind(faces_i, dim=0)) 
                colors_ += list(torch.unbind(colors_i, dim=0)) 

        mesh = create_meshes(verts_, faces_, colors_)

        if default_color:
            materials = Materials(
                device=self.device,
                ambient_color=((0.8, 0.8, 0.8),),
                diffuse_color=((1.1, 1.1, 1.1),),
                specular_color=(specular,), # or (0.0, 1.1, 0.65)
                shininess=0
            )
        else:
            materials = Materials(
                device=self.device,
                shininess=0
            )
            
        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())

        return image
    
    def render_meshes_with_ground(self, verts, faces, colors=None, 
                                  cameras=None, lights=None, floor_scale=2,
                                  gravity_cam=None, topview=False):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        if colors is None:
            colors = self.colors[:len(verts)]
            
        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts, faces, colors = prep_shared_geometry(verts, faces, colors)

        # Floor
        floor_color = [[0.73, 0.78, 0.82], [0.61, 0.69, 0.72]]
        if gravity_cam is not None:
            verts, [gv, gf, gc], R, T = align_meshes_to_gravity(verts, 
                                                                gravity_cam,
                                                                floor_scale=floor_scale, 
                                                                floor_color=floor_color)
        else:
            verts, [gv, gf, gc], R, T = align_meshes_to_ground(verts, 
                                                               floor_scale=floor_scale, 
                                                               floor_color=floor_color)
            
        # View Camera: top view
        if topview:
            centers = verts.mean(dim=1)
            _, idx = (centers - T).norm(dim=-1).sort()
            scene_center = centers[idx][:5].mean(dim=0)
            rx = rotation_about_x(-3.14/8).cuda()[:3,:3]
            ry = rotation_about_y(3.14/8).cuda()[:3,:3]
            r = ry @ rx
            R = r @ R
            T = r @ (T - scene_center) + scene_center

        # View Camera
        if (cameras is None) or (lights is None):
            cam_r = R.mT
            cam_t = - cam_r @ T
            cameras, lights = self.create_camera_from_cv(R=cam_r[None], T=cam_t[None])

        # Render background with floor
        floor = create_meshes([gv], [gf], [gc[..., :3]/150])
        materials = Materials(device=self.device, shininess=0)
        background = self.renderer(floor, cameras=cameras, lights=lights, materials=materials)
        background = background[0,...,:3].cpu().numpy()
        background = (background * 255).astype(np.uint8)

        # (V, 3), (F, 3), (V, 3)
        verts = list(torch.unbind(verts, dim=0)) 
        faces = list(torch.unbind(faces, dim=0)) 
        colors = list(torch.unbind(colors, dim=0)) 
        mesh = create_meshes(verts, faces, colors)

        # materials = Materials(device=self.device, shininess=0)
        materials = Materials(
            device=self.device,
            ambient_color=((0.8, 0.8, 0.8),),
            diffuse_color=((1.1, 1.1, 1.1),),
            specular_color=((0.0, 0.6, 1.2),),
            shininess = 0
        )

        results = self.renderer(mesh, materials=materials, cameras=cameras, lights=lights)
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3
        image = overlay_image_onto_background(image, mask, self.bboxes, background)

        return image


    def render_meshes_with_ground__(self, verts_list, faces, colors_list, cameras, lights, ground=None):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        if ground is None:
            verts_list, [gv, gf, gc] = align_meshes_to_ground(verts_list, floor_scale=2)
        else:
            [gv, gf, gc] = ground

        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts_, faces_, colors_ = [], [], []
        for i, verts in enumerate(verts_list):
            colors = colors_list[[i]]
            verts_i, faces_i, colors_i = prep_shared_geometry(verts, faces, colors)
            if i == 0:
                verts_ = list(torch.unbind(verts_i, dim=0))
                faces_ = list(torch.unbind(faces_i, dim=0)) 
                colors_ = list(torch.unbind(colors_i, dim=0))
            else:
                verts_ += list(torch.unbind(verts_i, dim=0))
                faces_ += list(torch.unbind(faces_i, dim=0)) 
                colors_ += list(torch.unbind(colors_i, dim=0)) 

        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts_ += [gv]
        faces_ += [gf]
        colors_ += [gc[..., :3]]
        mesh = create_meshes(verts_, faces_, colors_)

        materials = Materials(
            device=self.device,
            shininess=0
        )
        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        return image
    

def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 4)
    """
    if type(verts) == np.ndarray:
        verts = torch.from_numpy(verts).cuda().float()
    else:
        verts = verts.cuda().float()

    if type(faces) == np.ndarray:
        faces = torch.from_numpy(faces).cuda().float()
    else:
        faces = faces.cuda().float()

    if type(colors) == np.ndarray or type(colors)== list:
        colors = torch.tensor(colors).float().cuda()[..., :3]
    else:
        colors = colors.float().cuda()[..., :3]

    if verts.dim() == 2:
        verts = verts[None] # (V,3) -> (B,V,3)

    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.reshape(-1, 1, 3).expand(B, V, -1)
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device, distance=5, position=(-5.0, 5.0, 0.0)):
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)
    
    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions
    
    rotation = look_at_rotation(positions, targets, ).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)
    
    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights
