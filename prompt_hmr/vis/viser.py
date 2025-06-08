import os.path as osp
import cv2
import copy
import time
import torch
from tqdm import tqdm
import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as R


def get_color(idx):
    colors_path = osp.join(osp.dirname(__file__), 'colors_hsfm.txt')
    colors = np.loadtxt(colors_path).astype(int)
    return colors[idx % len(colors)]


def add_camera_frustm(image, server, quat, trans):
    gui_line_width = server.gui.add_slider(
        "Frustum Line Width", initial_value=2.0, step=0.01, min=0.0, max=20.0
    )

    gui_frustum_scale = server.gui.add_slider(
        "Frustum Scale", initial_value=0.3, step=0.001, min=0.01, max=20.0
    )

    @gui_line_width.on_update
    def _(_) -> None:
        for cam in camera_frustums:
            cam.line_width = gui_line_width.value

    @gui_frustum_scale.on_update
    def _(_) -> None:
        for cam in camera_frustums:
            cam.scale = gui_frustum_scale.value


    # dummy instrinsics
    fov = 0.96
    aspect = 1.7 if image is None else image.shape[1]/image.shape[0]
    
    camera_frustums = []
    camera_frustm = server.scene.add_camera_frustum(
        f"/image",
        fov=fov,
        aspect=aspect,
        scale=gui_frustum_scale.value,
        line_width=gui_line_width.value,
        color=(255, 127, 14),
        wxyz=quat,
        position=trans,
        image=image,
    )
    camera_frustums.append(camera_frustm)


def viser_vis_human(vertices: torch.Tensor, faces: torch.Tensor, 
                    image=None, cameras=None, floor=None, block=True, track_id=None):
    
    if type(vertices) is torch.Tensor:
        vertices = vertices.cpu().numpy()
    if type(faces) is torch.Tensor:
        faces = faces.cpu().numpy()
   
    human_vertices = {i:v for i,v in enumerate(vertices)}
    human_vertices = copy.deepcopy(human_vertices)
    if track_id is not None:
        human_idx = track_id
    else:
        human_idx = [i for i in human_vertices]

    try:
        server.scene.reset()
    except NameError:
        server = viser.ViserServer()

    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    for i, human_name in enumerate(sorted(human_vertices.keys())):
        idx = human_idx[i]
        vertices = human_vertices[human_name]    
        server.scene.add_mesh_simple(
            f"/{human_name}_human/mesh",
            vertices=vertices,
            faces=faces,
            flat_shading=False,
            wireframe=False,
            color=get_color(idx),
        )

    # Add camera in origin
    cam_handles = []
    for cam_name, camera in enumerate(cameras):
        # rotation matrix to quaternion
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        quat = np.concatenate([quat[3:], quat[:3]])
        trans = camera[:3, 3]

        # add camera
        cam_handle = server.scene.add_frame(
            f"/cam_{cam_name}",
            wxyz=quat,
            position=trans,
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.04,
        )
        cam_handles.append(cam_handle)
        add_camera_frustm(image, server, quat, trans)

    # Add floor
    if floor is not None:
        fv, ff = floor
        server.scene.add_mesh_simple(
            f"/floor",
            vertices=fv,
            faces=ff,
            flat_shading=False,
            wireframe=True,
            color=(50, 50, 50),
        )

    if block:
        start_time = time.time()
        while True:
            time.sleep(0.01)
            timing_handle.value = (time.time() - start_time) 
    
    return server


def viser_vis_world4d(images, world4d, faces, init_fps=25, block=False, floor=None,
                      img_maxsize=320):
    try:
        server.scene.reset()
    except NameError:
        server = viser.ViserServer()

    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    num_frames = len(world4d)
    gui_timestep = server.gui.add_slider(
        "Timestep",
        min=0,
        max=num_frames - 1,
        step=1,
        initial_value=0,
        disabled=True,
    )
    gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
    gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
    gui_playing = server.gui.add_checkbox("Playing", True)
    gui_framerate = server.gui.add_slider(
        "FPS", min=1, max=60, step=0.1, initial_value=init_fps
    )
    gui_framerate_options = server.gui.add_button_group(
        "FPS options", ("10", "20", "30", "60")
    )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        # wxyz=vtf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
        wxyz=vtf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    mesh_nodes: list[viser.MeshHandle] = []

    for i in tqdm(range(num_frames)):
        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place meshes in the frame
        track_id = world4d[i]['track_id']
        vertices = world4d[i]['vertices']
        vertices = copy.deepcopy(vertices)
        for tid, verts in zip(track_id, vertices):
            mesh_nodes.append(
                server.scene.add_mesh_simple(
                    name=f"/frames/t{i}/human_{tid}",
                    vertices=verts,
                    faces=faces,
                    flat_shading=False,
                    wireframe=False,
                    color=get_color(tid),
                )
            )

        # Place the frustum.
        image = images[i]
        camera = world4d[i]['camera']
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        quat = np.concatenate([quat[3:], quat[:3]])
        trans = camera[:3, 3]

        if max(image.shape) > img_maxsize:
            scale = img_maxsize / max(image.shape)
            image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        fov = 0.96
        aspect = 1.7 if image is None else image.shape[1]/image.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            line_width=1.5,
            color=(255, 127, 14),
            scale=0.4,
            wxyz=quat,
            position=trans,
            image=image,
        )

        # Add some axes.
        # add_camera_frustm(image, server, quat, trans, f"/frames/t{i}/frustum")
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.3,
            axes_radius=0.02,
        )


    # Add floor
    if floor is not None:
        fv, ff = floor
        server.scene.add_mesh_simple(
            f"/floor",
            vertices=fv,
            faces=ff,
            flat_shading=False,
            wireframe=True,
            color=(50, 50, 50),
        )


    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    if block:
        while True:
            # Update the timestep if we're playing.
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames

            time.sleep(1.0 / gui_framerate.value)
    
    gui = [gui_playing, gui_timestep, gui_framerate, num_frames]

    return server, gui



