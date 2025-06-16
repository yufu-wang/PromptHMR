import os
import json
import base64
from typing import List, Dict, Any, Union, Tuple
import numpy as np
import scipy as sp
from numpy.typing import NDArray

from gloss import ViewerDummy

from smpl_rs import SmplCache
from smpl_rs.codec import McsCodec, GltfCodec
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender, GltfCompatibilityMode
from smpl_rs.components import GlossInterop


def create_gltf_structure(num_frames: int) -> Dict[str, Any]:
    """Create the initial structure of the GLTF file.
    
    Args:
        num_frames: Number of frames in the animation.
        
    Returns:
        Dict containing the basic GLTF structure.
    """
    return {
        "asset": {
            "version": "2.0",
            "generator": "Custom Python Exporter"
        },
        "scene": 0,
        "scenes": [
            {
                "nodes": [0],
                "extensions": {
                    "MC_scene_description": {
                        "num_frames": num_frames,
                        "smpl_bodies": []
                    }
                }
            }
        ],
        "nodes": [
            {
                "name": "RootNode",
                "children": [1]
            },
            {
                "name": "AnimatedCamera",
                "camera": 0,
                "translation": [0.0, 0.0, 0.0],  # Placeholder translation
                "rotation": [0.0, 0.0, 0.0, 1.0]  # Identity rotation
            }
        ],
        "cameras": [
            {
                "type": "perspective",
                "perspective": {}
            }
        ],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
        "animations": [],
        "extensionsUsed": ["MC_scene_description"]
    }

def add_camera_intrinsics(
    gltf: Dict[str, Any],
    focal_length: float,
    principal_point: Tuple[float, float]
) -> None:
    """Set the perspective camera intrinsics using focal length and principal point.
    
    Args:
        gltf: The GLTF structure to modify.
        focal_length: Camera focal length.
        principal_point: Tuple of (x, y) principal point coordinates.
    """
    img_width, img_height = principal_point[0] * 2, principal_point[1] * 2
    yfov = 2 * np.arctan(img_height / (2 * focal_length))
    aspect_ratio = img_width / img_height

    gltf["cameras"][0]["perspective"] = {
        "yfov": float(yfov),
        "znear": 0.01, # We do not set zfar, for infinite perspective camera
        # "zfar": 100.0,
        "aspectRatio": float(aspect_ratio)
    }


def add_smpl_buffers_to_gltf(
    gltf: Dict[str, Any],
    smpl_buffers: List[bytes],
    frame_presences: List[List[int]]
) -> None:
    """Add SMPL buffers and corresponding metadata to the GLTF structure.
    
    Args:
        gltf: The GLTF structure to modify.
        smpl_buffers: List of SMPL buffer data.
        frame_presences: List of frame presence ranges for each buffer.
    """
    for i, (smpl_buffer, frame_presence) in enumerate(zip(smpl_buffers, frame_presences)):
        smpl_base64 = base64.b64encode(smpl_buffer).decode('utf-8')
        gltf["buffers"].append({
            "byteLength": len(smpl_buffer),
            "uri": f"data:application/octet-stream;base64,{smpl_base64}"
        })
        gltf["bufferViews"].append({
            "buffer": i,
            "byteOffset": 0,
            "byteLength": len(smpl_buffer)
        })
        gltf["scenes"][0]["extensions"]["MC_scene_description"]["smpl_bodies"].append({
            "frame_presence": frame_presence,
            "bufferView": i
        })

def add_camera_animation(
    gltf: Dict[str, Any],
    rotation_matrices: NDArray[np.float32],
    translations: NDArray[np.float32],
    num_frames: int,
    frame_rate: float
) -> None:
    """Add animation data for the camera node to the GLTF structure.
    
    Args:
        gltf: The GLTF structure to modify.
        rotation_matrices: Array of shape (N,3,3) containing camera rotation matrices.
        translations: Array of shape (N,3) containing camera translations.
        num_frames: Number of frames in the animation.
        frame_rate: Animation frame rate.
    """
    times = np.arange(num_frames, dtype=np.float32) * (1.0 / frame_rate)

    # Convert from CV to glTF convention
    # CV camera: +X right, +Y down, +Z forward (looking direction)
    # glTF camera: +X right, +Y up, -Z forward (looking direction)
    cv_to_gltf = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],  # Flip Y (down to up)
        [0.0,  0.0, -1.0]   # Flip Z (forward to backward)
    ])

    camera_position = np.zeros((num_frames, 3), dtype=np.float32)
    rotations = np.zeros((num_frames, 4), dtype=np.float32)

    for i in range(num_frames):
        rotation = rotation_matrices[i]
        translation = translations[i]

        camera_position[i] = -rotation.T @ translation

        rotation_gltf = rotation.T @ cv_to_gltf
        rot = sp.spatial.transform.Rotation.from_matrix(rotation_gltf)
        rotations[i] = rot.as_quat()

    buffers_start_idx = len(gltf["buffers"])
    gltf["buffers"].extend([
        {
            "byteLength": times.nbytes,
            "uri": (f"data:application/octet-stream;base64,"
                   f"{base64.b64encode(times.tobytes()).decode('utf-8')}")
        },
        {
            "byteLength": camera_position.nbytes, 
            "uri": (f"data:application/octet-stream;base64,"
                   f"{base64.b64encode(camera_position.tobytes()).decode('utf-8')}")
        },
        {
            "byteLength": rotations.nbytes,
            "uri": (f"data:application/octet-stream;base64,"
                   f"{base64.b64encode(rotations.tobytes()).decode('utf-8')}")
        }
    ])

    gltf["bufferViews"].extend([
        {
            "name": "TimeBufferView",
            "buffer": buffers_start_idx,
            "byteOffset": 0,
            "byteLength": times.nbytes
        },
        {
            "name": "camera_track_translations_buffer_view",
            "buffer": buffers_start_idx + 1,
            "byteOffset": 0,
            "byteLength": camera_position.nbytes
        },
        {
            "name": "camera_track_rotations_buffer_view",
            "buffer": buffers_start_idx + 2,
            "byteOffset": 0,
            "byteLength": rotations.nbytes
        }
    ])

    gltf["accessors"].extend([
        {
            "name": "TimeAccessor",
            "bufferView": len(gltf["bufferViews"]) - 3,
            "componentType": 5126,
            "count": num_frames,
            "type": "SCALAR",
            "min": [float(times.min())],
            "max": [float(times.max())]
        },
        {
            "name": "camera_track_translations_accessor",
            "bufferView": len(gltf["bufferViews"]) - 2,
            "componentType": 5126,
            "count": num_frames,
            "type": "VEC3"
        },
        {
            "name": "camera_track_rotations_accessor",
            "bufferView": len(gltf["bufferViews"]) - 1,
            "componentType": 5126,
            "count": num_frames,
            "type": "VEC4"
        }
    ])

    gltf["animations"].append({
        "channels": [
            {
                "sampler": 0,
                "target": {"node": 1, "path": "translation"}
            },
            {
                "sampler": 1,
                "target": {"node": 1, "path": "rotation"}
            }
        ],
        "samplers": [
            {
                "input": len(gltf["accessors"]) - 3,
                "interpolation": "LINEAR",
                "output": len(gltf["accessors"]) - 2
            },
            {
                "input": len(gltf["accessors"]) - 3,
                "interpolation": "LINEAR",
                "output": len(gltf["accessors"]) - 1
            }
        ]
    })


def write_gltf_to_file(gltf: Dict[str, Any], output_path: str) -> None:
    """Write the GLTF structure to a file.
    
    Args:
        gltf: The GLTF structure to write.
        output_path: Path to write the GLTF file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gltf, f, indent=2)


def convert_mcs_to_gltf(mcs_path: str, gltf_path: str, smplx_path: str) -> None:
    """Convert an MCS file to a GLTF file.
    
    Args:
        mcs_path: Path to the MCS file.
        gltf_path: Path to write the GLTF file.
    """
    viewer = ViewerDummy()    

    mcs_codec = McsCodec.from_file(mcs_path)

    print("\nInformation from the MCS file:")
    print(f"Number of frames: {mcs_codec.num_frames}")
    print(f"Number of bodies: {mcs_codec.num_bodies}")
    print(f"Has camera: {mcs_codec.has_camera}")
    print(f"Frame rate: {mcs_codec.frame_rate}")

    entity_builders = mcs_codec.to_entity_builders()

    for current_ent, builder in enumerate(entity_builders):
        entity = viewer.get_or_create_entity(name=f"mcs_entity_{current_ent}")
        entity.insert(builder)
        interop = GlossInterop(with_uv=True)
        entity.insert(interop)

    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, smplx_path)

    viewer.add_resource(smpl_models)
    viewer.insert_plugin(SmplPlugin(autorun=False))
    viewer.run_manual_plugins()

    # Create the writer and export as Glb
    gltf_codec = GltfCodec.from_scene(viewer.get_scene().ptr_idx(), export_camera = True)
    gltf_codec.save(gltf_path, GltfCompatibilityMode.Unreal)
    print(f"glTF file exported to {os.path.abspath(gltf_path)}")
    

def export_scene_with_camera(
    smpl_buffers: List[bytes],
    frame_presences: List[List[int]],
    num_frames: int,
    output_path: str,
    rotation_matrices: NDArray[np.float32],
    translations: NDArray[np.float32],
    focal_length: float,
    principal_point: Tuple[float, float],
    frame_rate: float,
    smplx_path: str
) -> None:
    """Export the GLTF file with SMPL bodies and animated camera.
    
    Args:
        smpl_buffers: List of SMPL buffer data.
        frame_presences: List of frame presence ranges for each buffer.
        num_frames: Number of frames in the animation.
        output_path: Path to write the GLTF file.
        rotation_matrices: Array of shape (N,3,3) containing camera rotation matrices.
        translations: Array of shape (N,3) containing camera translations.
        focal_length: Camera focal length.
        principal_point: Tuple of (x, y) principal point coordinates.
        frame_rate: Animation frame rate.
    """
    gltf = create_gltf_structure(num_frames)
    add_camera_intrinsics(gltf, focal_length, principal_point)
    add_smpl_buffers_to_gltf(gltf, smpl_buffers, frame_presences)
    add_camera_animation(gltf, rotation_matrices, translations, num_frames, frame_rate)
    write_gltf_to_file(gltf, output_path)
    print(f"MCS file exported to {os.path.abspath(output_path)}")
    convert_mcs_to_gltf(output_path, output_path.replace(".mcs", ".glb"), smplx_path)
