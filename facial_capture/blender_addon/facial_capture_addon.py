"""
Blender 插件 - 面部表情捕捉
从 2D 图像导入 ARKit Blendshapes 到 Blender

安装方法:
1. 打开 Blender
2. Edit > Preferences > Add-ons > Install
3. 选择此文件
4. 启用 "Import-Export: Facial Expression Capture"
"""

bl_info = {
    "name": "Facial Expression Capture",
    "author": "PromptHMR Team",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Facial Capture",
    "description": "从 2D 图像导入面部表情到 ARKit Blendshapes",
    "category": "Import-Export",
}

import bpy
import json
import sys
import os
from pathlib import Path
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy.types import Operator, Panel, AddonPreferences


# 添加 facial_capture 模块到 Python 路径
def get_facial_capture_path():
    """获取 facial_capture 模块路径"""
    addon_dir = Path(__file__).parent.parent
    return str(addon_dir)


class FacialCapturePreferences(AddonPreferences):
    """插件首选项"""
    bl_idname = __name__

    python_exe: StringProperty(
        name="Python Executable",
        description="Python 可执行文件路径（需要安装 mediapipe）",
        default=sys.executable,
        subtype='FILE_PATH'
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "python_exe")
        layout.label(text="确保已安装: pip install mediapipe opencv-python numpy")


class FACIALCAPTURE_OT_extract_from_image(Operator):
    """从图像提取面部表情"""
    bl_idname = "facialcapture.extract_from_image"
    bl_label = "Extract Expression from Image"
    bl_description = "从图像文件提取面部表情"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(
        name="Image File",
        description="输入图像文件",
        subtype='FILE_PATH'
    )

    filter_glob: StringProperty(
        default="*.jpg;*.jpeg;*.png;*.bmp",
        options={'HIDDEN'}
    )

    apply_to_active: BoolProperty(
        name="Apply to Active Object",
        description="将表情应用到当前选中的对象",
        default=True
    )

    def execute(self, context):
        if not self.filepath:
            self.report({'ERROR'}, "未选择图像文件")
            return {'CANCELLED'}

        try:
            # 导入模块
            facial_capture_path = get_facial_capture_path()
            if facial_capture_path not in sys.path:
                sys.path.insert(0, facial_capture_path)

            from facial_capture import FacialExpressionCapture

            # 提取表情
            capturer = FacialExpressionCapture()
            blendshapes = capturer.extract_from_image(self.filepath)

            if blendshapes is None:
                self.report({'ERROR'}, "未检测到面部")
                return {'CANCELLED'}

            # 应用到选中的对象
            if self.apply_to_active and context.active_object:
                self.apply_blendshapes(context.active_object, blendshapes)
                self.report({'INFO'}, f"已应用表情到 {context.active_object.name}")
            else:
                # 保存到场景属性
                context.scene.facial_capture_data = blendshapes.to_dict()
                self.report({'INFO'}, "表情已提取，请使用 'Apply to Object' 应用")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"提取失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    def apply_blendshapes(self, obj, blendshapes):
        """将 blendshapes 应用到对象"""
        if obj.type != 'MESH':
            return

        # 检查是否有 shape keys
        if not obj.data.shape_keys:
            self.report({'WARNING'}, f"{obj.name} 没有 Shape Keys")
            return

        # 应用 blendshapes
        shape_keys = obj.data.shape_keys.key_blocks
        applied_count = 0

        for bs_name, value in blendshapes.to_dict().items():
            # ARKit 名称可能需要映射到 Blender shape key 名称
            # 尝试多种命名约定
            possible_names = [
                bs_name,  # eyeBlinkLeft
                f"ARKit_{bs_name}",  # ARKit_eyeBlinkLeft
                bs_name.replace('eye', 'Eye'),  # EyeBlinkLeft
            ]

            for name in possible_names:
                if name in shape_keys:
                    shape_keys[name].value = value
                    applied_count += 1
                    break

        if applied_count == 0:
            self.report({'WARNING'}, "未找到匹配的 Shape Keys，请检查命名")
        else:
            self.report({'INFO'}, f"已应用 {applied_count} 个 blendshapes")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class FACIALCAPTURE_OT_import_json(Operator):
    """导入 JSON 表情数据"""
    bl_idname = "facialcapture.import_json"
    bl_label = "Import Expression JSON"
    bl_description = "导入 JSON 格式的表情数据"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(
        name="JSON File",
        description="ARKit JSON 文件",
        subtype='FILE_PATH'
    )

    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'}
    )

    create_animation: BoolProperty(
        name="Create Animation",
        description="为序列数据创建动画",
        default=True
    )

    def execute(self, context):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查是否为序列数据
            if 'frames' in data:
                # 序列数据
                if self.create_animation:
                    self.create_animation_from_sequence(context, data)
                else:
                    self.report({'INFO'}, f"加载了 {data['num_frames']} 帧数据")
            else:
                # 单帧数据
                blendshapes_dict = data['blendshapes']
                if context.active_object:
                    self.apply_blendshapes_dict(context.active_object, blendshapes_dict)

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"导入失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    def create_animation_from_sequence(self, context, data):
        """从序列数据创建动画"""
        obj = context.active_object
        if not obj or obj.type != 'MESH' or not obj.data.shape_keys:
            self.report({'ERROR'}, "请选择带有 Shape Keys 的 Mesh 对象")
            return

        fps = data.get('fps', 30.0)
        frames = data['frames']

        # 设置场景帧率
        context.scene.render.fps = int(fps)

        # 创建关键帧
        shape_keys = obj.data.shape_keys.key_blocks

        for frame_data in frames:
            frame_num = frame_data['frame']
            blendshapes = frame_data['blendshapes']

            for bs_name, value in blendshapes.items():
                # 查找对应的 shape key
                possible_names = [bs_name, f"ARKit_{bs_name}"]

                for name in possible_names:
                    if name in shape_keys:
                        shape_key = shape_keys[name]
                        shape_key.value = value
                        shape_key.keyframe_insert(data_path="value", frame=frame_num)
                        break

        # 设置时间轴范围
        context.scene.frame_start = 0
        context.scene.frame_end = frames[-1]['frame']

        self.report({'INFO'}, f"创建了 {len(frames)} 帧的动画")

    def apply_blendshapes_dict(self, obj, blendshapes_dict):
        """应用 blendshapes 字典"""
        if obj.type != 'MESH' or not obj.data.shape_keys:
            return

        shape_keys = obj.data.shape_keys.key_blocks
        applied_count = 0

        for bs_name, value in blendshapes_dict.items():
            possible_names = [bs_name, f"ARKit_{bs_name}"]
            for name in possible_names:
                if name in shape_keys:
                    shape_keys[name].value = value
                    applied_count += 1
                    break

        self.report({'INFO'}, f"已应用 {applied_count} 个 blendshapes")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class FACIALCAPTURE_OT_create_arkit_shapekeys(Operator):
    """创建 ARKit Shape Keys"""
    bl_idname = "facialcapture.create_arkit_shapekeys"
    bl_label = "Create ARKit Shape Keys"
    bl_description = "为选中的 Mesh 创建 52 个 ARKit Shape Keys"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object

        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "请选择一个 Mesh 对象")
            return {'CANCELLED'}

        # ARKit 52 个 blendshape 名称
        arkit_names = [
            'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
            'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
            'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
            'eyeWideLeft', 'eyeWideRight', 'browDownLeft', 'browDownRight',
            'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'mouthClose',
            'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
            'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
            'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
            'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
            'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
            'mouthUpperUpLeft', 'mouthUpperUpRight', 'cheekPuff', 'cheekSquintLeft',
            'cheekSquintRight', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
            'noseSneerLeft', 'noseSneerRight', 'tongueOut'
        ]

        # 创建 basis shape key (如果不存在)
        if not obj.data.shape_keys:
            obj.shape_key_add(name='Basis', from_mix=False)

        # 创建所有 ARKit shape keys
        created = 0
        for name in arkit_names:
            if name not in obj.data.shape_keys.key_blocks:
                obj.shape_key_add(name=name, from_mix=False)
                created += 1

        self.report({'INFO'}, f"创建了 {created} 个 Shape Keys")
        return {'FINISHED'}


class FACIALCAPTURE_PT_panel(Panel):
    """面部表情捕捉面板"""
    bl_label = "Facial Expression Capture"
    bl_idname = "FACIALCAPTURE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Facial Capture'

    def draw(self, context):
        layout = self.layout

        # 提取表情
        box = layout.box()
        box.label(text="提取表情:", icon='TRACKING')
        box.operator("facialcapture.extract_from_image", icon='IMAGE_DATA')
        box.operator("facialcapture.import_json", icon='FILE_FOLDER')

        # Shape Keys 管理
        box = layout.box()
        box.label(text="Shape Keys:", icon='SHAPEKEY_DATA')
        box.operator("facialcapture.create_arkit_shapekeys", icon='ADD')

        # 对象信息
        obj = context.active_object
        if obj and obj.type == 'MESH':
            box.label(text=f"对象: {obj.name}")
            if obj.data.shape_keys:
                box.label(text=f"Shape Keys: {len(obj.data.shape_keys.key_blocks)}")
            else:
                box.label(text="无 Shape Keys", icon='ERROR')


# 注册类
classes = (
    FacialCapturePreferences,
    FACIALCAPTURE_OT_extract_from_image,
    FACIALCAPTURE_OT_import_json,
    FACIALCAPTURE_OT_create_arkit_shapekeys,
    FACIALCAPTURE_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
