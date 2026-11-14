"""Render one inventory icon from a prop model: the object alone on a transparent background.

Run headless:
    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --python render_item_icon.py --
        --model <path to .glb, .gltf or .blend> --out <path to .png> [--yaw <deg>] [--pitch <deg>]

Arguments (all after the lone `--`):
    --model    The model to picture: a .glb / .gltf, or a .blend (its objects are appended into an
               empty scene; cameras and lights saved in it are left out). The file is only read.
    --out      The .png to write: square, RGBA, transparent everywhere but the object.
    --yaw      Where the camera stands around the model, in degrees. 0 looks at the model's front
               (its -Y side); a positive yaw walks the camera towards the model's +X end.
               Default 30.
    --pitch    How far above the model the camera stands, in degrees. 0 is level, 90 is straight
               down. Default 30.
    --roll     Turns the object in the picture, anticlockwise, in degrees: lays a long thin thing
               along the diagonal so it is drawn bigger. Default 0.
    --fill     How much of the frame the object's longer side takes, 0 to 1. Default 0.8.
    --size     Side of the picture in pixels. Default 512.
    --samples  Cycles samples. Default 256.

The camera is framed from the model's own bounds, so any model fits whatever its size, and the
lights and the room the metal mirrors are sized from the same bounds and follow the camera: the
key light always comes from the picture's top left. There is no floor, so no shadow under the
object.
"""

import argparse
import math
import os
import sys

import bpy
import numpy
from mathutils import Matrix, Vector

#: Object types that can be rendered and measured as a mesh.
SOLID_TYPES = {"MESH", "CURVE", "SURFACE", "FONT", "META"}

#: A long lens: nearly no perspective, as on a product shot.
CAMERA_LENS = 100.0
CAMERA_SENSOR = 36.0

#: Lights, as (direction seen from the object in camera terms (right, up, towards the camera),
#: distance and side in bounding radii, strength in watts per square metre of distance).
KEY_LIGHT = ((-0.62, 0.68, 0.40), 6.0, 6.0, 20.0)
FILL_LIGHT = ((0.80, 0.10, 0.60), 6.0, 9.0, 9.0)

#: The room the metal mirrors, from straight down to straight up, as (height 0 to 1, brightness):
#: a pale floor, a dark line at the horizon, a bright wall above it, a softer ceiling. Polished
#: metal reads as metal by what it mirrors; a plain even room would leave it flat grey.
ROOM_STOPS = (
    (0.00, 0.40),
    (0.30, 0.50),
    (0.45, 0.42),
    (0.50, 0.08),
    (0.55, 0.12),
    (0.63, 0.55),
    (0.78, 0.85),
    (0.90, 0.55),
    (1.00, 0.45),
)
#: How much brighter the room is towards the key light than away from it.
ROOM_KEY_BIAS = 0.35


def read_arguments() -> argparse.Namespace:
    """
    Read what follows the lone `--` on Blender's command line

    :return: The arguments
    """
    own = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="render_item_icon.py", description="Render one inventory icon from a prop model.")
    parser.add_argument("--model", required=True, help="the .glb, .gltf or .blend to picture")
    parser.add_argument("--out", required=True, help="the .png to write")
    parser.add_argument("--yaw", type=float, default=30.0, help="camera angle around the model, degrees (0 = front, the -Y side)")
    parser.add_argument("--pitch", type=float, default=30.0, help="camera angle above the model, degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="turn the object anticlockwise in the picture, degrees")
    parser.add_argument("--fill", type=float, default=0.8, help="share of the frame the object's longer side takes")
    parser.add_argument("--size", type=int, default=512, help="side of the picture in pixels")
    parser.add_argument("--samples", type=int, default=256, help="Cycles samples")
    return parser.parse_args(own)


def load_model(path: str) -> None:
    """
    Bring the model into the empty scene, without its own cameras and lights

    :param path: A .glb, .gltf or .blend file
    """
    extension = os.path.splitext(path)[1].lower()
    scene = bpy.context.scene
    if extension in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif extension == ".blend":
        # Appended, not opened: the file on disk can never be saved over from here.
        with bpy.data.libraries.load(path, link=False) as (source, target):
            target.objects = list(source.objects)
        for thing in target.objects:
            if thing is not None:
                scene.collection.objects.link(thing)
    else:
        raise SystemExit(f"[render_item_icon] cannot read a '{extension}' model, give a .glb, .gltf or .blend")

    for thing in list(scene.objects):
        if thing.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(thing, do_unlink=True)


def model_points() -> numpy.ndarray:
    """
    Every point of everything that will be rendered, where it stands in the world

    :return: An N by 3 array of positions
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    clouds = []
    for thing in bpy.context.scene.objects:
        if thing.type not in SOLID_TYPES or thing.hide_render:
            continue
        shaped = thing.evaluated_get(depsgraph)
        mesh = shaped.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            continue
        flat = numpy.empty(len(mesh.vertices) * 3, dtype=numpy.float64)
        mesh.vertices.foreach_get("co", flat)
        shaped.to_mesh_clear()
        local = flat.reshape(-1, 3)
        world = numpy.array(shaped.matrix_world)
        clouds.append(local @ world[:3, :3].T + world[:3, 3])
    if not clouds:
        raise SystemExit("[render_item_icon] the model holds nothing to render")
    return numpy.concatenate(clouds)


def camera_axes(yaw: float, pitch: float, roll: float) -> tuple:
    """
    The camera's own axes for a view

    :param yaw: Degrees around the model, 0 at its -Y side
    :param pitch: Degrees above it
    :param roll: Degrees the object turns anticlockwise in the picture
    :return: The camera's right, up and backward directions, as unit Vectors
    """
    yaw, pitch, roll = math.radians(yaw), math.radians(pitch), math.radians(roll)
    back = Vector((math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)))
    level = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    upright = back.cross(level)
    # The object turns one way in the picture when the camera turns the other way.
    right = level * math.cos(roll) - upright * math.sin(roll)
    up = level * math.sin(roll) + upright * math.cos(roll)
    return right, up, back


def frame_camera(points: numpy.ndarray, axes: tuple, fill: float) -> tuple:
    """
    Find where the camera stands so the model sits in the middle of the picture at the wanted size

    :param points: The model's points in the world
    :param axes: The camera's right, up and backward directions
    :param fill: Share of the frame the model's longer side takes
    :return: The camera's position, and its distance to the model's middle
    """
    right, up, back = (numpy.array(axis) for axis in axes)
    tangent = (CAMERA_SENSOR / 2.0) / CAMERA_LENS  # half the frame, at one metre from the camera
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = float(numpy.linalg.norm(points - centre, axis=1).max())
    distance = radius / tangent / fill
    position = centre + back * distance

    # With perspective the outline changes as the camera moves, so close in on it step by step.
    for _ in range(40):
        offset = points - position
        depth = -(offset @ back)
        across = (offset @ right) / depth
        upward = (offset @ up) / depth
        middle_across = (across.min() + across.max()) / 2.0
        middle_upward = (upward.min() + upward.max()) / 2.0
        span = max(across.max() - across.min(), upward.max() - upward.min())
        scale = span / (2.0 * tangent * fill)
        distance = float((position - centre) @ back)
        position = position + right * middle_across * distance + up * middle_upward * distance
        position = position + back * distance * (scale - 1.0)
        if abs(scale - 1.0) < 1e-6 and abs(middle_across) < 1e-6 and abs(middle_upward) < 1e-6:
            break
    return Vector(position), float((position - centre) @ back)


def place(thing: object, position: Vector, axes: tuple) -> None:
    """
    Put a camera or a light somewhere, turned to the given axes (both look down their own -Z)

    :param thing: The object to place
    :param position: Where it goes
    :param axes: Its right, up and backward directions
    """
    right, up, back = axes
    matrix = Matrix((right, up, back)).transposed().to_4x4()
    matrix.translation = position
    thing.matrix_world = matrix


def add_camera(position: Vector, axes: tuple, distance: float, radius: float) -> object:
    """
    Make the camera and make it the scene's

    :param position: Where it stands
    :param axes: Its right, up and backward directions
    :param distance: How far the model's middle is
    :param radius: The model's bounding radius
    :return: The camera object
    """
    data = bpy.data.cameras.new("IconCamera")
    data.lens = CAMERA_LENS
    data.sensor_width = CAMERA_SENSOR
    data.sensor_fit = "HORIZONTAL"
    data.clip_start = max(distance - radius * 2.0, distance * 0.01)
    data.clip_end = distance + radius * 2.0
    camera = bpy.data.objects.new("IconCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    place(camera, position, axes)
    bpy.context.scene.camera = camera
    return camera


def add_light(name: str, rig: tuple, centre: Vector, radius: float, axes: tuple) -> Vector:
    """
    Make one soft area light that the camera does not see, aimed at the model

    :param name: Name of the light
    :param rig: One of the light constants above
    :param centre: The model's middle
    :param radius: The model's bounding radius
    :param axes: The camera's right, up and backward directions
    :return: The direction from the model to the light
    """
    (across, upward, towards), reach, side, strength = rig
    right, up, back = axes
    direction = (right * across + up * upward + back * towards).normalized()
    distance = radius * reach

    data = bpy.data.lights.new(name, "AREA")
    data.shape = "DISK"
    data.size = radius * side
    # The power grows with the square of the distance, so a big model is lit as a small one is.
    data.energy = strength * distance * distance
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.visible_camera = False

    aside = direction.cross(Vector((0.0, 0.0, 1.0)))
    if aside.length < 1e-6:
        aside = Vector((1.0, 0.0, 0.0))
    aside.normalize()
    place(light, centre + direction * distance, (aside, direction.cross(aside), direction))
    return direction


def build_room(key_direction: Vector) -> None:
    """
    The world: a soft grey room for the metal to mirror, brighter towards the key light.
    The camera itself never sees it, the film is transparent.

    :param key_direction: The direction from the model to the key light
    """
    world = bpy.data.worlds.new("IconRoom")
    bpy.context.scene.world = world
    if bpy.app.version < (5, 0, 0):
        world.use_nodes = True  # from Blender 5 a world always has its nodes
    tree = world.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputWorld")
    room = tree.nodes.new("ShaderNodeBackground")
    coords = tree.nodes.new("ShaderNodeTexCoord")
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    remap = tree.nodes.new("ShaderNodeMapRange")
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    facing = tree.nodes.new("ShaderNodeVectorMath")
    bias = tree.nodes.new("ShaderNodeMath")

    remap.inputs["From Min"].default_value = -1.0
    remap.inputs["From Max"].default_value = 1.0

    elements = ramp.color_ramp.elements
    while len(elements) < len(ROOM_STOPS):
        elements.new(0.5)
    for element, (position, value) in zip(elements, ROOM_STOPS):
        element.position = position
        element.color = (value, value, value, 1.0)

    facing.operation = "DOT_PRODUCT"
    facing.inputs[1].default_value = key_direction
    bias.operation = "MULTIPLY_ADD"  # facing * bias + 1
    bias.inputs[1].default_value = ROOM_KEY_BIAS
    bias.inputs[2].default_value = 1.0

    tree.links.new(coords.outputs["Generated"], split.inputs["Vector"])
    tree.links.new(split.outputs["Z"], remap.inputs["Value"])
    tree.links.new(remap.outputs["Result"], ramp.inputs["Fac"])
    tree.links.new(coords.outputs["Generated"], facing.inputs[0])
    tree.links.new(facing.outputs["Value"], bias.inputs[0])
    tree.links.new(ramp.outputs["Color"], room.inputs["Color"])
    tree.links.new(bias.outputs["Value"], room.inputs["Strength"])
    tree.links.new(room.outputs["Background"], output.inputs["Surface"])


def set_up_render(size: int, samples: int, out: str) -> None:
    """
    Cycles, a square transparent RGBA picture, colours shown as they are

    :param size: Side of the picture in pixels
    :param samples: Cycles samples
    :param out: The .png to write
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 90
    # Standard, not a film look: the icon keeps the model's own colours and clean whites.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.filepath = out


def report(out: str) -> None:
    """
    Read the picture back and print how the object sits in it

    :param out: The .png that was written
    """
    image = bpy.data.images.load(out)
    width, height = image.size
    pixels = numpy.empty(width * height * 4, dtype=numpy.float32)
    image.pixels.foreach_get(pixels)
    alpha = pixels.reshape(height, width, 4)[:, :, 3]
    columns = numpy.flatnonzero(alpha.max(axis=0) > 0.02)
    rows = numpy.flatnonzero(alpha.max(axis=1) > 0.02)
    if len(columns) == 0:
        print("[render_item_icon] WARNING: the picture is empty")
        return
    left, right = int(columns[0]), int(width - 1 - columns[-1])
    bottom, top = int(rows[0]), int(height - 1 - rows[-1])
    wide, tall = width - left - right, height - top - bottom
    print(f"[render_item_icon] wrote {out}: {width}x{height}, {image.channels} channels, depth {image.depth}")
    print(f"[render_item_icon] object {wide}x{tall} px ({max(wide, tall) / max(width, height):.0%} of the frame), margins left {left} right {right} top {top} bottom {bottom}")


def main() -> None:
    """Load the model, frame it, light it and render the icon."""
    arguments = read_arguments()
    model = os.path.abspath(arguments.model)
    out = os.path.abspath(arguments.out)
    if not os.path.isfile(model):
        raise SystemExit(f"[render_item_icon] no such model: {model}")
    if not 0.05 <= arguments.fill <= 1.0:
        raise SystemExit("[render_item_icon] --fill must be between 0.05 and 1")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    load_model(model)

    points = model_points()
    centre = Vector((points.min(axis=0) + points.max(axis=0)) / 2.0)
    radius = float(numpy.linalg.norm(points - numpy.array(centre), axis=1).max())

    axes = camera_axes(arguments.yaw, arguments.pitch, arguments.roll)
    position, distance = frame_camera(points, axes, arguments.fill)
    add_camera(position, axes, distance, radius)

    key_direction = add_light("IconKey", KEY_LIGHT, centre, radius, axes)
    add_light("IconFill", FILL_LIGHT, centre, radius, axes)
    build_room(key_direction)

    set_up_render(arguments.size, arguments.samples, out)
    bpy.ops.render.render(write_still=True)

    print(f"[render_item_icon] model {model}")
    print(f"[render_item_icon] yaw {arguments.yaw:g} pitch {arguments.pitch:g} roll {arguments.roll:g} fill {arguments.fill:g}")
    report(out)


if __name__ == "__main__":
    main()
