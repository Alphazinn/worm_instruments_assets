"""Build the worm_instr_tiphat prop from nothing.

Run headless:
    blender -b --python make_tiphat.py

Writes, next to this file:
    worm_instr_tiphat.glb     the model the prop converter takes
    worm_instr_tiphat.blend   the same model with the preview cameras, lights and ground
    Textures/*.png            the flat colour pictures the materials wear
    Parts.json                sizes and the points the script drops tips at
    Preview/*.png             four renders

Add `-- --no-render` to skip the previews (the saved scene then has no cameras or lights), or
`-- --draft` to render them small and rough while the shape is being worked on.

A busker's felt hat lying upside down on the ground: it stands on the flat top of its crown, the
opening is up and tips drop into it. Stylised, with no brand, label or text on it.

Axes (metres): X the hat's longer way (front to back of the head), Y its shorter way, Z up. The
origin is on the ground under the middle of the hat: nothing is below Z = 0. The hat is the same
from both sides, so it has no facing; the bow on the band is at +X.

The game's own hat (prop_busker_hat_01) was measured for size only, with Anim-Forge's
`forge inspect-model` and the XML it writes: its felt is 0.333 x 0.263 x 0.145 m, standing on its
crown as this one does, with a 0.22 x 0.17 m opening. Nothing of it is copied.

The materials are plain colours, but each wears a small flat picture: the prop converter rebuilds
every material as a game shader and carries over pictures only, never a colour set on the shader.
"""

import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

NAME = "worm_instr_tiphat"
HERE = os.path.dirname(os.path.abspath(__file__))
PREVIEW_DIR = os.path.join(HERE, "Preview")
TEXTURE_DIR = os.path.join(HERE, "Textures")

#: The geometry below is written in millimetres and scaled to metres as it is built.
MM = 0.001

FELT = 0
BAND = 1
LINING = 2

#: Faces of one smoothing group shade as one curved surface where they meet under this angle.
SMOOTH_ANGLE = math.radians(50.0)

# ---- the hat, in millimetres --------------------------------------------------------------

#: How many sides the hat has round its middle.
SIDES = 48

#: The hat's outline cut through its middle, as rings: half the ring's size along X, half along Y,
#: its height, how far out along the brim it is (0 at the crown, 1 at the brim's edge), then the
#: material and the smoothing group of the band of faces up to the next ring. The rings run from
#: the flat top the hat stands on, up the outside of the crown, out under the brim, round its edge,
#: back over it and down the inside to the floor.
PROFILE = (
    (72.0, 57.0, 0.0, 0.0, FELT, 1),  # the edge of the flat top
    (81.0, 65.5, 3.0, 0.0, FELT, 1),
    (87.5, 71.5, 10.5, 0.0, FELT, 1),
    (93.0, 76.5, 30.0, 0.0, FELT, 1),
    (101.0, 83.0, 74.0, 0.0, BAND, 2),  # the band starts, a little proud of the felt
    (102.5, 84.5, 76.0, 0.0, BAND, 2),
    (105.5, 87.5, 106.0, 0.0, FELT, 3),  # the band ends where the brim leaves the crown
    (107.0, 89.0, 108.5, 0.0, FELT, 3),
    (126.0, 108.0, 114.5, 0.32, FELT, 3),
    (148.0, 130.0, 120.0, 0.69, FELT, 3),
    (164.0, 146.0, 121.5, 0.96, FELT, 3),
    (166.5, 148.5, 123.4, 1.0, FELT, 3),  # the brim's edge
    (164.0, 146.0, 125.3, 0.96, FELT, 3),
    (148.0, 130.0, 124.0, 0.69, FELT, 3),
    (126.0, 108.0, 118.5, 0.32, FELT, 3),
    (104.0, 86.0, 113.0, 0.0, FELT, 3),  # the lip of the opening
    (100.5, 82.5, 110.5, 0.0, BAND, 4),  # the sweatband
    (97.0, 79.0, 80.0, 0.0, LINING, 5),
    (89.5, 73.0, 30.0, 0.0, LINING, 5),
    (84.0, 68.0, 13.0, 0.0, LINING, 5),
    (71.0, 56.0, 4.5, 0.0, LINING, 5),  # the edge of the floor
)

# The brim is not a flat plate. On a head its two sides roll up and its front and back dip; the hat
# lies upside down, so here the sides (at +Y and -Y) roll down and the two ends lift a little.
BRIM_SIDE_DROP = 15.0  # how far the edge comes down at the sides
BRIM_SIDE_PULL = 0.05  # how much of its width the brim loses there as it rolls
BRIM_END_LIFT = 3.0  # how far the edge lifts at the two ends

#: Which ring is the lip of the opening, and which the floor's edge.
LIP_RING = 15
FLOOR_RING = len(PROFILE) - 1

# The bow on the band, at +X: a knot and a flat loop to each side of it.
BOW_Z = 91.0
BOW_RADIUS = 104.0  # the band's half size along X at that height
BOW_KNOT_HALF = (3.5, 7.0, 13.0)  # out from the band, along it, up
BOW_LOOP_HALF = (2.0, 10.0, 10.0)
BOW_LOOP_OFFSET = 16.0

# ---- colours ------------------------------------------------------------------------------

#: name, colour as seen (sRGB), metallic, roughness
MATERIALS = (
    ("felt", (0.10, 0.103, 0.112), 0.0, 0.95),
    ("band", (0.165, 0.165, 0.178), 0.0, 0.8),
    ("lining", (0.045, 0.045, 0.05), 0.0, 0.6),
)
TEXTURE_SIDE = 8

# ---- previews -----------------------------------------------------------------------------

RENDER_SIDE = 1024
RENDER_SAMPLES = 160
DRAFT_SIDE = 512
DRAFT_SAMPLES = 24
CAMERA_LENS = 85.0
BACKGROUND_GREY = 0.215  # linear; mid grey once shown
#: How much bigger than the harmonica's the preview set is: its lights' sizes and distances are the
#: harmonica's times this, and their power times its square, so the hat is lit as the harmonica is.
PREVIEW_SCALE = 1.9


class Builder:
    """Collects every part into one mesh, welding points part by part."""

    def __init__(self) -> None:
        self.bm = bmesh.new()
        self.group_layer = self.bm.faces.layers.int.new("group")
        self.points: dict = {}
        self.part_points: dict = {}
        self.flat_group = 10_000

    def vert(self, part: str, point: tuple):
        """
        Return the vertex of a part at a point, making it the first time

        :param part: Name of the part the point belongs to
        :param point: (x, y, z) in millimetres
        :return: The bmesh vertex
        """
        key = (part, round(point[0], 4), round(point[1], 4), round(point[2], 4))
        found = self.points.get(key)
        if found is None:
            found = self.bm.verts.new((point[0] * MM, point[1] * MM, point[2] * MM))
            self.points[key] = found
            self.part_points.setdefault(part, []).append(found)
        return found

    def face(self, part: str, points: list, material: int, outward: tuple, group: int = 0):
        """
        Add a face whose front looks the way it is told to

        The winding is worked out, not trusted: the game draws a face from its front only, and a
        face wound the wrong way is a hole in the prop there.

        :param part: Name of the part
        :param points: Corner points in millimetres, in either order round the face
        :param material: Material slot
        :param outward: A direction the face's front roughly looks along
        :param group: Smoothing group; 0 for a flat face of its own
        :return: The face, or None when the points do not span one
        """
        verts = []
        for point in points:
            vertex = self.vert(part, point)
            if vertex not in verts:
                verts.append(vertex)
        if len(verts) < 3:
            return None

        normal = Vector((0.0, 0.0, 0.0))
        for index, vertex in enumerate(verts):
            current, following = vertex.co, verts[(index + 1) % len(verts)].co
            normal.x += (current.y - following.y) * (current.z + following.z)
            normal.y += (current.z - following.z) * (current.x + following.x)
            normal.z += (current.x - following.x) * (current.y + following.y)
        if normal.length < 1e-14:
            return None
        if normal.dot(Vector(outward)) < 0.0:
            verts.reverse()

        face = self.bm.faces.new(verts)
        face.material_index = material
        face.smooth = True
        if group:
            face[self.group_layer] = group
        else:
            self.flat_group += 1
            face[self.group_layer] = self.flat_group
        return face

    def box(self, part: str, centre: tuple, half: tuple, material: int, skip: tuple = ()) -> None:
        """
        Add a box that stands square to the axes

        :param part: Name of the part
        :param centre: (x, y, z) of its middle, in millimetres
        :param half: Half its size along X, Y and Z
        :param material: Material slot
        :param skip: The sides left out because they are buried, as (axis, sign) pairs
        """
        for axis in range(3):
            first, second = [index for index in range(3) if index != axis]
            for sign in (-1.0, 1.0):
                if (axis, sign) in skip:
                    continue
                corners = []
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    corner = list(centre)
                    corner[axis] += sign * half[axis]
                    corner[first] += a * half[first]
                    corner[second] += b * half[second]
                    corners.append(tuple(corner))
                outward = [0.0, 0.0, 0.0]
                outward[axis] = sign
                self.face(part, corners, material, tuple(outward))

    def bounds(self, parts: list) -> tuple:
        """
        Measure the box round some parts

        :param parts: Names of the parts to measure
        :return: Minimum and maximum corners in metres
        """
        coords = [vertex.co for part in parts for vertex in self.part_points[part]]
        low = Vector([min(co[axis] for co in coords) for axis in range(3)])
        high = Vector([max(co[axis] for co in coords) for axis in range(3)])
        return low, high


def ring(index: int) -> list:
    """
    The points of one ring of the hat's outline

    :param index: Which ring of PROFILE
    :return: (x, y, z) of each point round it, in millimetres
    """
    half_x, half_y, height, along_brim = PROFILE[index][:4]
    roll = along_brim**1.5
    points = []
    for k in range(SIDES):
        angle = 2 * math.pi * k / SIDES
        to_side = math.sin(angle) ** 2
        points.append(
            (
                half_x * math.cos(angle),
                half_y * math.sin(angle) * (1.0 - BRIM_SIDE_PULL * roll * to_side),
                height + roll * (BRIM_END_LIFT * (1.0 - to_side) - BRIM_SIDE_DROP * to_side),
            )
        )
    return points


def build_hat(builder: Builder) -> None:
    """
    The whole hat as one skin: crown, band, brim, sweatband and the inside

    :param builder: Where the faces go
    """
    part = "hat"

    # The flat top it stands on, and the floor seen through the opening.
    builder.face(part, ring(0), FELT, (0, 0, -1))
    builder.face(part, ring(FLOOR_RING), LINING, (0, 0, 1))

    for index in range(len(PROFILE) - 1):
        first, second = ring(index), ring(index + 1)
        (x0, _, z0, _, material, group), (x1, _, z1) = PROFILE[index], PROFILE[index + 1][:3]
        # The skin's front is on the right of the way the rings run: out on the way up the crown,
        # down under the brim, up over it, in on the way down the inside.
        away, lift = z1 - z0, -(x1 - x0)
        for k in range(SIDES):
            following = (k + 1) % SIDES
            angle = 2 * math.pi * (k + 0.5) / SIDES
            builder.face(
                part,
                [first[k], first[following], second[following], second[k]],
                material,
                (away * math.cos(angle), away * math.sin(angle), lift),
                group,
            )


def build_bow(builder: Builder) -> None:
    """
    The bow on the band: a knot and a flat loop to each side of it

    :param builder: Where the faces go
    """
    part = "bow"
    # The knot's back is buried in the band.
    builder.box(part, (BOW_RADIUS + BOW_KNOT_HALF[0] - 1.0, 0.0, BOW_Z), BOW_KNOT_HALF, BAND, ((0, -1.0),))
    for side in (-1.0, 1.0):
        # The band curves away from the knot, so each loop sits further in than the knot does: its
        # near end is under the knot and its far end stands off the band, where its back shows.
        builder.box(part, (BOW_RADIUS + BOW_LOOP_HALF[0] - 4.5, side * BOW_LOOP_OFFSET, BOW_Z), BOW_LOOP_HALF, BAND)


def finish_mesh(builder: Builder) -> object:
    """
    Mark the hard edges, lay out the UVs and turn what was collected into a mesh

    :param builder: The collected faces
    :return: The mesh
    """
    bm = builder.bm
    bm.normal_update()

    for edge in bm.edges:
        faces = edge.link_faces
        edge.smooth = (
            len(faces) == 2
            and faces[0][builder.group_layer] == faces[1][builder.group_layer]
            and faces[0].normal.angle(faces[1].normal, 0.0) < SMOOTH_ANGLE
        )

    # The converter refuses a mesh with no UV map. The pictures are one flat colour each, so a
    # plain projection along each face's own axis is all that is asked of it.
    uv_layer = bm.loops.layers.uv.new("UVMap")
    span = 0.36
    for face in bm.faces:
        normal = face.normal
        axis = max(range(3), key=lambda index: abs(normal[index]))
        first, second = [index for index in range(3) if index != axis]
        for loop in face.loops:
            co = loop.vert.co
            loop[uv_layer].uv = ((co[first] + span / 2) / span, (co[second] + span / 2) / span)

    mesh = bpy.data.meshes.new(NAME)
    bm.to_mesh(mesh)
    mesh.update()
    return mesh


def flat_image(name: str, colour: tuple, is_colour: bool) -> object:
    """
    Write a small picture of one colour and load it

    :param name: Name of the picture, and of its file
    :param colour: (r, g, b) as stored in the file
    :param is_colour: False for a picture that holds measurements
    :return: The image, packed into the scene
    """
    os.makedirs(TEXTURE_DIR, exist_ok=True)
    image = bpy.data.images.new(name, TEXTURE_SIDE, TEXTURE_SIDE, alpha=False)
    image.colorspace_settings.name = "sRGB" if is_colour else "Non-Color"
    image.pixels.foreach_set([*colour, 1.0] * (TEXTURE_SIDE * TEXTURE_SIDE))
    image.file_format = "PNG"
    image.filepath_raw = os.path.join(TEXTURE_DIR, f"{name}.png")
    image.save()
    image.pack()
    return image


def build_material(name: str, colour: tuple, metallic: float, roughness: float) -> object:
    """
    A Principled material of one colour, worn as flat pictures so it survives the conversion

    :param name: Short name of the material
    :param colour: The colour as seen (sRGB)
    :param metallic: 0 to 1
    :param roughness: 0 to 1
    :return: The material
    """
    material = bpy.data.materials.new(f"{NAME}_{name}")
    material.use_backface_culling = True
    try:
        material.use_nodes = True
    except AttributeError:
        pass
    tree = material.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    base = tree.nodes.new("ShaderNodeTexImage")
    surface = tree.nodes.new("ShaderNodeTexImage")
    split = tree.nodes.new("ShaderNodeSeparateColor")

    base.image = flat_image(f"{NAME}_{name}_d", colour, True)
    # glTF's own layout: occlusion in red, roughness in green, metallic in blue.
    surface.image = flat_image(f"{NAME}_{name}_orm", (1.0, roughness, metallic), False)

    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness

    tree.links.new(base.outputs["Color"], shader.inputs["Base Color"])
    tree.links.new(surface.outputs["Color"], split.inputs["Color"])
    tree.links.new(split.outputs["Green"], shader.inputs["Roughness"])
    tree.links.new(split.outputs["Blue"], shader.inputs["Metallic"])
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    for index, node in enumerate((surface, base, split, shader, output)):
        node.location = (index * 280 - 560, 0 if index != 1 else 320)
    return material


def build_model() -> tuple:
    """
    Build the hat as one object at the world's origin

    :return: The object and the builder that holds its parts
    """
    builder = Builder()
    build_hat(builder)
    build_bow(builder)

    mesh = finish_mesh(builder)
    for name, colour, metallic, roughness in MATERIALS:
        mesh.materials.append(build_material(name, colour, metallic, roughness))

    hat = bpy.data.objects.new(NAME, mesh)
    bpy.context.scene.collection.objects.link(hat)
    return hat, builder


def measure_parts(hat: object, builder: Builder) -> dict:
    """
    Work out what Parts.json says, from the mesh itself

    :param hat: The built object
    :param builder: The builder that holds its parts
    :return: The document
    :raises ValueError: When the hat does not stand on the ground over the origin
    """
    low, high = builder.bounds(["hat"])
    if abs(low.z) > 1e-6 or abs(low.x + high.x) > 1e-6 or abs(low.y + high.y) > 1e-6:
        raise ValueError(f"The hat spans {tuple(low)} to {tuple(high)}: not standing on the ground over the origin")
    all_low, all_high = builder.bounds(["hat", "bow"])

    lip_x, lip_y, lip_z = PROFILE[LIP_RING][:3]
    floor_x, floor_y, floor_z = PROFILE[FLOOR_RING][:3]

    hat.data.calc_loop_triangles()

    def metres(vector) -> list:
        return [round(value, 5) for value in vector]

    return {
        "Name": NAME,
        "Units": "metres",
        "Axes": "X the longer way, Y the shorter way, Z up; origin on the ground under the middle; the opening is up",
        "Size": metres(high - low),
        "OpeningCentre": metres((0.0, 0.0, lip_z * MM)),
        "OpeningHalfSize": metres((lip_x * MM, lip_y * MM)),
        "FloorCentre": metres((0.0, 0.0, floor_z * MM)),
        "FloorHalfSize": metres((floor_x * MM, floor_y * MM)),
        "TriangleCount": len(hat.data.loop_triangles),
        "BoundsWithBow": {"Min": metres(all_low), "Max": metres(all_high)},
    }


def export_glb(hat: object) -> str:
    """
    Write the model alone as a GLB

    :param hat: The built object
    :return: Path of the file
    """
    path = os.path.join(HERE, f"{NAME}.glb")
    bpy.ops.object.select_all(action="DESELECT")
    hat.select_set(True)
    bpy.context.view_layer.objects.active = hat
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_cameras=False,
        export_lights=False,
        export_animations=False,
    )
    return path


def build_world(scene: object) -> None:
    """
    A plain grey seen by the camera, and a soft bright-above dark-below room seen by the model

    :param scene: The scene to light
    """
    world = bpy.data.worlds.new("PreviewWorld")
    scene.world = world
    try:
        world.use_nodes = True
    except AttributeError:
        pass
    tree = world.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputWorld")
    mix = tree.nodes.new("ShaderNodeMixShader")
    path = tree.nodes.new("ShaderNodeLightPath")
    room = tree.nodes.new("ShaderNodeBackground")
    backdrop = tree.nodes.new("ShaderNodeBackground")
    coords = tree.nodes.new("ShaderNodeTexCoord")
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    remap = tree.nodes.new("ShaderNodeMapRange")
    ramp = tree.nodes.new("ShaderNodeValToRGB")

    remap.inputs["From Min"].default_value = -1.0
    remap.inputs["From Max"].default_value = 1.0

    elements = ramp.color_ramp.elements
    # From straight down to straight up: a lit floor, a dark line at the horizon, a bright band of
    # wall above it and a dimmer ceiling. The same room the harmonica is pictured in.
    stops = ((0.0, 0.16), (0.40, 0.30), (0.49, 0.34), (0.52, 0.05), (0.60, 0.07), (0.66, 0.95), (0.80, 1.05), (0.87, 0.30), (1.0, 0.42))
    while len(elements) < len(stops):
        elements.new(0.5)
    for element, (position, value) in zip(elements, stops):
        element.position = position
        element.color = (value, value, value, 1.0)

    backdrop.inputs["Color"].default_value = (BACKGROUND_GREY, BACKGROUND_GREY, BACKGROUND_GREY, 1.0)

    tree.links.new(coords.outputs["Generated"], split.inputs["Vector"])
    tree.links.new(split.outputs["Z"], remap.inputs["Value"])
    tree.links.new(remap.outputs["Result"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], room.inputs["Color"])
    tree.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    tree.links.new(room.outputs["Background"], mix.inputs[1])
    tree.links.new(backdrop.outputs["Background"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])


def build_ground(scene: object) -> object:
    """
    The ground the prop lies on in the previews: it shows nothing but the prop's shadow

    :param scene: The scene to put it in
    :return: The ground object
    """
    side = 6.0 * PREVIEW_SCALE
    mesh = bpy.data.meshes.new("PreviewGround")
    mesh.from_pydata(
        [(-side, -side, 0.0), (side, -side, 0.0), (side, side, 0.0), (-side, side, 0.0)], [], [(0, 1, 2, 3)]
    )
    ground = bpy.data.objects.new("PreviewGround", mesh)
    scene.collection.objects.link(ground)
    ground.is_shadow_catcher = True
    return ground


def area_light(name: str, size: float, power: float) -> object:
    """
    Make an area light that the camera does not see

    :param name: Name of the light
    :param size: Side of the light in metres
    :param power: Watts
    :return: The light object
    """
    data = bpy.data.lights.new(name, "AREA")
    data.shape = "SQUARE"
    data.size = size
    data.energy = power
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.visible_camera = False
    return light


def aim(thing: object, position: Vector, target: Vector) -> None:
    """
    Put a camera or a light somewhere, looking at a point

    :param thing: The object to place
    :param position: Where it goes
    :param target: What it looks at
    """
    thing.location = position
    thing.rotation_euler = (target - position).to_track_quat("-Z", "Y").to_euler()


#: file suffix, where the camera looks, the direction it is seen from, how far away
VIEWS = (
    ("front34", (0.0, 0.0, 0.055), (0.50, 1.0, 0.62), 1.10),  # from the audience's side
    ("top", (0.0, 0.0, 0.06), (0.0, 0.0, 1.0), 1.05),
    ("side", (0.0, 0.0, 0.062), (1.0, 0.0, 0.10), 1.00),
    ("player", (0.0, 0.0, 0.05), (-0.20, -1.0, 1.30), 1.10),  # as the busker looks down at it
)


def render_previews(scene: object, draft: bool) -> list:
    """
    Render the four previews, and leave their cameras in the scene

    :param scene: The scene holding the model
    :param draft: True for small rough pictures
    :return: Paths of the pictures
    """
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    build_world(scene)
    build_ground(scene)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = DRAFT_SAMPLES if draft else RENDER_SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = DRAFT_SIDE if draft else RENDER_SIDE
    scene.render.resolution_y = DRAFT_SIDE if draft else RENDER_SIDE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    reach = PREVIEW_SCALE
    key = area_light("PreviewKey", 0.30 * reach, 16.0 * reach * reach)
    fill = area_light("PreviewFill", 0.60 * reach, 5.0 * reach * reach)

    written = []
    for suffix, target, direction, distance in VIEWS:
        target_point = Vector(target)
        camera_data = bpy.data.cameras.new(f"Preview_{suffix}")
        camera_data.lens = CAMERA_LENS
        camera_data.clip_start = 0.01
        camera_data.clip_end = 50.0
        camera = bpy.data.objects.new(f"Preview_{suffix}", camera_data)
        scene.collection.objects.link(camera)

        position = target_point + Vector(direction).normalized() * distance
        if suffix == "top":
            # Straight down, the busker's side at the bottom of the picture.
            camera.location = position
            camera.rotation_euler = (0.0, 0.0, 0.0)
        else:
            aim(camera, position, target_point)
        bpy.context.view_layer.update()

        # The lights follow the camera: the key from the picture's top left, a soft fill from its right.
        basis = camera.matrix_world.to_3x3()
        right, up, back = basis @ Vector((1, 0, 0)), basis @ Vector((0, 1, 0)), basis @ Vector((0, 0, 1))
        aim(key, target_point + (-right * 0.34 + up * 0.42 + back * 0.30) * reach, target_point)
        aim(fill, target_point + (right * 0.50 + up * 0.05 + back * 0.35) * reach, target_point)

        scene.camera = camera
        scene.render.filepath = os.path.join(PREVIEW_DIR, f"{NAME}_{suffix}.png")
        bpy.ops.render.render(write_still=True)
        written.append(scene.render.filepath)
    return written


def main() -> None:
    """Build the model, export it, write Parts.json, save the scene and render the previews."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    hat, builder = build_model()
    parts = measure_parts(hat, builder)

    glb = export_glb(hat)

    with open(os.path.join(HERE, "Parts.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(parts, handle, indent=2)
        handle.write("\n")

    skip_render = "--no-render" in sys.argv
    previews = [] if skip_render else render_previews(scene, "--draft" in sys.argv)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, f"{NAME}.blend"))

    print(f"[make_tiphat] {parts['TriangleCount']} triangles, size {parts['Size']}")
    print(f"[make_tiphat] wrote {glb}")
    for path in previews:
        print(f"[make_tiphat] wrote {path}")


if __name__ == "__main__":
    main()
