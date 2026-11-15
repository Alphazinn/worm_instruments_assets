"""Build and render the tip window's hero picture: a banded stack of banknotes.

Run headless:
    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --python make_banknotes.py
    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --python make_banknotes.py -- --size 320 --out <test.png>

Arguments (all after the lone `--`, all optional):
    --out      The .png to write. Default Banknotes.png next to this file.
    --size     Side of the picture in pixels. Default 1024.
    --engine   cycles or eevee. Default cycles: it alone gives the soft shadow under the stack.
    --samples  Render samples. Default 48 (the denoiser cleans the rest).
    --yaw      Where the camera stands around the stack, degrees. Default 28.
    --pitch    How far above the stack the camera stands, degrees. Default 40.
    --fill     Share of the frame the notes' longer side takes, 0 to 1. Default 0.82.
    --no-shadow  Leave out the soft shadow under the stack.

Writes, next to this file:
    Textures/banknote_face.png   the note's printed face, drawn here by code
    Banknotes.png                the render: square, RGBA, transparent but for the notes and their shadow

The note is an invented one. Nothing on it is taken from a real currency: no portrait, no words,
no seal, no serial number, no real note's layout. It is a pale green paper with a dark green woven
border, a plain oval medallion, line hatching, line rosettes and a "$" in the corners and the
middle, all worked out from formulas below.

Axes (the geometry is written in millimetres and scaled to metres): X along the notes' length,
Y their width, Z up. The stack stands over the origin, on three loose notes fanned under it.
"""

import argparse
import math
import os
import random
import sys
import time

import bmesh
import bpy
import numpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEXTURE_DIR = os.path.join(HERE, "Textures")
FACE_TEXTURE = os.path.join(TEXTURE_DIR, "banknote_face.png")

#: The geometry below is written in millimetres and scaled to metres as it is built.
MM = 0.001

# ---- the note, in millimetres -------------------------------------------------------------

NOTE_LENGTH = 160.0  # X
NOTE_WIDTH = 68.0  # Y
HALF_LENGTH = NOTE_LENGTH / 2.0
HALF_WIDTH = NOTE_WIDTH / 2.0

# ---- the printed face ---------------------------------------------------------------------

TEXTURE_WIDTH = 2048
TEXTURE_HEIGHT = 870  # the note's own proportions
TEXEL = NOTE_LENGTH / TEXTURE_WIDTH  # millimetres of note under one pixel

#: Colours as seen (sRGB).
PAPER_COLOUR = (0.86, 0.90, 0.77)
WASH_COLOUR = (0.55, 0.78, 0.53)  # the light green tint printed under the line work
INK_COLOUR = (0.05, 0.27, 0.15)

MARGIN = 3.0  # blank paper round the border
BORDER_INNER = 10.0  # where the border ends, in from the paper's edge
MEDALLION_HALF = (17.5, 21.5)  # the oval's half sizes along X and Y
MEDALLION_SIGN = 25.0  # height of the "$" in it
ROSETTE_CENTRE = 45.0  # the two line rosettes sit this far left and right of the middle
ROSETTE_RADIUS = 13.0
CORNER_CENTRE = (67.0, 21.0)  # the corner discs, mirrored into all four corners
CORNER_RADIUS = 9.5
CORNER_SIGN = 11.5  # height of the "$" in each

# ---- the stack ----------------------------------------------------------------------------

STACK_NOTES = 24
#: A real note is a tenth of a millimetre thick, and 24 of them would be a sliver. The sheets are
#: drawn several times thicker so the picture reads as a stack when it is shown small.
SHEET_THICKNESS = 0.52
SHEET_PITCH = 0.68  # from one sheet's underside to the next one's, where the band squeezes them
FLARE = 0.16  # how much looser the sheets lie at the two ends than under the band
FLARE_START = 21.0  # the sheets start to part this far from the middle
SHEET_SHIFT = (0.7, 0.4)  # how far a sheet may sit off the others, along X and Y
SHEET_TURN = 0.25  # and how far it may be turned, degrees
SHEET_COLUMNS = 32
SHEET_ROWS = 4

BAND_HALF_WIDTH = 15.0  # along X
BAND_THICKNESS = 0.3
BAND_CORNER_RADIUS = 0.9
BAND_CORNER_SEGMENTS = 5
BAND_SLACK = 0.55  # room between the notes' long edges and the band
BAND_COLOUR = (0.79, 0.62, 0.27)  # mustard, as seen (sRGB)
EDGE_COLOUR = (0.78, 0.85, 0.70)  # the cut edge of the paper, as seen (sRGB)

# ---- the loose notes under the stack ------------------------------------------------------

LOOSE_THICKNESS = 0.22
LOOSE_GAP = 0.04
#: turn in degrees, centre (X, Y), how hard the free end curls up. Lowest first, and the curl
#: grows upwards so no note cuts through the one under it.
LOOSE_NOTES = (
    (12.0, (6.0, -20.0), 0.0011),
    (-15.0, (20.0, -16.0), 0.0016),
    (-3.0, (-10.0, -10.0), 0.0021),
)
LOOSE_COLUMNS = 48
LOOSE_ROWS = 20
#: The notes lie flat under this box and curl up outside it.
FLAT_HALF = (HALF_LENGTH + 1.5, HALF_WIDTH + 2.0)

#: Where the stack's lowest sheet lies: on the loose notes and on the band that goes under it.
LOOSE_TOP = len(LOOSE_NOTES) * (LOOSE_THICKNESS + LOOSE_GAP)
STACK_BASE = LOOSE_TOP + 0.05 + BAND_THICKNESS + 0.02
STACK_TOP = STACK_BASE + (STACK_NOTES - 1) * SHEET_PITCH + SHEET_THICKNESS

#: Faces shade as one curved surface where they meet under this angle.
SMOOTH_ANGLE = math.radians(40.0)

# ---- the picture --------------------------------------------------------------------------

#: A long lens: nearly no perspective, as on a product shot.
CAMERA_LENS = 100.0
CAMERA_SENSOR = 36.0

#: Lights, as (direction seen from the stack in camera terms (right, up, towards the camera),
#: distance and side in bounding radii, strength in watts per square metre of distance).
KEY_LIGHT = ((-0.62, 0.68, 0.40), 6.0, 5.0, 15.0)
FILL_LIGHT = ((0.80, 0.10, 0.60), 6.0, 9.0, 5.0)
ROOM_GREY = 0.30  # the even light of the room, linear
#: The picture's alpha fades to nothing over this share of its side, along the rim.
RIM_FADE = 0.07


def read_arguments() -> argparse.Namespace:
    """
    Read what follows the lone `--` on Blender's command line

    :return: The arguments
    """
    own = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="make_banknotes.py", description="Render the banknote stack.")
    parser.add_argument("--out", default=os.path.join(HERE, "Banknotes.png"), help="the .png to write")
    parser.add_argument("--size", type=int, default=1024, help="side of the picture in pixels")
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles", help="the renderer")
    parser.add_argument("--samples", type=int, default=48, help="render samples")
    parser.add_argument("--yaw", type=float, default=28.0, help="camera angle around the stack, degrees")
    parser.add_argument("--pitch", type=float, default=40.0, help="camera angle above the stack, degrees")
    parser.add_argument("--fill", type=float, default=0.82, help="share of the frame the notes' longer side takes")
    parser.add_argument("--no-shadow", action="store_true", help="leave out the shadow under the stack")
    return parser.parse_args(own)


# ---- drawing the face ---------------------------------------------------------------------


def smoothstep(low: float, high: float, value: numpy.ndarray) -> numpy.ndarray:
    """
    Ease from 0 at one edge to 1 at the other

    :param low: Where the result is 0
    :param high: Where the result is 1
    :param value: The values to ease
    :return: The eased values
    """
    amount = numpy.clip((value - low) / (high - low), 0.0, 1.0)
    return amount * amount * (3.0 - 2.0 * amount)


def between(value: numpy.ndarray, low: float, high: float, soft: float) -> numpy.ndarray:
    """
    Cover the stretch between two values, with soft ends

    :param value: The values to test
    :param low: Where the cover starts
    :param high: Where it ends
    :param soft: Half the width of each soft end
    :return: 1 inside, 0 outside
    """
    return smoothstep(low - soft, low + soft, value) * (1.0 - smoothstep(high - soft, high + soft, value))


def ruled(phase: numpy.ndarray, share: numpy.ndarray, period: numpy.ndarray) -> numpy.ndarray:
    """
    Parallel engraved lines: one line for each whole step of the phase

    Where the lines come closer than the picture can hold, they give way to the even tone they
    would average to, so no moire shows.

    :param phase: Counts the lines; a line lies on every half
    :param share: How much of each period is ink, 0 to 1
    :param period: How wide one period is there, in millimetres
    :return: Ink cover, 0 to 1
    """
    period = numpy.maximum(period, 1e-6)
    wave = numpy.abs((phase % 1.0) - 0.5) * 2.0
    soft = 1.2 * TEXEL / period
    drawn = 1.0 - smoothstep(share - soft, share + soft, wave)
    sharp = smoothstep(3.5 * TEXEL, 7.0 * TEXEL, period)
    return sharp * drawn + (1.0 - sharp) * share


def arc_distance(x: numpy.ndarray, y: numpy.ndarray, centre: tuple, radius: float, start: float, sweep: float) -> numpy.ndarray:
    """
    Distance to an arc of a circle

    :param x: Where, along X
    :param y: Where, along Y
    :param centre: The circle's centre
    :param radius: Its radius
    :param start: The angle the arc starts at, radians
    :param sweep: How far it runs anticlockwise from there, radians
    :return: The distance to the arc's line
    """
    across, upward = x - centre[0], y - centre[1]
    angle = (numpy.arctan2(upward, across) - start) % (2.0 * math.pi)
    on_ring = numpy.abs(numpy.hypot(across, upward) - radius)
    ends = []
    for end in (start, start + sweep):
        ends.append(numpy.hypot(x - (centre[0] + radius * math.cos(end)), y - (centre[1] + radius * math.sin(end))))
    return numpy.where(angle <= sweep, on_ring, numpy.minimum(ends[0], ends[1]))


def dollar_distance(x: numpy.ndarray, y: numpy.ndarray, height: float) -> numpy.ndarray:
    """
    Distance to the outline of a "$" built from two arcs and a bar; negative inside it

    :param x: Where, measured from the sign's middle
    :param y: Where, measured from the sign's middle
    :param height: Height of the sign, from the bar's foot to its head
    :return: The distance to the sign's outline
    """
    radius = 0.185 * height
    upper = arc_distance(x, y, (0.0, radius), radius, math.radians(25.0), math.radians(245.0))
    lower = arc_distance(x, y, (0.0, -radius), radius, math.radians(-155.0), math.radians(245.0))
    curve = numpy.minimum(upper, lower) - 0.072 * height
    bar = numpy.maximum(numpy.abs(x) - 0.032 * height, numpy.abs(y) - 0.5 * height)
    return numpy.minimum(curve, bar)


def stamp_sign(ink: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, height: float) -> numpy.ndarray:
    """
    Print a solid "$" with a clear rim round it

    :param ink: The ink so far
    :param x: Where, measured from the sign's middle
    :param y: Where, measured from the sign's middle
    :param height: Height of the sign
    :return: The ink with the sign on it
    """
    distance = dollar_distance(x, y, height)
    rim = 0.05 * height
    ink = ink * smoothstep(rim, rim + 0.5, distance)
    return numpy.maximum(ink, 1.0 - smoothstep(-0.7 * TEXEL, 0.7 * TEXEL, distance))


def woven_rings(x: numpy.ndarray, y: numpy.ndarray, radius: float, petals: int, rings: float) -> numpy.ndarray:
    """
    A line rosette: two sets of wavy rings, one waving out where the other waves in

    :param x: Where, measured from the rosette's middle
    :param y: Where, measured from the rosette's middle
    :param radius: The rosette's radius
    :param petals: How many waves go round
    :param rings: How many rings fit from the middle to the rim
    :return: Ink cover, 0 to 1
    """
    reach = numpy.hypot(x, y) / radius
    wave = 0.11 * numpy.cos(petals * numpy.arctan2(y, x))
    period = radius / rings
    first = ruled(reach * rings * (1.0 + wave), 0.30, period)
    second = ruled(reach * rings * (1.0 - wave) + 0.5, 0.30, period)
    return numpy.maximum(first, second)


def draw_face() -> numpy.ndarray:
    """
    Draw the note's printed face

    :return: The picture as rows of (r, g, b) from the note's near edge to its far edge, sRGB, 0 to 1
    """
    x_axis = ((numpy.arange(TEXTURE_WIDTH) + 0.5) / TEXTURE_WIDTH - 0.5) * NOTE_LENGTH
    y_axis = ((numpy.arange(TEXTURE_HEIGHT) + 0.5) / TEXTURE_HEIGHT - 0.5) * NOTE_WIDTH
    x, y = numpy.meshgrid(x_axis.astype(numpy.float32), y_axis.astype(numpy.float32))
    soft = 0.75 * TEXEL
    inward = numpy.minimum(HALF_LENGTH - numpy.abs(x), HALF_WIDTH - numpy.abs(y))  # in from the paper's edge

    # The field inside the border: wavy hatching, heavier towards the two ends.
    field = smoothstep(BORDER_INNER - soft, BORDER_INNER + soft, inward)
    weight = 0.20 + 0.30 * smoothstep(16.0, 64.0, numpy.abs(x))
    hatching = ruled(y / 0.85 + 0.30 * numpy.sin(x * (2.0 * math.pi / 11.0)), weight, 0.85)
    ink = field * hatching
    wash = field * 0.72

    # The border: dark, with a lattice of pale wavy lines woven through it, ruled on both sides.
    border = between(inward, MARGIN, BORDER_INNER, soft)
    lattice = numpy.sin(x * (2.0 * math.pi / 3.2) + 1.3 * numpy.sin(y * (2.0 * math.pi / 6.4)))
    lattice = lattice * numpy.sin(y * (2.0 * math.pi / 3.2) + 1.3 * numpy.sin(x * (2.0 * math.pi / 6.4)))
    woven = 0.95 - 0.66 * (1.0 - smoothstep(0.10, 0.26, numpy.abs(lattice)))
    rules = numpy.maximum(between(inward, MARGIN, MARGIN + 0.7, soft), between(inward, BORDER_INNER - 0.7, BORDER_INNER, soft))
    ink = ink * (1.0 - border) + border * numpy.maximum(woven, rules)
    ink = numpy.maximum(ink, 0.85 * between(inward, 1.5, 1.9, soft))
    ink = numpy.maximum(ink, 0.90 * between(inward, BORDER_INNER + 1.0, BORDER_INNER + 1.4, soft))

    # A line rosette on each side of the middle.
    for side in (-1.0, 1.0):
        across = x - side * ROSETTE_CENTRE
        reach = numpy.hypot(across, y)
        disc = 1.0 - smoothstep(ROSETTE_RADIUS - soft, ROSETTE_RADIUS + soft, reach)
        ink = ink * (1.0 - disc) + disc * woven_rings(across, y, ROSETTE_RADIUS, 12, 8.0)
        ink = numpy.maximum(ink, between(reach, ROSETTE_RADIUS, ROSETTE_RADIUS + 0.5, soft))
        ink = numpy.maximum(ink, between(reach, 0.0, 1.6, soft))

    # The medallion: a plain oval, ringed, with rays and rings inside and the sign in its middle.
    half_x, half_y = MEDALLION_HALF
    oval = numpy.hypot(x / half_x, y / half_y)
    oval_soft = soft / half_x
    angle = numpy.arctan2(y / half_y, x / half_x)
    reach = oval * half_x
    ink = ink * smoothstep(1.09 - oval_soft, 1.09 + oval_soft, oval)
    inside = 1.0 - smoothstep(0.84 - oval_soft, 0.84 + oval_soft, oval)
    rays = ruled(angle * (72.0 / (2.0 * math.pi)), 0.16 + 0.44 * oval * oval, reach * (2.0 * math.pi / 72.0))
    rings = ruled(oval * 9.0 * (1.0 + 0.04 * numpy.cos(14.0 * angle)), 0.20, half_x / 9.0)
    ink = numpy.maximum(ink, inside * numpy.maximum(rays, rings))
    beads = ruled(angle * (44.0 / (2.0 * math.pi)), 0.5, reach * (2.0 * math.pi / 44.0))
    ring = between(oval, 0.90, 1.0, oval_soft) * (1.0 - 0.8 * between(oval, 0.93, 0.97, oval_soft) * beads)
    ink = numpy.maximum(ink, ring)
    ink = numpy.maximum(ink, between(oval, 0.84, 0.862, oval_soft))
    ink = numpy.maximum(ink, between(oval, 1.045, 1.075, oval_soft))
    ink = stamp_sign(ink, x, y, MEDALLION_SIGN)
    wash = numpy.where(oval < 1.0, 0.45, wash)

    # A disc on each corner of the border, with the sign in it.
    for side_x in (-1.0, 1.0):
        for side_y in (-1.0, 1.0):
            across = x - side_x * CORNER_CENTRE[0]
            upward = y - side_y * CORNER_CENTRE[1]
            reach = numpy.hypot(across, upward)
            disc = 1.0 - smoothstep(CORNER_RADIUS - soft, CORNER_RADIUS + soft, reach)
            around = numpy.arctan2(upward, across)
            rays = ruled(around * (40.0 / (2.0 * math.pi)), 0.20, reach * (2.0 * math.pi / 40.0))
            inner = 1.0 - smoothstep(CORNER_RADIUS - 2.3 - soft, CORNER_RADIUS - 2.3 + soft, reach)
            ink = ink * (1.0 - disc) + inner * rays
            ink = numpy.maximum(ink, between(reach, CORNER_RADIUS - 1.3, CORNER_RADIUS, soft))
            ink = numpy.maximum(ink, between(reach, CORNER_RADIUS - 2.3, CORNER_RADIUS - 2.0, soft))
            ink = stamp_sign(ink, across, upward, CORNER_SIGN)
            wash = numpy.where(reach < CORNER_RADIUS, 0.50, wash)

    # Paper is never even: soft blotches a few millimetres wide, and a fine grain.
    random_state = numpy.random.RandomState(11)
    coarse = random_state.rand(TEXTURE_HEIGHT // 24 + 2, TEXTURE_WIDTH // 24 + 2).astype(numpy.float32)
    blotches = numpy.kron(coarse, numpy.ones((24, 24), dtype=numpy.float32))[:TEXTURE_HEIGHT, :TEXTURE_WIDTH]
    for axis in (0, 1):
        for _ in range(3):
            blotches = (blotches + numpy.roll(blotches, 8, axis=axis) + numpy.roll(blotches, -8, axis=axis)) / 3.0
    grain = random_state.rand(TEXTURE_HEIGHT, TEXTURE_WIDTH).astype(numpy.float32)
    shade = 1.0 + 0.05 * (blotches - 0.5) + 0.025 * (grain - 0.5)

    ink = numpy.clip(ink, 0.0, 1.0)[:, :, None]
    wash = wash[:, :, None]
    paper = numpy.array(PAPER_COLOUR, dtype=numpy.float32) * (1.0 - wash) + numpy.array(WASH_COLOUR, dtype=numpy.float32) * wash
    colour = paper * (1.0 - ink) + numpy.array(INK_COLOUR, dtype=numpy.float32) * ink
    return numpy.clip(colour * shade[:, :, None], 0.0, 1.0)


def write_face_texture() -> object:
    """
    Draw the note's face, write it as a .png and load it

    :return: The image
    """
    os.makedirs(TEXTURE_DIR, exist_ok=True)
    colour = draw_face()
    pixels = numpy.ones((TEXTURE_HEIGHT, TEXTURE_WIDTH, 4), dtype=numpy.float32)
    pixels[:, :, :3] = colour
    image = bpy.data.images.new("banknote_face", TEXTURE_WIDTH, TEXTURE_HEIGHT, alpha=False)
    image.colorspace_settings.name = "sRGB"
    image.pixels.foreach_set(pixels.ravel())
    image.file_format = "PNG"
    image.filepath_raw = FACE_TEXTURE
    image.save()
    return image


# ---- materials ----------------------------------------------------------------------------


def linear(colour: tuple) -> tuple:
    """
    Turn a colour as seen (sRGB) into what a shader's colour socket takes

    :param colour: (r, g, b), sRGB
    :return: (r, g, b, 1), linear
    """
    return tuple(((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in colour) + (1.0,)


def new_material(name: str) -> tuple:
    """
    Make a material with a Principled shader and nothing else

    :param name: Name of the material
    :return: The material, its node tree and the shader
    """
    material = bpy.data.materials.new(name)
    try:
        material.use_nodes = True
    except AttributeError:
        pass  # from Blender 5 a material always has its nodes
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    shader.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in shader.inputs:
        shader.inputs["Specular IOR Level"].default_value = 0.25
    return material, tree, shader


def face_material(image: object) -> object:
    """
    The printed paper: the drawn face on both sides of a note

    :param image: The face picture
    :return: The material
    """
    material, tree, shader = new_material("banknote_face")
    picture = tree.nodes.new("ShaderNodeTexImage")
    picture.image = image
    picture.interpolation = "Cubic"
    picture.extension = "EXTEND"
    shader.inputs["Roughness"].default_value = 0.78
    tree.links.new(picture.outputs["Color"], shader.inputs["Base Color"])
    return material


def edge_material() -> object:
    """
    The cut edge of a sheet: pale paper, each sheet a touch lighter or darker than the next

    :return: The material
    """
    material, tree, shader = new_material("banknote_edge")
    which = tree.nodes.new("ShaderNodeObjectInfo")
    spread = tree.nodes.new("ShaderNodeMapRange")
    tint = tree.nodes.new("ShaderNodeMixRGB")
    spread.inputs["To Min"].default_value = 0.62
    spread.inputs["To Max"].default_value = 1.0
    tint.blend_type = "MULTIPLY"
    tint.inputs["Fac"].default_value = 1.0
    tint.inputs["Color1"].default_value = linear(EDGE_COLOUR)
    shader.inputs["Roughness"].default_value = 0.9
    tree.links.new(which.outputs["Random"], spread.inputs["Value"])
    tree.links.new(spread.outputs["Result"], tint.inputs["Color2"])
    tree.links.new(tint.outputs["Color"], shader.inputs["Base Color"])
    return material


def band_material() -> object:
    """
    The plain paper band: mustard, a little uneven as paper is

    :return: The material
    """
    material, tree, shader = new_material("banknote_band")
    coords = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    spread = tree.nodes.new("ShaderNodeMapRange")
    tint = tree.nodes.new("ShaderNodeMixRGB")
    noise.inputs["Scale"].default_value = 260.0
    noise.inputs["Detail"].default_value = 3.0
    spread.inputs["To Min"].default_value = 0.88
    spread.inputs["To Max"].default_value = 1.08
    tint.blend_type = "MULTIPLY"
    tint.inputs["Fac"].default_value = 1.0
    tint.inputs["Color1"].default_value = linear(BAND_COLOUR)
    shader.inputs["Roughness"].default_value = 0.7
    tree.links.new(coords.outputs["Object"], noise.inputs["Vector"])
    tree.links.new(noise.outputs["Fac"], spread.inputs["Value"])
    tree.links.new(spread.outputs["Result"], tint.inputs["Color2"])
    tree.links.new(tint.outputs["Color"], shader.inputs["Base Color"])
    return material


# ---- geometry -----------------------------------------------------------------------------


def finish_object(name: str, bm: object, materials: list) -> object:
    """
    Mark the hard edges, turn the collected faces into a mesh and put it in the scene

    :param name: Name of the object
    :param bm: The collected faces
    :param materials: The materials, in slot order
    :return: The object
    """
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    for face in bm.faces:
        face.smooth = True
    for edge in bm.edges:
        faces = edge.link_faces
        edge.smooth = len(faces) == 2 and faces[0].normal.angle(faces[1].normal, 0.0) < SMOOTH_ANGLE

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for material in materials:
        mesh.materials.append(material)
    thing = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(thing)
    return thing


def build_note(name: str, turn: float, centre: tuple, under, thickness: float, columns: int, rows: int, materials: list) -> object:
    """
    One sheet: a thin slab that follows whatever height it is told to lie at

    :param name: Name of the object
    :param turn: How far the note is turned about Z, degrees
    :param centre: Where its middle lies, (X, Y) in millimetres
    :param under: under(x, y, along, across) gives the height of the sheet's underside at a place:
        x and y in the world, along and across measured on the note itself, all millimetres
    :param thickness: The sheet's thickness in millimetres
    :param columns: Faces along the length
    :param rows: Faces along the width
    :param materials: The face and edge materials
    :return: The object
    """
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    cos_turn, sin_turn = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    printed = {}
    top, bottom = [], []
    for row in range(rows + 1):
        across = (row / rows - 0.5) * NOTE_WIDTH
        top.append([])
        bottom.append([])
        for column in range(columns + 1):
            along = (column / columns - 0.5) * NOTE_LENGTH
            x = centre[0] + along * cos_turn - across * sin_turn
            y = centre[1] + along * sin_turn + across * cos_turn
            height = under(x, y, along, across)
            for sheet, z in ((bottom, height), (top, height + thickness)):
                vertex = bm.verts.new((x * MM, y * MM, z * MM))
                printed[vertex] = (along / NOTE_LENGTH + 0.5, across / NOTE_WIDTH + 0.5)
                sheet[-1].append(vertex)

    def add(corners: list, slot: int) -> None:
        face = bm.faces.new(corners)
        face.material_index = slot
        for loop in face.loops:
            loop[uv_layer].uv = printed[loop.vert]

    for row in range(rows):
        for column in range(columns):
            add([top[row][column], top[row][column + 1], top[row + 1][column + 1], top[row + 1][column]], 0)
            add([bottom[row][column], bottom[row + 1][column], bottom[row + 1][column + 1], bottom[row][column + 1]], 0)
    for column in range(columns):
        for row in (0, rows):
            add([bottom[row][column], bottom[row][column + 1], top[row][column + 1], top[row][column]], 1)
    for row in range(rows):
        for column in (0, columns):
            add([bottom[row][column], bottom[row + 1][column], top[row + 1][column], top[row][column]], 1)

    return finish_object(name, bm, materials)


def build_stack(materials: list) -> list:
    """
    The banded stack: every sheet a little off the others, and looser at the ends than under the band

    :param materials: The face and edge materials
    :return: The sheets, lowest first
    """
    chance = random.Random(7)
    sheets = []
    for index in range(STACK_NOTES):
        shift = (chance.uniform(-SHEET_SHIFT[0], SHEET_SHIFT[0]), chance.uniform(-SHEET_SHIFT[1], SHEET_SHIFT[1]))
        turn = chance.uniform(-SHEET_TURN, SHEET_TURN)
        lift = chance.uniform(-0.05, 0.05)
        tilt = chance.uniform(-0.03, 0.03)

        def under(x: float, y: float, along: float, across: float, index=index, lift=lift, tilt=tilt) -> float:
            parted = min(max((abs(along) - FLARE_START) / (HALF_LENGTH - FLARE_START), 0.0), 1.0) ** 2
            wander = parted * (lift + tilt * across / HALF_WIDTH) if index else 0.0
            return STACK_BASE + index * SHEET_PITCH * (1.0 + FLARE * parted) + wander

        sheets.append(build_note(f"Sheet{index:02d}", turn, shift, under, SHEET_THICKNESS, SHEET_COLUMNS, SHEET_ROWS, materials))
    return sheets


def build_loose_notes(materials: list) -> list:
    """
    Three notes fanned under the stack, flat where it stands on them and curling up where they are free

    :param materials: The face and edge materials
    :return: The notes, lowest first
    """
    notes = []
    for index, (turn, centre, curl) in enumerate(LOOSE_NOTES):
        base = index * (LOOSE_THICKNESS + LOOSE_GAP)

        def under(x: float, y: float, along: float, across: float, base=base, curl=curl) -> float:
            free = math.hypot(max(abs(x) - FLAT_HALF[0], 0.0), max(abs(y) - FLAT_HALF[1], 0.0))
            return base + curl * free * free * (1.0 + 0.3 * math.sin(x * 0.02 + 1.0))

        notes.append(build_note(f"Loose{index}", turn, centre, under, LOOSE_THICKNESS, LOOSE_COLUMNS, LOOSE_ROWS, materials))
    return notes


def build_band(material: object) -> object:
    """
    The paper band: one loop round the stack's middle, with rounded corners

    :param material: The band's material
    :return: The object
    """
    half_y = HALF_WIDTH + BAND_SLACK
    low, high = STACK_BASE - 0.02, STACK_TOP + 0.03
    radius = BAND_CORNER_RADIUS

    def loop(grow: float) -> list:
        points = []
        corners = (
            (half_y - radius, high - radius, 0.0),
            (-half_y + radius, high - radius, 90.0),
            (-half_y + radius, low + radius, 180.0),
            (half_y - radius, low + radius, 270.0),
        )
        for centre_y, centre_z, start in corners:
            for step in range(BAND_CORNER_SEGMENTS + 1):
                angle = math.radians(start + 90.0 * step / BAND_CORNER_SEGMENTS)
                points.append((centre_y + (radius + grow) * math.cos(angle), centre_z + (radius + grow) * math.sin(angle)))
        return points

    bm = bmesh.new()
    rings = {}
    for skin, grow in (("inner", 0.0), ("outer", BAND_THICKNESS)):
        for side in (-1.0, 1.0):
            rings[skin, side] = [bm.verts.new((side * BAND_HALF_WIDTH * MM, y * MM, z * MM)) for y, z in loop(grow)]

    count = len(rings["inner", 1.0])
    for index in range(count):
        following = (index + 1) % count
        for first, second in (
            (rings["outer", -1.0], rings["outer", 1.0]),
            (rings["inner", 1.0], rings["inner", -1.0]),
            (rings["inner", -1.0], rings["outer", -1.0]),
            (rings["outer", 1.0], rings["inner", 1.0]),
        ):
            bm.faces.new([first[index], second[index], second[following], first[following]])
    return finish_object("Band", bm, [material])


# ---- the picture --------------------------------------------------------------------------


def scene_points(things: list) -> numpy.ndarray:
    """
    Every point of the given objects, where it stands in the world

    :param things: The objects to measure
    :return: An N by 3 array of positions
    """
    clouds = []
    for thing in things:
        flat = numpy.empty(len(thing.data.vertices) * 3, dtype=numpy.float64)
        thing.data.vertices.foreach_get("co", flat)
        world = numpy.array(thing.matrix_world)
        clouds.append(flat.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3])
    return numpy.concatenate(clouds)


def camera_axes(yaw: float, pitch: float) -> tuple:
    """
    The camera's own axes for a view

    :param yaw: Degrees around the stack, 0 at its -Y side
    :param pitch: Degrees above it
    :return: The camera's right, up and backward directions, as unit Vectors
    """
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    back = Vector((math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)))
    right = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    return right, back.cross(right), back


def frame_camera(points: numpy.ndarray, axes: tuple, fill: float) -> tuple:
    """
    Find where the camera stands so the notes sit in the middle of the picture at the wanted size

    :param points: The notes' points in the world
    :param axes: The camera's right, up and backward directions
    :param fill: Share of the frame the notes' longer side takes
    :return: The camera's position, and its distance to the notes' middle
    """
    right, up, back = (numpy.array(axis) for axis in axes)
    tangent = (CAMERA_SENSOR / 2.0) / CAMERA_LENS  # half the frame, at one metre from the camera
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = float(numpy.linalg.norm(points - centre, axis=1).max())
    position = centre + back * (radius / tangent / fill)

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


def add_camera(position: Vector, axes: tuple, distance: float, radius: float) -> None:
    """
    Make the camera and make it the scene's

    :param position: Where it stands
    :param axes: Its right, up and backward directions
    :param distance: How far the notes' middle is
    :param radius: The notes' bounding radius
    """
    data = bpy.data.cameras.new("BanknotesCamera")
    data.lens = CAMERA_LENS
    data.sensor_width = CAMERA_SENSOR
    data.sensor_fit = "HORIZONTAL"
    data.clip_start = max(distance - radius * 3.0, distance * 0.01)
    data.clip_end = distance + radius * 6.0
    camera = bpy.data.objects.new("BanknotesCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    place(camera, position, axes)
    bpy.context.scene.camera = camera


def add_light(name: str, rig: tuple, centre: Vector, radius: float, axes: tuple, lit: object) -> None:
    """
    Make one soft area light that the camera does not see, aimed at the notes and lighting them alone

    :param name: Name of the light
    :param rig: One of the light constants above
    :param centre: The notes' middle
    :param radius: The notes' bounding radius
    :param axes: The camera's right, up and backward directions
    :param lit: The collection of the objects this light falls on
    """
    (across, upward, towards), reach, side, strength = rig
    right, up, back = axes
    direction = (right * across + up * upward + back * towards).normalized()
    distance = radius * reach

    data = bpy.data.lights.new(name, "AREA")
    data.shape = "DISK"
    data.size = radius * side
    # The power grows with the square of the distance, so the light is the same at any scale.
    data.energy = strength * distance * distance
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.visible_camera = False
    light.light_linking.receiver_collection = lit

    aside = direction.cross(Vector((0.0, 0.0, 1.0)))
    if aside.length < 1e-6:
        aside = Vector((1.0, 0.0, 0.0))
    aside.normalize()
    place(light, centre + direction * distance, (aside, direction.cross(aside), direction))


def build_room() -> None:
    """The world: an even soft grey, so no side of the paper falls to black. The film is transparent."""
    world = bpy.data.worlds.new("BanknotesRoom")
    bpy.context.scene.world = world
    if bpy.app.version < (5, 0, 0):
        world.use_nodes = True  # from Blender 5 a world always has its nodes
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputWorld")
    room = tree.nodes.new("ShaderNodeBackground")
    room.inputs["Color"].default_value = (ROOM_GREY, ROOM_GREY, ROOM_GREY, 1.0)
    room.inputs["Strength"].default_value = 1.0
    tree.links.new(room.outputs["Background"], output.inputs["Surface"])


def add_shadow_floor(radius: float) -> None:
    """
    A floor that shows nothing but the shadow that falls on it (Cycles only)

    The two lamps light the notes alone, so the floor is lit by the even room and nothing else:
    what it catches is the soft dark close under the notes, not a long shadow thrown to one side.

    :param radius: The notes' bounding radius
    """
    side = radius * 12.0
    mesh = bpy.data.meshes.new("ShadowFloor")
    mesh.from_pydata([(-side, -side, 0.0), (side, -side, 0.0), (side, side, 0.0), (-side, side, 0.0)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("ShadowFloor", mesh)
    floor.location = (0.0, 0.0, -0.00002)
    bpy.context.scene.collection.objects.link(floor)
    floor.is_shadow_catcher = True


def set_up_render(engine: str, size: int, samples: int, out: str) -> None:
    """
    A square transparent RGBA picture, colours shown as they are

    :param engine: cycles or eevee
    :param size: Side of the picture in pixels
    :param samples: Render samples
    :param out: The .png to write
    """
    scene = bpy.context.scene
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.use_denoising = True
        # Paper only: a few bounces are all the light that counts, and each one costs time.
        scene.cycles.max_bounces = 4
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 2
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
    else:
        names = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        scene.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in names else "BLENDER_EEVEE_NEXT"
        for name, value in (("taa_render_samples", samples), ("use_shadows", True), ("use_raytracing", True), ("shadow_ray_count", 2), ("shadow_step_count", 8)):
            if hasattr(scene.eevee, name):
                setattr(scene.eevee, name, value)
        for light in bpy.data.lights:
            for name, value in (("use_shadow", True), ("use_shadow_jitter", True), ("shadow_jitter_overblur", 5.0)):
                if hasattr(light, name):
                    setattr(light, name, value)
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 90
    # Standard, not a film look: the paper keeps its own colours and clean whites.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.filepath = out


def clear_rim(out: str) -> None:
    """
    Fade the picture's alpha to nothing over a thin strip along its rim, and write it back

    The shadow under the notes thins out with distance but never quite ends, and the faint rest of
    it would stop dead at the picture's edge. The notes themselves stand well inside the strip.

    :param out: The .png that was written
    """
    image = bpy.data.images.load(out)
    width, height = image.size
    pixels = numpy.empty(width * height * 4, dtype=numpy.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(height, width, 4)
    across = (numpy.arange(width) + 0.5) / width
    upward = (numpy.arange(height) + 0.5) / height
    across = smoothstep(0.0, RIM_FADE, numpy.minimum(across, 1.0 - across))
    upward = smoothstep(0.0, RIM_FADE, numpy.minimum(upward, 1.0 - upward))
    pixels[:, :, 3] *= upward[:, None] * across[None, :]
    image.pixels.foreach_set(pixels.ravel())
    image.save()
    bpy.data.images.remove(image)


def report(out: str) -> None:
    """
    Read the picture back and print how the notes sit in it

    :param out: The .png that was written
    """
    image = bpy.data.images.load(out)
    width, height = image.size
    pixels = numpy.empty(width * height * 4, dtype=numpy.float32)
    image.pixels.foreach_get(pixels)
    alpha = pixels.reshape(height, width, 4)[:, :, 3]
    print(f"[make_banknotes] wrote {out}: {width}x{height}, {image.channels} channels, depth {image.depth}")
    for name, level in (("notes", 0.9), ("notes and shadow", 0.02)):
        columns = numpy.flatnonzero(alpha.max(axis=0) > level)
        rows = numpy.flatnonzero(alpha.max(axis=1) > level)
        if len(columns) == 0:
            print(f"[make_banknotes] WARNING: no {name} in the picture")
            continue
        left, right = int(columns[0]), int(width - 1 - columns[-1])
        bottom, top = int(rows[0]), int(height - 1 - rows[-1])
        print(f"[make_banknotes] {name}: {width - left - right}x{height - top - bottom} px, margins left {left} right {right} top {top} bottom {bottom}")
    rim = max(float(alpha[0].max()), float(alpha[-1].max()), float(alpha[:, 0].max()), float(alpha[:, -1].max()))
    print(f"[make_banknotes] strongest alpha on the picture's rim: {rim:.3f}")


def main() -> None:
    """Draw the note's face, build the stack, light it and render the picture."""
    arguments = read_arguments()
    out = os.path.abspath(arguments.out)
    if not 0.05 <= arguments.fill <= 1.0:
        raise SystemExit("[make_banknotes] --fill must be between 0.05 and 1")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    started = time.time()
    paper = [face_material(write_face_texture()), edge_material()]
    print(f"[make_banknotes] drew {FACE_TEXTURE} in {time.time() - started:.1f} s")

    notes = build_loose_notes(paper) + build_stack(paper)
    notes.append(build_band(band_material()))
    bpy.context.view_layer.update()

    points = scene_points(notes)
    centre = Vector((points.min(axis=0) + points.max(axis=0)) / 2.0)
    radius = float(numpy.linalg.norm(points - numpy.array(centre), axis=1).max())

    axes = camera_axes(arguments.yaw, arguments.pitch)
    position, distance = frame_camera(points, axes, arguments.fill)
    add_camera(position, axes, distance, radius)
    lit = bpy.data.collections.new("BanknotesLit")
    for note in notes:
        lit.objects.link(note)
    add_light("BanknotesKey", KEY_LIGHT, centre, radius, axes, lit)
    add_light("BanknotesFill", FILL_LIGHT, centre, radius, axes, lit)
    build_room()
    if arguments.engine == "cycles" and not arguments.no_shadow:
        add_shadow_floor(radius)

    set_up_render(arguments.engine, arguments.size, arguments.samples, out)
    started = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[make_banknotes] engine {arguments.engine}, {arguments.samples} samples, {arguments.size} px: rendered in {time.time() - started:.1f} s")
    print(f"[make_banknotes] yaw {arguments.yaw:g} pitch {arguments.pitch:g} fill {arguments.fill:g}")
    if arguments.fill > 1.0 - 2.0 * RIM_FADE:
        print("[make_banknotes] WARNING: the notes reach into the strip where the rim fades out")
    clear_rim(out)
    report(out)


if __name__ == "__main__":
    main()
