"""Build the worm_instr_harmonica prop from nothing.

Run headless:
    blender -b --python make_harmonica.py

Writes, next to this file:
    worm_instr_harmonica.glb     the model the prop converter takes
    worm_instr_harmonica.blend   the same model with the preview cameras and lights
    Textures/*.png               the flat colour pictures the materials wear
    Parts.json                   sizes and the points the script attaches things to
    Preview/*.png                four renders

Add `-- --no-render` to skip the previews (the saved scene then has no cameras or lights).
ConvertAnswers.txt beside this file is not written here: it holds the answers to the prop
converter's nine questions (name, height, keep the origin, no collision, ...), one a line.

A 16-hole chromatic harmonica, stylised, with no brand, logo or text on it.

Axes (metres): X along the length (slide button at +X), Y the depth with the MOUTH side at -Y,
Z up. The origin is the centre of the body, the slide button not counted.

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

NAME = "worm_instr_harmonica"
HERE = os.path.dirname(os.path.abspath(__file__))
PREVIEW_DIR = os.path.join(HERE, "Preview")
TEXTURE_DIR = os.path.join(HERE, "Textures")

#: The geometry below is written in millimetres and scaled to metres as it is built.
MM = 0.001

CHROME = 0
PLASTIC = 1

#: Faces of one smoothing group shade as one curved surface where they meet under this angle.
SMOOTH_ANGLE = math.radians(50.0)

# ---- the body, in millimetres -------------------------------------------------------------

BODY_HALF_LENGTH = 95.0  # X
BODY_FRONT = -25.0  # Y of the mouth side
BODY_BACK = 25.0  # Y of the back
BODY_HALF_HEIGHT = 14.0  # Z

# The comb: the black slab in the middle.
COMB_FRONT = -16.0
COMB_HALF_HEIGHT = 6.0
COMB_CORNER_RADIUS = 7.0
COMB_CORNER_SEGMENTS = 4
COMB_CHAMFER = 0.8

# The mouthpiece: the chrome bar on the mouth side.
MOUTH_HALF_HEIGHT = 9.0
MOUTH_FLAT_HALF_HEIGHT = 5.5
MOUTH_ROUND_RADIUS = 3.5
MOUTH_ROUND_SEGMENTS = 3
MOUTH_END_CHAMFER = 1.5

HOLE_COUNT = 16
HOLE_PITCH = 10.8
HOLE_HALF_WIDTH = 3.4
HOLE_HALF_HEIGHT = 3.2
HOLE_DEPTH = 4.0

# The cover plates: (Y, Z) of the top one's outer skin from its front edge to its back edge. The
# front edge starts inside the mouthpiece, so no gap opens where the plate comes out from behind it.
COVER_PROFILE = (
    (-17.0, 5.7),
    (-14.5, 10.2),
    (-12.0, 12.2),
    (-8.5, 13.5),
    (-4.0, 14.0),
    (2.0, 14.0),
    (8.0, 13.7),
    (14.0, 12.9),
    (19.0, 11.6),
    (22.0, 10.4),
)
COVER_STRAIGHT_HALF = 79.0  # the cover is a straight extrusion out to here
COVER_END_LENGTH = 9.0  # and rounds down onto the comb over this much
COVER_END_STEPS = (0.0, 3.0, 5.5, 7.5, 8.6, 9.0)
COVER_BACK_PULL = 3.0  # how far the back corners are drawn in as the end rounds down
COVER_LIP = 1.0  # thickness of the plate seen at the open back
COVER_SLOT_DEPTH = 3.0  # how far in the dark of the open back sits
#: The plate's skirt ends inside the comb, under its chamfer, so nowhere does the plate's inside show.
COVER_FOOT = COMB_HALF_HEIGHT - COMB_CHAMFER - 0.3

# The slide button, out of the right end where the mouthpiece meets the comb.
BUTTON_Y = -16.0
BUTTON_Z = 0.0
BUTTON_SHAFT_RADIUS = 2.2
BUTTON_SHAFT_SIDES = 8
BUTTON_RADIUS = 5.0
BUTTON_SIDES = 12
BUTTON_KNOB_START = 100.5
BUTTON_KNOB_FLAT_END = 104.5
BUTTON_TIP = 107.0

# ---- colours ------------------------------------------------------------------------------

#: name, colour as seen (sRGB), metallic, roughness
MATERIALS = (
    ("chrome", (0.86, 0.87, 0.89), 1.0, 0.2),
    ("plastic", (0.085, 0.085, 0.09), 0.0, 0.5),
)
TEXTURE_SIDE = 8

# ---- previews -----------------------------------------------------------------------------

RENDER_SIDE = 1024
RENDER_SAMPLES = 160
CAMERA_LENS = 85.0
BACKGROUND_GREY = 0.215  # linear; mid grey once shown


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


def inset_polygon(points: list, distance: float) -> list:
    """
    Move every side of a flat polygon inwards by the same distance

    :param points: The polygon's corners, (a, b) pairs, in either order
    :param distance: How far to move each side
    :return: The corners of the smaller polygon, one for each given
    """
    count = len(points)
    area = sum(
        points[i][0] * points[(i + 1) % count][1] - points[(i + 1) % count][0] * points[i][1] for i in range(count)
    )
    turn = 1.0 if area > 0 else -1.0
    result = []
    for index in range(count):
        before, here, after = points[index - 1], points[index], points[(index + 1) % count]
        first = Vector((here[0] - before[0], here[1] - before[1])).normalized()
        second = Vector((after[0] - here[0], after[1] - here[1])).normalized()
        normal_first = Vector((-first.y, first.x)) * turn
        normal_second = Vector((-second.y, second.x)) * turn
        mitre = normal_first + normal_second
        mitre = normal_first if mitre.length < 1e-9 else mitre.normalized()
        reach = distance / max(mitre.dot(normal_first), 0.3)
        result.append((here[0] + mitre.x * reach, here[1] + mitre.y * reach))
    return result


def build_comb(builder: Builder) -> None:
    """
    The black slab between the cover plates: it shows at the back and at both ends

    :param builder: Where the faces go
    """
    part = "comb"
    half = BODY_HALF_LENGTH
    radius = COMB_CORNER_RADIUS

    outline = [(-half, COMB_FRONT), (half, COMB_FRONT)]
    for corner_x, start in ((half - radius, 0.0), (-half + radius, math.pi / 2)):
        for step in range(COMB_CORNER_SEGMENTS + 1):
            angle = start + (math.pi / 2) * step / COMB_CORNER_SEGMENTS
            outline.append((corner_x + radius * math.cos(angle), BODY_BACK - radius + radius * math.sin(angle)))
    inner = inset_polygon(outline, COMB_CHAMFER)

    top = COMB_HALF_HEIGHT
    side = COMB_HALF_HEIGHT - COMB_CHAMFER
    middle = Vector((0.0, (COMB_FRONT + BODY_BACK) / 2))

    builder.face(part, [(x, y, top) for x, y in inner], PLASTIC, (0, 0, 1))
    builder.face(part, [(x, y, -top) for x, y in inner], PLASTIC, (0, 0, -1))

    count = len(outline)
    for index in range(count):
        following = (index + 1) % count
        (x0, y0), (x1, y1) = outline[index], outline[following]
        (ix0, iy0), (ix1, iy1) = inner[index], inner[following]
        away = Vector(((x0 + x1) / 2, (y0 + y1) / 2)) - middle
        # The straight front and its corners stay crisp; the rounded back corners shade as curves.
        builder.face(part, [(x0, y0, -side), (x1, y1, -side), (x1, y1, side), (x0, y0, side)], PLASTIC, (*away, 0), 20)
        for sign, group in ((1.0, 21), (-1.0, 22)):
            builder.face(
                part,
                [(x0, y0, sign * side), (x1, y1, sign * side), (ix1, iy1, sign * top), (ix0, iy0, sign * top)],
                PLASTIC,
                (away.x, away.y, sign * away.length),
                group,
            )


def mouth_profile() -> list:
    """
    The mouthpiece's outline seen from an end

    :return: (Y, Z) corners from the back's top, round the front, to the back's bottom
    """
    arc_y = BODY_FRONT + MOUTH_ROUND_RADIUS
    upper = [(COMB_FRONT, MOUTH_HALF_HEIGHT)]
    for step in range(MOUTH_ROUND_SEGMENTS + 1):
        angle = (math.pi / 2) * step / MOUTH_ROUND_SEGMENTS
        upper.append(
            (arc_y - MOUTH_ROUND_RADIUS * math.sin(angle), MOUTH_FLAT_HALF_HEIGHT + MOUTH_ROUND_RADIUS * math.cos(angle))
        )
    return upper + [(y, -z) for y, z in reversed(upper)]


def hole_centres() -> list:
    """
    Where the sixteen holes sit along the mouthpiece

    :return: The X of each hole's centre, left to right, in millimetres
    """
    return [(index - (HOLE_COUNT - 1) / 2) * HOLE_PITCH for index in range(HOLE_COUNT)]


def build_mouthpiece(builder: Builder) -> None:
    """
    The chrome bar on the mouth side, with its sixteen dark holes

    :param builder: Where the faces go
    """
    part = "mouthpiece"
    profile = mouth_profile()
    end_profile = inset_polygon(profile, MOUTH_END_CHAMFER)
    count = len(profile)
    front_index = MOUTH_ROUND_SEGMENTS + 1  # the side from the upper round to the lower one
    straight = BODY_HALF_LENGTH - MOUTH_END_CHAMFER
    end = BODY_HALF_LENGTH
    middle = Vector(((BODY_FRONT + COMB_FRONT) / 2, 0.0))
    front = BODY_FRONT
    flat = MOUTH_FLAT_HALF_HEIGHT
    hole = HOLE_HALF_HEIGHT

    for index in range(count):
        following = (index + 1) % count
        (y0, z0), (y1, z1) = profile[index], profile[following]
        (ey0, ez0), (ey1, ez1) = end_profile[index], end_profile[following]
        away = Vector(((y0 + y1) / 2, (z0 + z1) / 2)) - middle
        is_back = following == 0
        is_front = index == front_index
        # The top flat and its round are one curved surface, and the bottom likewise.
        group = 0 if (is_back or is_front) else (1 if (z0 + z1) > 0 else 2)

        if not is_front:
            builder.face(
                part,
                [(-straight, y0, z0), (straight, y0, z0), (straight, y1, z1), (-straight, y1, z1)],
                CHROME,
                (0, away.x, away.y),
                group,
            )

        for sign in (-1.0, 1.0):
            outward = (sign * away.length, away.x, away.y)
            if not is_front:
                builder.face(
                    part,
                    [
                        (sign * straight, y0, z0),
                        (sign * straight, y1, z1),
                        (sign * end, ey1, ez1),
                        (sign * end, ey0, ez0),
                    ],
                    CHROME,
                    outward,
                    3 if sign < 0 else 4,
                )
                continue
            # The front's end chamfer meets the hole band's corners, so it is cut to share them.
            upper, lower = (sign * straight, front, flat), (sign * straight, front, -flat)
            band_upper, band_lower = (sign * straight, front, hole), (sign * straight, front, -hole)
            end_upper, end_lower = (sign * end, ey0, ez0), (sign * end, ey1, ez1)
            for corners in (
                (upper, band_upper, end_upper),
                (band_upper, band_lower, end_upper),
                (band_lower, end_lower, end_upper),
                (band_lower, lower, end_lower),
            ):
                builder.face(part, list(corners), CHROME, outward, 5 if sign < 0 else 6)

    for sign in (-1.0, 1.0):
        builder.face(part, [(sign * end, y, z) for y, z in end_profile], CHROME, (sign, 0, 0))

    # The front face. Its breaks along X are the sides of the holes.
    breaks = [-straight]
    for centre in hole_centres():
        breaks += [centre - HOLE_HALF_WIDTH, centre + HOLE_HALF_WIDTH]
    breaks.append(straight)
    out = (0, -1, 0)
    front_group = 10

    # Above and below the holes: fans from the two outer corners, so the long edge the rounds
    # share stays one edge and no point of the hole band is left hanging on it.
    half_way = len(breaks) // 2
    for sign in (1.0, -1.0):
        left, right = (-straight, front, sign * flat), (straight, front, sign * flat)
        band = [(x, front, sign * hole) for x in breaks]
        for index in range(len(band) - 1):
            builder.face(part, [left if index < half_way else right, band[index], band[index + 1]], CHROME, out, front_group)
        builder.face(part, [left, band[half_way], right], CHROME, out, front_group)

    back = front + HOLE_DEPTH
    for index in range(len(breaks) - 1):
        x0, x1 = breaks[index], breaks[index + 1]
        if index % 2 == 0:
            builder.face(
                part, [(x0, front, -hole), (x1, front, -hole), (x1, front, hole), (x0, front, hole)], CHROME, out, front_group
            )
            continue
        # A hole: four dark walls and a dark floor, set into the bar.
        builder.face(part, [(x0, front, -hole), (x0, front, hole), (x0, back, hole), (x0, back, -hole)], PLASTIC, (1, 0, 0))
        builder.face(part, [(x1, front, -hole), (x1, front, hole), (x1, back, hole), (x1, back, -hole)], PLASTIC, (-1, 0, 0))
        builder.face(part, [(x0, front, hole), (x1, front, hole), (x1, back, hole), (x0, back, hole)], PLASTIC, (0, 0, -1))
        builder.face(part, [(x0, front, -hole), (x1, front, -hole), (x1, back, -hole), (x0, back, -hole)], PLASTIC, (0, 0, 1))
        builder.face(part, [(x0, back, -hole), (x1, back, -hole), (x1, back, hole), (x0, back, hole)], PLASTIC, out)


def cover_section(offset: float) -> list:
    """
    The top cover's outline at one place along its rounded end

    :param offset: How far past the straight part the section is, in millimetres
    :return: (Y, Z) of the outer skin there, front edge to back edge
    """
    height = math.sqrt(max(0.0, 1.0 - (offset / COVER_END_LENGTH) ** 2))
    back_edge = COVER_PROFILE[-1][0]
    section = []
    for y, z in COVER_PROFILE:
        pulled = y - (1.0 - height) * COVER_BACK_PULL * (max(y, 0.0) / back_edge) ** 2
        section.append((pulled, COVER_FOOT + (z - COVER_FOOT) * height))
    return section


def cover_height_at(y: float) -> float:
    """
    How high the top cover's skin is over a place on its straight part

    :param y: The place, in millimetres from the body's centre
    :return: Z of the skin there, in millimetres
    """
    for (y0, z0), (y1, z1) in zip(COVER_PROFILE, COVER_PROFILE[1:]):
        if y0 <= y <= y1:
            return z0 + (z1 - z0) * (y - y0) / (y1 - y0)
    raise ValueError(f"The cover does not reach Y = {y} mm")


def build_cover(builder: Builder, side: float) -> None:
    """
    One chrome cover plate: closed at its ends, open along the back where the dark shows

    :param builder: Where the faces go
    :param side: 1 for the top plate, -1 for the bottom one
    """
    part = "cover_top" if side > 0 else "cover_bottom"
    base = COVER_FOOT
    stations = [(-(COVER_STRAIGHT_HALF + step), step) for step in reversed(COVER_END_STEPS)]
    stations += [(COVER_STRAIGHT_HALF + step, step) for step in COVER_END_STEPS]
    skin_group = 30 if side > 0 else 31
    back_group = 32 if side > 0 else 33

    def at(x: float, y: float, z: float) -> tuple:
        return (x, y, side * z)

    for (xa, step_a), (xb, step_b) in zip(stations, stations[1:]):
        first, second = cover_section(step_a), cover_section(step_b)
        inside = Vector((max(-COVER_STRAIGHT_HALF, min(COVER_STRAIGHT_HALF, (xa + xb) / 2)), 3.0, side * base))

        for index in range(len(first) - 1):
            corners = [
                at(xa, *first[index]),
                at(xb, *second[index]),
                at(xb, *second[index + 1]),
                at(xa, *first[index + 1]),
            ]
            centre = sum((Vector(corner) for corner in corners), Vector()) / 4
            builder.face(part, corners, CHROME, tuple(centre - inside), skin_group)

        back_a, back_b = first[-1], second[-1]
        slot_top = COVER_PROFILE[-1][1] - COVER_LIP
        slot_floor = COVER_PROFILE[-1][0] - COVER_SLOT_DEPTH

        if step_a == 0.0 and step_b == 0.0:
            # The straight part: the plate's edge, its underside, then the dark of the open back.
            y, z = back_a
            builder.face(part, [at(xa, y, z), at(xb, y, z), at(xb, y, slot_top), at(xa, y, slot_top)], CHROME, (0, 1, 0))
            builder.face(
                part,
                [at(xa, y, slot_top), at(xb, y, slot_top), at(xb, slot_floor, slot_top), at(xa, slot_floor, slot_top)],
                CHROME,
                (0, 0, -side),
            )
            builder.face(
                part,
                [at(xa, slot_floor, slot_top), at(xb, slot_floor, slot_top), at(xb, slot_floor, base), at(xa, slot_floor, base)],
                PLASTIC,
                (0, 1, 0),
            )
            for x, look in ((xa, 1.0), (xb, -1.0)):
                builder.face(
                    part,
                    [at(x, y, slot_top), at(x, slot_floor, slot_top), at(x, slot_floor, base), at(x, y, base)],
                    CHROME,
                    (look, 0, 0),
                )
            continue

        # A rounded end: the back is a closed wall down to the comb.
        out = (0.3 if xa > 0 else -0.3, 1, 0)
        if step_a == 0.0 or step_b == 0.0:
            # Next to the open back, cut so the corner of the plate's edge is shared.
            (x_open, open_back), (x_shut, shut_back) = ((xa, back_a), (xb, back_b)) if step_a == 0.0 else ((xb, back_b), (xa, back_a))
            edge_top = at(x_open, *open_back)
            edge_lip = at(x_open, open_back[0], slot_top)
            edge_foot = at(x_open, open_back[0], base)
            shut_top = at(x_shut, *shut_back)
            shut_foot = at(x_shut, shut_back[0], base)
            for corners in ((edge_top, edge_lip, shut_top), (edge_lip, edge_foot, shut_foot), (edge_lip, shut_foot, shut_top)):
                builder.face(part, list(corners), CHROME, out, back_group)
            continue
        builder.face(
            part,
            [at(xa, *back_a), at(xb, *back_b), at(xb, back_b[0], base), at(xa, back_a[0], base)],
            CHROME,
            out,
            back_group,
        )


def build_button(builder: Builder) -> None:
    """
    The slide button: a thin shaft out of the right end and a round-ended knob on it

    :param builder: Where the faces go
    """
    part = "button"

    def ring(x: float, radius: float, sides: int) -> list:
        return [
            (x, BUTTON_Y + radius * math.cos(2 * math.pi * k / sides), BUTTON_Z + radius * math.sin(2 * math.pi * k / sides))
            for k in range(sides)
        ]

    def band(first: list, second: list, inside: Vector, group: int) -> None:
        sides = len(first)
        for k in range(sides):
            corners = [first[k], first[(k + 1) % sides], second[(k + 1) % sides], second[k]]
            centre = sum((Vector(corner) for corner in corners), Vector()) / 4
            builder.face(part, corners, CHROME, tuple(centre - inside), group)

    # The shaft starts well inside the body, past the chamfer on the mouthpiece's end, so its open
    # end is buried and no gap shows where it comes out.
    shaft_start = BODY_HALF_LENGTH - MOUTH_END_CHAMFER - 1.5
    for k in range(BUTTON_SHAFT_SIDES):
        start, stop = ring(shaft_start, BUTTON_SHAFT_RADIUS, BUTTON_SHAFT_SIDES), ring(BUTTON_KNOB_START, BUTTON_SHAFT_RADIUS, BUTTON_SHAFT_SIDES)
        following = (k + 1) % BUTTON_SHAFT_SIDES
        corners = [start[k], start[following], stop[following], stop[k]]
        centre = sum((Vector(corner) for corner in corners), Vector()) / 4
        builder.face(part, corners, CHROME, (0, centre.y - BUTTON_Y, centre.z - BUTTON_Z), 40)

    builder.face(part, ring(BUTTON_KNOB_START, BUTTON_RADIUS, BUTTON_SIDES), CHROME, (-1, 0, 0))

    inside = Vector(((BUTTON_KNOB_START + BUTTON_KNOB_FLAT_END) / 2, BUTTON_Y, BUTTON_Z))
    dome = BUTTON_TIP - BUTTON_KNOB_FLAT_END
    rings = [ring(BUTTON_KNOB_START, BUTTON_RADIUS, BUTTON_SIDES), ring(BUTTON_KNOB_FLAT_END, BUTTON_RADIUS, BUTTON_SIDES)]
    for degrees in (40.0, 70.0):
        angle = math.radians(degrees)
        rings.append(ring(BUTTON_KNOB_FLAT_END + dome * math.sin(angle), BUTTON_RADIUS * math.cos(angle), BUTTON_SIDES))
    for first, second in zip(rings, rings[1:]):
        band(first, second, inside, 41)

    tip = (BUTTON_TIP, BUTTON_Y, BUTTON_Z)
    last = rings[-1]
    for k in range(BUTTON_SIDES):
        builder.face(part, [last[k], last[(k + 1) % BUTTON_SIDES], tip], CHROME, (1, 0, 0), 41)


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
    span = 0.22
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
    Build the harmonica as one object at the world's origin

    :return: The object and the builder that holds its parts
    """
    builder = Builder()
    build_comb(builder)
    build_mouthpiece(builder)
    build_cover(builder, 1.0)
    build_cover(builder, -1.0)
    build_button(builder)

    mesh = finish_mesh(builder)
    for name, colour, metallic, roughness in MATERIALS:
        mesh.materials.append(build_material(name, colour, metallic, roughness))

    harmonica = bpy.data.objects.new(NAME, mesh)
    bpy.context.scene.collection.objects.link(harmonica)
    return harmonica, builder


def measure_parts(harmonica: object, builder: Builder) -> dict:
    """
    Work out what Parts.json says, from the mesh itself

    :param harmonica: The built object
    :param builder: The builder that holds its parts
    :return: The document
    :raises ValueError: When the body is not centred on the origin
    """
    body = ["comb", "mouthpiece", "cover_top", "cover_bottom"]
    low, high = builder.bounds(body)
    centre = (low + high) / 2
    if centre.length > 1e-6:
        raise ValueError(f"The body's centre is {tuple(centre)}, not the origin")

    mouth_low, mouth_high = builder.bounds(["mouthpiece"])
    button_low, button_high = builder.bounds(["button"])
    all_low, all_high = builder.bounds(body + ["button"])

    # Where a hand holds it: on the top plate, towards the back, a little in from each end.
    grip_x, grip_y = 70.0, 10.0
    grip_z = cover_height_at(grip_y)

    harmonica.data.calc_loop_triangles()

    def metres(vector) -> list:
        return [round(value, 5) for value in vector]

    return {
        "Name": NAME,
        "Units": "metres",
        "Axes": "X along the length (slide button at +X), Y depth (mouth side at -Y), Z up; origin at the body's centre",
        "Size": metres(high - low),
        "MouthCentre": metres(((mouth_low.x + mouth_high.x) / 2, mouth_low.y, (mouth_low.z + mouth_high.z) / 2)),
        "SlideTip": metres((button_high.x, BUTTON_Y * MM, BUTTON_Z * MM)),
        "LeftGrip": metres((-grip_x * MM, grip_y * MM, grip_z * MM)),
        "RightGrip": metres((grip_x * MM, grip_y * MM, grip_z * MM)),
        "TriangleCount": len(harmonica.data.loop_triangles),
        "BoundsWithSlide": {"Min": metres(all_low), "Max": metres(all_high)},
    }


def export_glb(harmonica: object) -> str:
    """
    Write the model alone as a GLB

    :param harmonica: The built object
    :return: Path of the file
    """
    path = os.path.join(HERE, f"{NAME}.glb")
    bpy.ops.object.select_all(action="DESELECT")
    harmonica.select_set(True)
    bpy.context.view_layer.objects.active = harmonica
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
    A plain grey seen by the camera, and a soft bright-above dark-below room seen by the metal

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
    # wall above it and a dimmer ceiling. Polished metal reads as metal by what it mirrors.
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
    ("front34", (0.006, 0.0, 0.0), (0.50, -1.0, 0.50), 0.57),
    ("back34", (0.006, 0.0, 0.0), (-0.50, 1.0, 0.50), 0.57),
    ("top", (0.006, 0.0, 0.0), (0.0, 0.0, 1.0), 0.58),
    ("mouth", (0.006, -0.018, 0.0), (0.10, -1.0, 0.24), 0.54),
)


def render_previews(scene: object) -> list:
    """
    Render the four previews, and leave their cameras in the scene

    :param scene: The scene holding the model
    :return: Paths of the pictures
    """
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    build_world(scene)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RENDER_SIDE
    scene.render.resolution_y = RENDER_SIDE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    key = area_light("PreviewKey", 0.30, 16.0)
    fill = area_light("PreviewFill", 0.60, 5.0)

    written = []
    for suffix, target, direction, distance in VIEWS:
        target_point = Vector(target)
        camera_data = bpy.data.cameras.new(f"Preview_{suffix}")
        camera_data.lens = CAMERA_LENS
        camera_data.clip_start = 0.01
        camera_data.clip_end = 10.0
        camera = bpy.data.objects.new(f"Preview_{suffix}", camera_data)
        scene.collection.objects.link(camera)

        position = target_point + Vector(direction).normalized() * distance
        if suffix == "top":
            # Straight down, the mouth side at the bottom of the picture.
            camera.location = position
            camera.rotation_euler = (0.0, 0.0, 0.0)
        else:
            aim(camera, position, target_point)
        bpy.context.view_layer.update()

        # The lights follow the camera: the key from the picture's top left, a soft fill from its right.
        basis = camera.matrix_world.to_3x3()
        right, up, back = basis @ Vector((1, 0, 0)), basis @ Vector((0, 1, 0)), basis @ Vector((0, 0, 1))
        aim(key, target_point - right * 0.34 + up * 0.42 + back * 0.30, target_point)
        aim(fill, target_point + right * 0.50 + up * 0.05 + back * 0.35, target_point)

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

    harmonica, builder = build_model()
    parts = measure_parts(harmonica, builder)

    glb = export_glb(harmonica)

    with open(os.path.join(HERE, "Parts.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(parts, handle, indent=2)
        handle.write("\n")

    skip_render = "--no-render" in sys.argv
    previews = [] if skip_render else render_previews(scene)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, f"{NAME}.blend"))

    print(f"[make_harmonica] {parts['TriangleCount']} triangles, size {parts['Size']}")
    print(f"[make_harmonica] wrote {glb}")
    for path in previews:
        print(f"[make_harmonica] wrote {path}")


if __name__ == "__main__":
    main()
