"""Build the worm_instr_tipcase prop from nothing.

Run headless:
    blender -b --python make_tipcase.py

Writes, next to this file:
    worm_instr_tipcase.glb     the model the prop converter takes
    worm_instr_tipcase.blend   the same model with the preview cameras, lights and ground
    Textures/*.png             the flat colour pictures the materials wear
    Parts.json                 sizes and the points the script drops tips at
    Preview/*.png              four renders

Add `-- --no-render` to skip the previews (the saved scene then has no cameras or lights), or
`-- --draft` to render them with few samples: a fifth of the work, and a little rougher.

An acoustic guitar's hard case lying flat on the ground, open and empty, for tips to land in.
Stylised, with no brand, badge or text on it.

Axes (metres): X along the case's length (the wide end at -X, the neck's end at +X), Y across it,
Z up. The busker stands on the -Y side and the audience on the +Y side: the lid is hinged on the
-Y side and has fallen right back there, its lining up and its far edge resting on the ground;
the handle and the latches are on the +Y side. The origin is on the ground under the middle of
the tray: nothing is below Z = 0.

The lid is opened as far as it goes, which is worked out and not set: the angle at which its
lowest point is on the ground (Parts.json says it). The tray's rim is higher than the lid is
deep, so that is a little past flat: the lid slopes down from its hinges to its far edge.

The lid's hinges sit where a shaped case has them, on the line that touches both of its wide
parts on the hinge side. That line is not quite square to the length, so the open lid is turned
a few degrees in plan (Parts.json says how many).

The game's own case (sf_prop_sf_guitar_case_01a, which is closed) was measured for size only, with
Anim-Forge's `forge inspect-model` and the XML it writes: 1.257 m long, 0.514 m across its wide
end, about 0.375 m across its shoulders and 0.155 m across its neck, 0.182 m deep, latches and
handle on one long side. Nothing of it is copied.

The materials are plain colours, but each wears a small flat picture: the prop converter rebuilds
every material as a game shader and carries over pictures only, never a colour set on the shader.
"""

import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

NAME = "worm_instr_tipcase"
HERE = os.path.dirname(os.path.abspath(__file__))
PREVIEW_DIR = os.path.join(HERE, "Preview")
TEXTURE_DIR = os.path.join(HERE, "Textures")

#: The geometry below is written in millimetres and scaled to metres as it is built.
MM = 0.001

SHELL = 0
PIPING = 1
LINING = 2
METAL = 3

#: Faces of one smoothing group shade as one curved surface where they meet under this angle.
SMOOTH_ANGLE = math.radians(50.0)

UP = Vector((0.0, 0.0, 1.0))

# ---- the case's outline, in millimetres ---------------------------------------------------

#: One half of the case seen from above: how far along it from the wide end, and half its width
#: there. A smooth curve is drawn through these and mirrored for the other half.
OUTLINE_HALF = (
    (0.0, 0.0),
    (4.0, 70.0),
    (22.0, 140.0),
    (60.0, 200.0),
    (115.0, 240.0),
    (185.0, 255.0),  # the wide end at its widest
    (245.0, 247.0),
    (300.0, 224.0),
    (350.0, 195.0),
    (390.0, 177.0),
    (420.0, 172.0),  # the waist
    (455.0, 175.0),
    (500.0, 186.0),
    (545.0, 195.0),
    (580.0, 196.0),  # the shoulders
    (615.0, 188.0),
    (648.0, 168.0),
    (678.0, 138.0),
    (706.0, 110.0),
    (735.0, 94.0),
    (775.0, 86.0),
    (830.0, 85.0),  # the neck, straight from here
    (1195.0, 85.0),
    (1228.0, 79.0),
    (1244.0, 64.0),
    (1250.0, 40.0),
    (1250.0, 0.0),
)
#: How finely the curve is drawn before the points it does not need are dropped again.
OUTLINE_STEPS = 16
#: A point is dropped when the outline without it strays no further than this.
OUTLINE_TOLERANCE = 2.0
#: Where along the case the waist is: the hinge line is the one that bridges it.
WAIST_ALONG = 420.0

# ---- the two halves, in millimetres -------------------------------------------------------

TRAY_HEIGHT = 115.0  # the tray's rim above the ground
LID_DEPTH = 68.0  # the lid from its rim to its outside
EDGE = 7.0  # the piping round the bottom and round the lid's top, a slanted strip this big
SEAM_BAND = 9.0  # the piping along each rim
RIM = 15.0  # the rim's width, shell to lining
PADDING = 22.0  # how far in the lining's wall stands at its foot
FLOOR = 22.0  # the tray's lining above the ground
CEILING = 18.0  # the lid's lining in from its outside
SOFT_CORNER = 13.0  # the round where the lining's wall meets its floor

#: Between which angles the lid comes to rest on the ground: 0 is shut, 90 upright, 180 flat on
#: its back. The angle itself is found between these, where the lid's lowest point is at Z = 0.
LID_ANGLE_RANGE = (120.0, 215.0)
#: The parts that swing with the lid.
LID_PARTS = ("lid", "lid_latches", "lid_hinges")
#: How far outside the shell the hinge pins are.
HINGE_OUT = 3.0
HINGE_PIN_RADIUS = 5.0
HINGE_PIN_HALF = 26.0
HINGE_PIN_SIDES = 8
HINGE_LEAF_HALF = (22.0, 2.75, 13.0)  # along the hinge line, out from the shell, up

# The latches, on the +Y side: where along the case each one is.
LATCHES_ALONG = (185.0, 590.0, 1050.0)
LATCH_PLATE_HALF = (17.0, 2.5, 15.0)  # along the shell, out from it, up
LATCH_PLATE_BELOW_RIM = 30.0
LATCH_CLASP_HALF = (11.0, 17.0, 1.5)  # along the shell, along the clasp, through it
LATCH_CLASP_SWING = 28.0  # degrees the open clasp hangs out from straight down
LATCH_KEEPER_HALF = (13.0, 2.5, 9.0)
LATCH_KEEPER_ABOVE_RIM = 17.0
LATCH_HOOK_HALF = (7.0, 1.5, 3.0)
LATCH_HOOK_ABOVE_RIM = 12.0

# The handle, on the +Y side across the waist.
HANDLE_ALONG = (345.0, 495.0)
HANDLE_BELOW_RIM = 40.0
HANDLE_BRACKET_HALF = (9.0, 5.0, 13.0)
HANDLE_REACH = 34.0  # how far the grip stands off the line between its brackets
HANDLE_DROOP = 20.0  # degrees the grip hangs below level
HANDLE_HALF_WIDTH = 12.0
HANDLE_HALF_THICKNESS = 4.5
HANDLE_STEPS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

# The neck rest: the padded block a guitar's neck lies on, across the narrow part of the tray.
REST_ALONG = (800.0, 900.0)
REST_HALF_WIDTH = 69.0  # a little into the lining on both sides
REST_TOP = 70.0
REST_DIP_TOP_HALF = 30.0
REST_DIP_FLOOR_HALF = 17.0
REST_DIP_FLOOR = 50.0

# Where the script drops tips: the middle of the wide part of the tray, and half the size of the
# flat area round it that is all lining.
TIP_ALONG = 310.0
TIP_HALF_SIZE = (150.0, 130.0)

# ---- colours ------------------------------------------------------------------------------

#: name, colour as seen (sRGB), metallic, roughness
MATERIALS = (
    ("shell", (0.085, 0.085, 0.092), 0.0, 0.65),
    ("piping", (0.55, 0.53, 0.49), 0.0, 0.6),
    ("lining", (0.28, 0.03, 0.065), 0.0, 0.95),  # burgundy plush
    ("metal", (0.74, 0.72, 0.67), 1.0, 0.35),
)
TEXTURE_SIDE = 8

# ---- previews -----------------------------------------------------------------------------

RENDER_SIDE = 1024
RENDER_SAMPLES = 160
DRAFT_SIDE = 1024
DRAFT_SAMPLES = 32
CAMERA_LENS = 85.0
BACKGROUND_GREY = 0.215  # linear; mid grey once shown
#: How much bigger than the harmonica's the preview set is: its lights' sizes and distances are the
#: harmonica's times this, and their power times its square, so the case is lit as the harmonica is.
PREVIEW_SCALE = 6.5


class Builder:
    """Collects every part into one mesh, welding points part by part."""

    def __init__(self) -> None:
        self.bm = bmesh.new()
        self.group_layer = self.bm.faces.layers.int.new("group")
        self.points: dict = {}
        self.part_points: dict = {}
        self.flat_group = 10_000
        #: When set, every point and direction given is moved by this first (the open lid's swing).
        self.matrix = None

    def vert(self, part: str, point: tuple):
        """
        Return the vertex of a part at a point, making it the first time

        :param part: Name of the part the point belongs to
        :param point: (x, y, z) in millimetres
        :return: The bmesh vertex
        """
        if self.matrix is not None:
            point = self.matrix @ Vector(point)
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

        look = Vector(outward)
        if self.matrix is not None:
            look = self.matrix.to_3x3() @ look

        normal = Vector((0.0, 0.0, 0.0))
        for index, vertex in enumerate(verts):
            current, following = vertex.co, verts[(index + 1) % len(verts)].co
            normal.x += (current.y - following.y) * (current.z + following.z)
            normal.y += (current.z - following.z) * (current.x + following.x)
            normal.z += (current.x - following.x) * (current.y + following.y)
        if normal.length < 1e-14:
            return None
        if normal.dot(look) < 0.0:
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

    def slab(self, part: str, centre: Vector, halves: tuple, material: int, skip: tuple = ()) -> None:
        """
        Add a box that may stand any way round

        :param part: Name of the part
        :param centre: Its middle, in millimetres
        :param halves: Three vectors from its middle to the middles of three of its sides
        :param material: Material slot
        :param skip: The sides left out because they are buried, as (which vector, sign) pairs
        """
        for axis in range(3):
            first, second = [halves[index] for index in range(3) if index != axis]
            for sign in (-1.0, 1.0):
                if (axis, sign) in skip:
                    continue
                middle = centre + halves[axis] * sign
                corners = [tuple(middle + first * a + second * b) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                self.face(part, corners, material, tuple(halves[axis] * sign))

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


def curve_point(before: Vector, start: Vector, end: Vector, after: Vector, along: float) -> Vector:
    """
    A point of the smooth curve through four points, between the middle two

    The curve is a centripetal Catmull-Rom one: it passes through every point given and does not
    swing wide of them where they are unevenly spaced.

    :param before: The point before the stretch
    :param start: Where the stretch starts
    :param end: Where it ends
    :param after: The point after it
    :param along: 0 at the start of the stretch, 1 at its end
    :return: The point
    """
    t0 = 0.0
    t1 = t0 + math.sqrt((start - before).length)
    t2 = t1 + math.sqrt((end - start).length)
    t3 = t2 + math.sqrt((after - end).length)
    t = t1 + (t2 - t1) * along
    a1 = before * ((t1 - t) / (t1 - t0)) + start * ((t - t0) / (t1 - t0))
    a2 = start * ((t2 - t) / (t2 - t1)) + end * ((t - t1) / (t2 - t1))
    a3 = end * ((t3 - t) / (t3 - t2)) + after * ((t - t2) / (t3 - t2))
    b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0))
    b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1))
    return b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1))


def drop_unneeded(points: list, tolerance: float) -> list:
    """
    Thin out a line's points, keeping its shape (Douglas and Peucker's way)

    :param points: The line's points, in order
    :param tolerance: How far the thinned line may stray from the given one
    :return: The points kept, the first and the last among them
    """
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    spans = [(0, len(points) - 1)]
    while spans:
        first, last = spans.pop()
        start, end = points[first], points[last]
        chord = end - start
        worst, worst_index = 0.0, -1
        for index in range(first + 1, last):
            offset = points[index] - start
            if chord.length < 1e-9:
                distance = offset.length
            else:
                distance = abs(chord.x * offset.y - chord.y * offset.x) / chord.length
            if distance > worst:
                worst, worst_index = distance, index
        if worst > tolerance:
            keep[worst_index] = True
            spans += [(first, worst_index), (worst_index, last)]
    return [point for point, kept in zip(points, keep) if kept]


class Outline:
    """The case seen from above, its middle over the origin."""

    def __init__(self) -> None:
        given = [Vector(point) for point in OUTLINE_HALF]
        # The curve runs on into the mirrored half at both ends, so it crosses the middle line square.
        padded = [Vector((given[1].x, -given[1].y))] + given + [Vector((given[-2].x, -given[-2].y))]
        fine = []
        for index in range(1, len(padded) - 2):
            for step in range(OUTLINE_STEPS):
                fine.append(curve_point(*padded[index - 1 : index + 3], step / OUTLINE_STEPS))
        fine.append(given[-1])
        half = drop_unneeded(fine, OUTLINE_TOLERANCE)

        lowest, highest = min(point.x for point in half), max(point.x for point in half)
        #: What is taken off a place along the case to get its X.
        self.shift = (lowest + highest) / 2
        #: The +Y half, from the wide end's middle to the neck end's middle.
        self.half = [Vector((point.x - self.shift, max(point.y, 0.0))) for point in half]
        self.half[0].y = self.half[-1].y = 0.0
        #: The whole outline as (x, y) corners.
        self.points = [tuple(point) for point in self.half]
        self.points += [(point.x, -point.y) for point in reversed(self.half[1:-1])]

    def wall_at(self, along: float, side: float) -> tuple:
        """
        Find the shell at a place along the case

        :param along: The place, in millimetres from the wide end
        :param side: 1 for the +Y side, -1 for the -Y side
        :return: The point of the outline there (on the ground), the way along the shell there
            (towards the neck) and the way out of it, all as 3D vectors
        :raises ValueError: When the place is not on a side of the case
        """
        x = along - self.shift
        for start, end in zip(self.half, self.half[1:]):
            if not (start.x <= x <= end.x) or end.x - start.x < 1e-6:
                continue
            point = start.lerp(end, (x - start.x) / (end.x - start.x))
            way = (end - start).normalized()
            return (
                Vector((point.x, side * point.y, 0.0)),
                Vector((way.x, side * way.y, 0.0)),
                Vector((-way.y, side * way.x, 0.0)),
            )
        raise ValueError(f"No side of the case at {along} mm along it")

    def hinge_line(self) -> tuple:
        """
        Find the straight line the lid swings on: the one touching both wide parts on the -Y side

        :return: The two points of the outline the line touches, the one at the wide end first
        :raises ValueError: When no such line is found
        """
        # The outline's convex hull (Andrew's monotone chain); the line is the hull's side over the waist.
        ordered = sorted(set(self.points))

        def chain(points: list) -> list:
            kept: list = []
            for point in points:
                while len(kept) >= 2 and (
                    (kept[-1][0] - kept[-2][0]) * (point[1] - kept[-2][1])
                    - (kept[-1][1] - kept[-2][1]) * (point[0] - kept[-2][0])
                ) <= 0:
                    kept.pop()
                kept.append(point)
            return kept

        lower = chain(ordered)  # the hull's -Y side, from -X to +X
        waist = WAIST_ALONG - self.shift
        for start, end in zip(lower, lower[1:]):
            if start[0] < waist < end[0] and start[1] < 0 and end[1] < 0:
                return Vector(start), Vector(end)
        raise ValueError("No straight line bridges the waist on the hinge side")


def lid_swing(outline: Outline, angle: float) -> tuple:
    """
    Work out how the shut lid is moved to where it is when opened by an angle

    :param outline: The case's outline
    :param angle: How far the lid is opened, in degrees: 0 shut, 90 upright, 180 flat on its back
    :return: The move as a matrix, the way along the hinge line and the way out of the shell
        there (3D vectors), and how far the line is turned from the case's length, in degrees
    """
    touch_wide, touch_shoulder = outline.hinge_line()
    way = (touch_shoulder - touch_wide).normalized()
    along = Vector((way.x, way.y, 0.0))
    out = Vector((way.y, -way.x, 0.0))
    pin = Vector((touch_wide.x, touch_wide.y, TRAY_HEIGHT)) + out * HINGE_OUT
    swing = Matrix.Translation(pin) @ Matrix.Rotation(math.radians(angle), 4, along) @ Matrix.Translation(-pin)
    return swing, along, out, math.degrees(math.atan2(way.y, way.x))


def resting_lid_angle(outline: Outline) -> float:
    """
    Find how far the lid opens before it rests on the ground

    The lid and what is fixed to it are built shut and swung by trial: the angle wanted is the one
    that puts their lowest point at Z = 0, to well under a hundredth of a millimetre.

    :param outline: The case's outline
    :return: The angle in degrees
    :raises ValueError: When the lid does not reach the ground inside LID_ANGLE_RANGE
    """
    shut = Builder()
    still = Matrix.Identity(4)
    _, along, out, _ = lid_swing(outline, 0.0)
    build_half(shut, "lid", outline, True)
    build_latches(shut, outline, still)
    build_hinges(shut, outline, still, along, out)
    points = [vertex.co / MM for part in LID_PARTS for vertex in shut.part_points[part]]
    shut.bm.free()

    def lowest(angle: float) -> float:
        swing = lid_swing(outline, angle)[0]
        return min((swing @ point).z for point in points)

    above, below = LID_ANGLE_RANGE
    if lowest(above) < 0.0 or lowest(below) > 0.0:
        raise ValueError(f"The lid does not come to rest between {above} and {below} degrees")
    for _ in range(60):
        middle = (above + below) / 2
        if lowest(middle) >= 0.0:
            above = middle
        else:
            below = middle
    return above


def build_half(builder: Builder, part: str, outline: Outline, is_lid: bool) -> None:
    """
    One half of the case as a skin: the outside, the rim, the lining's wall and its flat

    The lid is built shut, on top of the tray; the builder's matrix swings it open.

    :param builder: Where the faces go
    :param part: Name of the part
    :param outline: The case's outline
    :param is_lid: True for the lid, False for the tray
    """
    rim = TRAY_HEIGHT
    if is_lid:
        outside = rim + LID_DEPTH
        flat = outside - CEILING
        towards_rim = -1.0
    else:
        outside = 0.0
        flat = FLOOR
        towards_rim = 1.0
    group = 20 if is_lid else 10

    #: How far in from the outline, how high, then the material and the smoothing group of the
    #: strip to the next: from the edge of the outside's flat, over the rim, to the lining's flat.
    stations = (
        (EDGE, outside, PIPING, group + 1),
        (0.0, outside + towards_rim * EDGE, SHELL, group + 2),
        (0.0, rim - towards_rim * SEAM_BAND, PIPING, group + 2),
        (0.0, rim, SHELL, group + 3),
        (RIM, rim, LINING, group + 4),
        (PADDING, flat + towards_rim * SOFT_CORNER, LINING, group + 4),
        (PADDING + SOFT_CORNER, flat, LINING, group + 4),
    )
    rings = [[(x, y, height) for x, y in inset_polygon(outline.points, inset)] for inset, height, _, _ in stations]

    builder.face(part, rings[0], SHELL, (0, 0, -towards_rim))
    builder.face(part, rings[-1], LINING, (0, 0, towards_rim))

    corners = outline.points
    count = len(corners)
    area = sum(
        corners[i][0] * corners[(i + 1) % count][1] - corners[(i + 1) % count][0] * corners[i][1] for i in range(count)
    )
    turn = 1.0 if area > 0 else -1.0

    for index in range(len(stations) - 1):
        (inset0, z0, material, strip_group), (inset1, z1) = stations[index], stations[index + 1][:2]
        # The skin's front is to one side of the way the stations run: the right on the tray
        # (bottom, outside, rim, inside), the left on the lid, which runs the other way up.
        away, lift = (z1 - z0) * towards_rim, (inset1 - inset0) * towards_rim
        first, second = rings[index], rings[index + 1]
        for k in range(count):
            following = (k + 1) % count
            side = Vector((corners[following][0] - corners[k][0], corners[following][1] - corners[k][1])).normalized()
            out = Vector((side.y, -side.x)) * turn
            builder.face(
                part,
                [first[k], first[following], second[following], second[k]],
                material,
                (out.x * away, out.y * away, lift),
                strip_group,
            )


def build_rest(builder: Builder, outline: Outline) -> None:
    """
    The neck rest: a padded block across the narrow part of the tray, dipped where a neck lies

    Its sides and its foot are buried in the lining.

    :param builder: Where the faces go
    :param outline: The case's outline
    """
    part = "rest"
    x0, x1 = (along - outline.shift for along in REST_ALONG)
    top = [
        (-REST_HALF_WIDTH, REST_TOP),
        (-REST_DIP_TOP_HALF, REST_TOP),
        (-REST_DIP_FLOOR_HALF, REST_DIP_FLOOR),
        (REST_DIP_FLOOR_HALF, REST_DIP_FLOOR),
        (REST_DIP_TOP_HALF, REST_TOP),
        (REST_HALF_WIDTH, REST_TOP),
    ]
    foot = FLOOR - 2.0
    end = top + [(REST_HALF_WIDTH, foot), (-REST_HALF_WIDTH, foot)]
    builder.face(part, [(x0, y, z) for y, z in end], LINING, (-1, 0, 0))
    builder.face(part, [(x1, y, z) for y, z in end], LINING, (1, 0, 0))
    for (y0, z0), (y1, z1) in zip(top, top[1:]):
        builder.face(part, [(x0, y0, z0), (x1, y0, z0), (x1, y1, z1), (x0, y1, z1)], LINING, (0, 0, 1))


def build_latches(builder: Builder, outline: Outline, swing: Matrix) -> None:
    """
    The latches: on the tray a plate with its clasp hanging open, on the lid's rim the keeper it grips

    :param builder: Where the faces go
    :param outline: The case's outline
    :param swing: The move that takes the shut lid to where it lies open
    """
    # A plate's back is kept though it is against the shell: the shell curves, and where it falls
    # away from a plate's ends the gap would show straight through the plate from behind.
    lean = math.radians(LATCH_CLASP_SWING)
    for along in LATCHES_ALONG:
        point, way, out = outline.wall_at(along, 1.0)

        plate = point + out * (LATCH_PLATE_HALF[1] - 1.0) + UP * (TRAY_HEIGHT - LATCH_PLATE_BELOW_RIM)
        builder.slab(
            "latches",
            plate,
            (way * LATCH_PLATE_HALF[0], out * LATCH_PLATE_HALF[1], UP * LATCH_PLATE_HALF[2]),
            METAL,
        )

        # The clasp hangs from the plate's lower edge, swung out and down.
        hang = out * math.sin(lean) - UP * math.cos(lean)
        through = out * math.cos(lean) + UP * math.sin(lean)
        pivot = plate + out * LATCH_PLATE_HALF[1] - UP * (LATCH_PLATE_HALF[2] - 1.0)
        builder.slab(
            "latches",
            pivot + hang * (LATCH_CLASP_HALF[1] - 1.0),
            (way * LATCH_CLASP_HALF[0], hang * LATCH_CLASP_HALF[1], through * LATCH_CLASP_HALF[2]),
            METAL,
        )

        builder.matrix = swing
        builder.slab(
            "lid_latches",
            point + out * (LATCH_KEEPER_HALF[1] - 1.0) + UP * (TRAY_HEIGHT + LATCH_KEEPER_ABOVE_RIM),
            (way * LATCH_KEEPER_HALF[0], out * LATCH_KEEPER_HALF[1], UP * LATCH_KEEPER_HALF[2]),
            METAL,
        )
        builder.slab(
            "lid_latches",
            point + out * (2 * LATCH_KEEPER_HALF[1] - 1.0 + LATCH_HOOK_HALF[1]) + UP * (TRAY_HEIGHT + LATCH_HOOK_ABOVE_RIM),
            (way * LATCH_HOOK_HALF[0], out * LATCH_HOOK_HALF[1], UP * LATCH_HOOK_HALF[2]),
            METAL,
            ((1, -1.0),),  # its back is flat against the keeper
        )
        builder.matrix = None


def build_handle(builder: Builder, outline: Outline) -> None:
    """
    The handle: two metal brackets on the shell and a flat grip arched between them, hanging a little

    :param builder: Where the faces go
    :param outline: The case's outline
    """
    part = "handle"
    height = TRAY_HEIGHT - HANDLE_BELOW_RIM
    ends = []
    for along in HANDLE_ALONG:
        point, way, out = outline.wall_at(along, 1.0)
        centre = point + out * (HANDLE_BRACKET_HALF[1] - 1.0) + UP * height
        builder.slab(
            part,
            centre,
            (way * HANDLE_BRACKET_HALF[0], out * HANDLE_BRACKET_HALF[1], UP * HANDLE_BRACKET_HALF[2]),
            METAL,
        )
        # The grip's ends are buried in the brackets.
        ends.append(point + out * (HANDLE_BRACKET_HALF[1] + 1.0) + UP * height)

    start, end = ends
    span = end - start
    level = Vector((-span.y, span.x, 0.0)).normalized()
    if level.y < 0:
        level = -level
    droop = math.radians(HANDLE_DROOP)
    reach = level * math.cos(droop) - UP * math.sin(droop)

    centres = [start + span * s + reach * (HANDLE_REACH * (1.0 - (2.0 * s - 1.0) ** 4)) for s in HANDLE_STEPS]
    sections = []
    for index, centre in enumerate(centres):
        way = (centres[min(index + 1, len(centres) - 1)] - centres[max(index - 1, 0)]).normalized()
        outer = (reach - way * reach.dot(way)).normalized()
        side = way.cross(outer).normalized()
        sections.append(
            (
                [
                    centre + side * (a * HANDLE_HALF_WIDTH) + outer * (b * HANDLE_HALF_THICKNESS)
                    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ],
                outer,
                side,
            )
        )
    for (first, outer, side), (second, _, _) in zip(sections, sections[1:]):
        for k, (look, group) in enumerate(((-outer, 31), (side, 32), (outer, 33), (-side, 34))):
            following = (k + 1) % 4
            builder.face(
                part,
                [tuple(first[k]), tuple(first[following]), tuple(second[following]), tuple(second[k])],
                SHELL,
                tuple(look),
                group,
            )


def build_hinges(builder: Builder, outline: Outline, swing: Matrix, along: Vector, out: Vector) -> None:
    """
    The two hinges: a pin on the hinge line, a leaf on the tray under it and a leaf on the lid

    :param builder: Where the faces go
    :param outline: The case's outline
    :param swing: The move that takes the shut lid to where it lies open
    :param along: The way along the hinge line
    :param out: The way out of the shell there
    """
    leaf = (along * HINGE_LEAF_HALF[0], out * HINGE_LEAF_HALF[1], UP * HINGE_LEAF_HALF[2])
    for touch in outline.hinge_line():
        base = Vector((touch.x, touch.y, 0.0))
        pin = base + out * HINGE_OUT + UP * TRAY_HEIGHT

        rings = []
        for end in (-HINGE_PIN_HALF, HINGE_PIN_HALF):
            rings.append(
                [
                    pin
                    + along * end
                    + (out * math.cos(2 * math.pi * k / HINGE_PIN_SIDES) + UP * math.sin(2 * math.pi * k / HINGE_PIN_SIDES))
                    * HINGE_PIN_RADIUS
                    for k in range(HINGE_PIN_SIDES)
                ]
            )
        for k in range(HINGE_PIN_SIDES):
            following = (k + 1) % HINGE_PIN_SIDES
            angle = 2 * math.pi * (k + 0.5) / HINGE_PIN_SIDES
            builder.face(
                "hinges",
                [tuple(rings[0][k]), tuple(rings[0][following]), tuple(rings[1][following]), tuple(rings[1][k])],
                METAL,
                tuple(out * math.cos(angle) + UP * math.sin(angle)),
                41,
            )
        builder.face("hinges", [tuple(point) for point in rings[0]], METAL, tuple(-along))
        builder.face("hinges", [tuple(point) for point in rings[1]], METAL, tuple(along))

        # The leaves are sunk into the shell: it curves away from the hinge line to both sides.
        seat = base + out * (HINGE_LEAF_HALF[1] - 2.5)
        builder.slab("hinges", seat + UP * (TRAY_HEIGHT - HINGE_LEAF_HALF[2] - 2.0), leaf, METAL)
        builder.matrix = swing
        builder.slab("lid_hinges", seat + UP * (TRAY_HEIGHT + HINGE_LEAF_HALF[2] + 2.0), leaf, METAL)
        builder.matrix = None


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
    span = 1.4
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
    Build the open case as one object at the world's origin

    :return: The object, the builder that holds its parts, the outline, how far the lid is
        opened and how far the hinge line is turned from the case's length, both in degrees
    """
    outline = Outline()
    lid_angle = resting_lid_angle(outline)
    swing, along, out, turned = lid_swing(outline, lid_angle)

    builder = Builder()
    build_half(builder, "tray", outline, False)
    builder.matrix = swing
    build_half(builder, "lid", outline, True)
    builder.matrix = None
    build_rest(builder, outline)
    build_latches(builder, outline, swing)
    build_handle(builder, outline)
    build_hinges(builder, outline, swing, along, out)

    mesh = finish_mesh(builder)
    for name, colour, metallic, roughness in MATERIALS:
        mesh.materials.append(build_material(name, colour, metallic, roughness))

    case = bpy.data.objects.new(NAME, mesh)
    bpy.context.scene.collection.objects.link(case)
    return case, builder, outline, lid_angle, turned


def inside_polygon(point: tuple, corners: list) -> bool:
    """
    Tell whether a point is inside a flat polygon

    :param point: (x, y) of the point
    :param corners: The polygon's corners, in order
    :return: True when it is inside
    """
    inside = False
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        if (y0 > point[1]) != (y1 > point[1]) and point[0] < x0 + (point[1] - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def measure_parts(case: object, builder: Builder, outline: Outline, lid_angle: float, turned: float) -> dict:
    """
    Work out what Parts.json says, from the mesh itself

    :param case: The built object
    :param builder: The builder that holds its parts
    :param outline: The case's outline
    :param lid_angle: How far the lid is opened, in degrees
    :param turned: How far the hinge line is turned from the case's length, in degrees
    :return: The document
    :raises ValueError: When the tray is not on the ground over the origin, the lid does not rest
        on the ground, or the tips' area is not all on the tray's floor
    """
    tray_low, tray_high = builder.bounds(["tray"])
    if abs(tray_low.z) > 1e-6 or abs(tray_low.x + tray_high.x) > 1e-6 or abs(tray_low.y + tray_high.y) > 1e-6:
        raise ValueError(f"The tray spans {tuple(tray_low)} to {tuple(tray_high)}: not on the ground over the origin")
    lid_low, lid_high = builder.bounds(list(LID_PARTS))
    if abs(lid_low.z) > 1e-6:
        raise ValueError(f"The lid's lowest point is at {lid_low.z} m, not on the ground")
    all_low, all_high = builder.bounds(list(builder.part_points))
    if all_low.z < -1e-6:
        raise ValueError(f"Something reaches {all_low.z} m under the ground")

    tip_x = TIP_ALONG - outline.shift
    floor = inset_polygon(outline.points, PADDING + SOFT_CORNER)
    for a in (-1, 1):
        for b in (-1, 1):
            corner = (tip_x + a * TIP_HALF_SIZE[0], b * TIP_HALF_SIZE[1])
            if not inside_polygon(corner, floor):
                raise ValueError(f"The tips' area reaches {corner} mm, off the tray's floor")

    case.data.calc_loop_triangles()

    def metres(vector) -> list:
        return [round(value, 5) for value in vector]

    return {
        "Name": NAME,
        "Units": "metres",
        "Axes": (
            "X along the length (wide end at -X), Y across (lid hinged at -Y, the busker's side, and lying back on "
            "the ground there with its lining up; handle and latches at +Y, the audience's side), Z up; origin on "
            "the ground under the tray's middle"
        ),
        "Size": metres(all_high - all_low),
        "Bounds": {"Min": metres(all_low), "Max": metres(all_high)},
        "TrayBounds": {"Min": metres(tray_low), "Max": metres(tray_high)},
        "LidBounds": {"Min": metres(lid_low), "Max": metres(lid_high)},
        "LidAngle": round(lid_angle, 2),
        "HingeLineTurn": round(turned, 2),
        "TipCentre": metres((tip_x * MM, 0.0, FLOOR * MM)),
        "TipHalfSize": metres((TIP_HALF_SIZE[0] * MM, TIP_HALF_SIZE[1] * MM)),
        "OutlinePoints": len(outline.points),
        "TriangleCount": len(case.data.loop_triangles),
    }


def export_glb(case: object) -> str:
    """
    Write the model alone as a GLB

    :param case: The built object
    :return: Path of the file
    """
    path = os.path.join(HERE, f"{NAME}.glb")
    bpy.ops.object.select_all(action="DESELECT")
    case.select_set(True)
    bpy.context.view_layer.objects.active = case
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
    ("front34", (0.0, -0.24, 0.03), (0.55, 1.0, 0.55), 4.3),  # from the audience's side
    ("top", (0.0, -0.25, 0.0), (0.0, 0.0, 1.0), 4.3),
    ("side", (0.0, -0.25, 0.06), (1.0, 0.0, 0.08), 3.3),  # from the neck's end
    ("player", (0.0, -0.24, 0.03), (-0.20, -1.0, 1.30), 4.3),  # as the busker looks down at it
)


def render_previews(scene: object, draft: bool) -> list:
    """
    Render the four previews, and leave their cameras in the scene

    :param scene: The scene holding the model
    :param draft: True for quick pictures of few samples
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

    case, builder, outline, lid_angle, turned = build_model()
    parts = measure_parts(case, builder, outline, lid_angle, turned)

    glb = export_glb(case)

    with open(os.path.join(HERE, "Parts.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(parts, handle, indent=2)
        handle.write("\n")

    skip_render = "--no-render" in sys.argv
    previews = [] if skip_render else render_previews(scene, "--draft" in sys.argv)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, f"{NAME}.blend"))

    print(f"[make_tipcase] {parts['TriangleCount']} triangles, size {parts['Size']}")
    print(f"[make_tipcase] wrote {glb}")
    for path in previews:
        print(f"[make_tipcase] wrote {path}")


if __name__ == "__main__":
    main()
