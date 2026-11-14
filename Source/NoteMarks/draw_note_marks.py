"""Draws worm_instruments' four note-mark textures (PLAN Phase 4) into Textures/ beside this file.

Run with `python -X utf8 draw_note_marks.py` (Pillow). The specs in Specs/ draw these PNGs; FiveM-PTFX-Forge's `pack`
embeds them in the particle dictionary stream/worm_instr_marks.ypt.

Each is a white note on a transparent square with a thin, half-dark outline, so it reads on a bright sky with an
alpha-blended rule; an additive rule draws the white alone (black adds no light). Drawn at four times the size and
scaled down for smooth edges.
"""

import math
import os

from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Textures")
SIZE = 256
SCALE = 4
BIG = SIZE * SCALE
WHITE = (255, 255, 255, 255)
OUTLINE = (0, 0, 0, 150)
OUTLINE_WIDTH = 14 * SCALE // 4 * 2
HEAD_W = 300
HEAD_H = 210
HEAD_TILT = -22
STEM_W = 46
STEM_H = 560


def Head(Draw, Cx, Cy, Fill):
    """A tilted filled note head centred on Cx, Cy, as a polygon (PIL's ellipse cannot turn)."""
    Points = []
    for Step in range(72):
        Angle = 2 * math.pi * Step / 72
        X = HEAD_W / 2 * math.cos(Angle)
        Y = HEAD_H / 2 * math.sin(Angle)
        Turn = math.radians(HEAD_TILT)
        Points.append((Cx + X * math.cos(Turn) - Y * math.sin(Turn), Cy + X * math.sin(Turn) + Y * math.cos(Turn)))
    Draw.polygon(Points, fill=Fill)


def Stem(Draw, Cx, Cy, Fill):
    """The stem up from the head's right side; returns its top (x, y)."""
    Left = Cx + HEAD_W / 2 - STEM_W - 6
    Draw.rectangle([Left, Cy - STEM_H, Left + STEM_W, Cy - 20], fill=Fill)
    return Left, Cy - STEM_H


def Flag(Draw, Top, Fill, Drop=0, Reach=1.0):
    """One flag hanging from the stem's top: a curve that leaves the stem thick, swings right and tapers down."""
    X, Y = Top
    Y += Drop
    P0 = (X + STEM_W / 2, Y + 10)
    P1 = (X + 40, Y + 150 * Reach)
    P2 = (X + 290 * Reach, Y + 170 * Reach)
    P3 = (X + 190 * Reach, Y + 400 * Reach)
    Steps = 160
    for Step in range(Steps + 1):
        T = Step / Steps
        Px = (1 - T) ** 3 * P0[0] + 3 * (1 - T) ** 2 * T * P1[0] + 3 * (1 - T) * T ** 2 * P2[0] + T ** 3 * P3[0]
        Py = (1 - T) ** 3 * P0[1] + 3 * (1 - T) ** 2 * T * P1[1] + 3 * (1 - T) * T ** 2 * P2[1] + T ** 3 * P3[1]
        Radius = STEM_W / 2 + 14 * math.sin(math.pi * T * 0.8) - 12 * T
        Draw.ellipse([Px - Radius, Py - Radius, Px + Radius, Py + Radius], fill=Fill)


def Shape(Kind, Fill):
    Layer = Image.new("RGBA", (BIG, BIG), (0, 0, 0, 0))
    Draw = ImageDraw.Draw(Layer)
    if Kind == "beamed":
        Heads = [(330, 760), (720, 680)]
        Tops = []
        for Cx, Cy in Heads:
            Head(Draw, Cx, Cy, Fill)
            Tops.append(Stem(Draw, Cx, Cy, Fill))
        (X1, Y1), (X2, Y2) = Tops
        Draw.polygon([(X1, Y1), (X2 + STEM_W, Y2), (X2 + STEM_W, Y2 + 90), (X1, Y1 + 90)], fill=Fill)
    else:
        Cx, Cy = 420, 760
        Head(Draw, Cx, Cy, Fill)
        Top = Stem(Draw, Cx, Cy, Fill)
        if Kind in ("eighth", "sixteenth"):
            Flag(Draw, Top, Fill)
        if Kind == "sixteenth":
            Flag(Draw, Top, Fill, Drop=150, Reach=0.8)
    return Layer


def Mark(Kind):
    Glyph = Shape(Kind, WHITE)
    Mask = Glyph.split()[3].filter(ImageFilter.MaxFilter(OUTLINE_WIDTH + 1))
    Outline = Image.new("RGBA", (BIG, BIG), OUTLINE)
    Result = Image.new("RGBA", (BIG, BIG), (0, 0, 0, 0))
    Result.paste(Outline, (0, 0), Mask)
    Result.alpha_composite(Glyph)
    return Result.resize((SIZE, SIZE), Image.LANCZOS)


os.makedirs(OUT, exist_ok=True)
for Name in ("quarter", "eighth", "sixteenth", "beamed"):
    Path = os.path.join(OUT, "mark_%s.png" % Name)
    Mark(Name).save(Path)
    print("wrote", Path)
