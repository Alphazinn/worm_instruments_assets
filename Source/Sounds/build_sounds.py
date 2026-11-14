"""worm_instruments' sound build (PLAN Phase 2, 2026-11-14).

Every note the script plays is a sample; this script makes them. For each instrument it fetches the CC0 recordings
its set is built from (or synthesises the notes itself), cuts and trims each note, measures its pitch, evens its
level, finds a seamless loop for sustained instruments, encodes mono 44.1 kHz OGG Vorbis into the script's
Assets/Sounds/<Instrument>/, and writes the set into Assets/Sounds/Manifest.json. It then decodes every OGG it wrote
and measures it again; Report.json beside this file keeps those measurements for the checks (checksound.py).

One method per instrument, chosen by the owner's ear (Phase 2's test, 2026-11-14): the piano and the violin are
recorded, the acoustic guitar synthesised. Tried and dropped there: a synthesised piano and violin (the recordings
won), and the University of Iowa guitar, whose chromatic runs carry each previous note's ring and the open strings'
resonance inside every note -- after its hum, clicks and hiss were taken out the owner still heard it.

Nothing here can hear: it measures pitch (cents from the note), level, clipping and loop seams; the owner judges the
sound in game.

    python -X utf8 -I build_sounds.py                 every instrument
    python -X utf8 -I build_sounds.py Piano Violin    the instruments named

Needs numpy and ffmpeg (on PATH). The downloads are cached in Cache/ (git-ignored); a second run reuses them.
"""

import io
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "Cache")
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "worm_instruments")
SOUNDS = os.path.join(SCRIPT, "Assets", "Sounds")
MANIFEST = os.path.join(SOUNDS, "Manifest.json")
REPORT = os.path.join(HERE, "Report.json")

RATE = 44100
FFMPEG = shutil.which("ffmpeg")
# Vorbis quality 5 (~160 kbps stereo, ~80 mono): transparent for a sampler's notes at a fraction of the WAV's size.
OGG_QUALITY = "5"

# Level: each note's body (its first 300 ms after the attack) is brought to this RMS, so a melody does not jump in
# loudness from one sample to the next; then the gain is lowered if the peak would pass PEAK_CEILING.
TARGET_RMS_DB = -18.0
PEAK_CEILING = 0.89  # -1 dBFS
# A note's tail is kept until its level falls this far below its loudest, then faded out.
TAIL_FLOOR_DB = -50.0
TAIL_FADE = 0.25
# Pitch is measured within this many cents of the expected note; a sample that measures past TUNE_LIMIT is the wrong
# note (or an octave error) and is reported, never silently kept.
SEARCH_CENTS = 80.0
TUNE_LIMIT = 50.0

Headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}


# -- Notes -------------------------------------------------------------------------------------------------------

NAMES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
         "A": 9, "A#": 10, "Bb": 10, "B": 11}


def Midi(Name):
    """'A#0' -> 22, 'C4' -> 60."""
    Pitch, Octave = (Name[:2], Name[2:]) if Name[1:2] in ("#", "b") else (Name[:1], Name[1:])
    return 12 * (int(Octave) + 1) + NAMES[Pitch]


def Frequency(Note):
    return 440.0 * 2.0 ** ((Note - 69) / 12.0)


# -- Audio in and out --------------------------------------------------------------------------------------------

def Decode(Path):
    """Any file ffmpeg reads, as mono float32 at RATE."""
    Result = subprocess.run([FFMPEG, "-v", "error", "-i", Path, "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
                            capture_output=True, check=True)
    return np.frombuffer(Result.stdout, dtype=np.float32).astype(np.float64)


def Encode(Samples, Path):
    os.makedirs(os.path.dirname(Path), exist_ok=True)
    Raw = np.clip(Samples, -1.0, 1.0).astype(np.float32).tobytes()
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", "-", "-c:a", "libvorbis",
                    "-q:a", OGG_QUALITY, Path], input=Raw, check=True)


def Fetch(Url, Name):
    """A download kept in Cache/ under its own name; fetched only when missing."""
    Path = os.path.join(CACHE, Name)
    if os.path.isfile(Path) and os.path.getsize(Path) > 0:
        return Path
    os.makedirs(os.path.dirname(Path), exist_ok=True)
    print("  fetch", Name)
    with urllib.request.urlopen(urllib.request.Request(Url, headers=Headers), timeout=300) as Response:
        Data = Response.read()
    with io.open(Path + ".part", "wb") as Handle:
        Handle.write(Data)
    os.replace(Path + ".part", Path)
    return Path


def GitHubRaw(Repo, Path):
    return "https://raw.githubusercontent.com/%s/master/%s" % (Repo, urllib.parse.quote(Path))


# -- Measuring ---------------------------------------------------------------------------------------------------

def Envelope(Samples, Hop=512):
    """RMS per hop of Hop samples."""
    Count = len(Samples) // Hop
    if Count == 0:
        return np.zeros(1)
    return np.sqrt(np.mean(Samples[:Count * Hop].reshape(Count, Hop) ** 2, axis=1) + 1e-20)


def Db(Value):
    return 20.0 * math.log10(max(Value, 1e-12))


def MeasureCents(Samples, Note, Start, Seconds=0.5, IsVibrato=False, Search=SEARCH_CENTS):
    """How far the note at Start sounds from Note, in cents: the strongest of its first two partials near where they
    should be, read from a zero-padded spectrum with a parabolic peak. The second partial carries the low notes whose
    fundamental is weak; a piano's stretch is under 2 cents there. A vibrato note's partial is a carrier with
    sidebands the vibrato's rate apart, and a sideband can be the tallest (the synthesised violin read up to +9 cents
    by its peak): there the partial is read as the power-weighted centre of its window instead."""
    Length = int(min(Seconds * RATE, len(Samples) - Start))
    if Length < 2048:
        return None
    Segment = Samples[Start:Start + Length] * np.hanning(Length)
    Size = 1 << 20
    Spectrum = np.abs(np.fft.rfft(Segment, Size))
    Bin = RATE / Size
    Expected = Frequency(Note)
    Found = []
    for Partial in (1, 2):
        Low = int(Partial * Expected * 2 ** (-Search / 1200) / Bin)
        High = int(Partial * Expected * 2 ** (Search / 1200) / Bin) + 1
        if High >= len(Spectrum) - 1 or Low < 1:
            continue
        Index = Low + int(np.argmax(Spectrum[Low:High]))
        if IsVibrato:
            Power = Spectrum[Low:High] ** 2
            Centre = float(np.sum(Power * np.arange(Low, High)) / (np.sum(Power) + 1e-20))
            Found.append((Spectrum[Index], Centre * Bin / Partial))
            continue
        A, B, C = np.log(Spectrum[Index - 1:Index + 2] + 1e-20)
        Offset = 0.5 * (A - C) / (A - 2 * B + C) if (A - 2 * B + C) != 0 else 0.0
        Found.append((Spectrum[Index], (Index + Offset) * Bin / Partial))
    if not Found:
        return None
    Strongest = max(Magnitude for Magnitude, _ in Found)
    Kept = [(Magnitude, Hertz) for Magnitude, Hertz in Found if Magnitude >= 0.25 * Strongest]
    Hertz = sum(Magnitude * Value for Magnitude, Value in Kept) / sum(Magnitude for Magnitude, _ in Kept)
    return 1200.0 * math.log2(Hertz / Expected)


def Onset(Samples, Share=0.02):
    """Where the note starts, less 5 ms of pre-roll: found backwards, from its loudest 3 ms to the last moment before
    it under Share of that level. 2 % catches a bow's slow swell; reading backwards, a click or a stray noise ahead of
    the note is cut away with the dip that parts it from the note (reading forwards, a click was the first thing over
    the line and kept 0.6 s of noise before an Iowa guitar note, heard by the owner as a 'pit', 2026-11-14)."""
    Hop = 128
    Env = Envelope(Samples, Hop)
    Loudest = int(np.argmax(Env))
    Under = np.nonzero(Env[:Loudest] < Share * Env[Loudest])[0]
    Start = int(Under[-1]) + 1 if len(Under) else 0
    return max(0, Start * Hop - int(0.005 * RATE))


# -- Shaping a note ----------------------------------------------------------------------------------------------

def Trim(Samples, MaxSeconds):
    """From the onset to where the tail falls TAIL_FLOOR_DB under the peak (at most MaxSeconds), faded out."""
    Start = Onset(Samples)
    Note = Samples[Start:]
    Env = Envelope(Note)
    Loud = np.nonzero(Env > Env.max() * 10 ** (TAIL_FLOOR_DB / 20))[0]
    End = min(len(Note), int((Loud[-1] + 1) * 512) if len(Loud) else len(Note), int(MaxSeconds * RATE))
    Note = Note[:End].copy()
    Fade = min(int(TAIL_FADE * RATE), len(Note) // 4)
    if Fade > 0:
        Note[-Fade:] *= np.linspace(1.0, 0.0, Fade)
    Attack = int(0.001 * RATE)
    Note[:Attack] *= np.linspace(0.0, 1.0, Attack)
    return Note


def Level(Note, BodyStart):
    """The gain that brings the note's body (300 ms from BodyStart seconds) to TARGET_RMS_DB, lowered so its peak stays
    under PEAK_CEILING."""
    Body = Note[int(BodyStart * RATE):int((BodyStart + 0.3) * RATE)]
    Rms = math.sqrt(float(np.mean(Body ** 2))) if len(Body) else 1e-6
    Gain = 10 ** (TARGET_RMS_DB / 20) / max(Rms, 1e-9)
    Peak = float(np.max(np.abs(Note))) + 1e-12
    return min(Gain, PEAK_CEILING / Peak)


def FindLoop(Note, Earliest, Latest, MinLength):
    """A loop inside [Earliest, Latest) seconds for a sustained note: the start at a rising zero crossing, the end where
    the waveform around it matches the start's best (normalised cross-correlation over 4096 samples, about two vibrato
    cycles' worth of shape at the top of the range). Returns (start, end, score) in samples, or None."""
    Window = 4096
    Start = int(Earliest * RATE)
    Crossings = np.nonzero((Note[Start:Start + 4000][:-1] <= 0) & (Note[Start:Start + 4000][1:] > 0))[0]
    Start += int(Crossings[0]) + 1 if len(Crossings) else 0
    Template = Note[Start - Window // 2:Start + Window // 2]
    First = Start + int(MinLength * RATE)
    Last = min(int(Latest * RATE), len(Note) - Window)
    if Last <= First or len(Template) < Window:
        return None
    Search = Note[First - Window // 2:Last + Window // 2]
    Size = 1 << int(math.ceil(math.log2(len(Search) + Window)))
    Correlation = np.fft.irfft(np.fft.rfft(Search, Size) * np.conj(np.fft.rfft(Template, Size)), Size)[:len(Search) - Window + 1]
    Energy = np.sqrt(np.convolve(Search ** 2, np.ones(Window), "valid")[:len(Correlation)] + 1e-20)
    Score = Correlation / (Energy * (np.sqrt(np.sum(Template ** 2)) + 1e-20))
    Best = int(np.argmax(Score))
    return Start, First + Best, float(Score[Best])


def Crossfade(Note, Start, End, Seconds):
    """Bakes the jump from End back to Start into the samples before End: they fade into the ones before Start, so the
    loop's seam is the same waveform on both sides."""
    Fade = min(int(Seconds * RATE), Start)
    Weight = np.linspace(0.0, 1.0, Fade)
    Note[End - Fade:End] = Note[End - Fade:End] * (1 - Weight) + Note[Start - Fade:Start] * Weight


# -- Synthesis ---------------------------------------------------------------------------------------------------

Random = np.random.default_rng(20261114)


def SynthGuitar(Note):
    """A plucked steel string by Karplus-Strong: a burst of noise, shaped by where the string is plucked, running
    round a delay line one period long that loses a little and dulls a little each pass (the fractional part of the
    period goes in the two-tap average, so the pitch is exact), then a small wooden body's resonances added."""
    F0 = Frequency(Note)
    Seconds = min(6.0, max(1.8, 6.0 * 2 ** (-(Note - 40) / 36.0)))
    Total = int(Seconds * RATE)
    Period = RATE / F0
    # The two-tap average delays by its weight on the older tap: a whole delay of Whole plus Fraction is the period.
    Whole = int(Period)
    Fraction = Period - Whole
    Decay = 10 ** (-3.0 / (Seconds * 0.8 * F0))
    Excite = np.convolve(Random.uniform(-1, 1, Whole), np.ones(2) / 2, "same")
    Excite = Excite - np.roll(Excite, int(Whole * 0.17))
    Line = np.zeros(Total + Whole + 2)
    Line[1:Whole + 1] = Excite
    for Start in range(Whole + 1, len(Line), Whole):
        End = min(Start + Whole, len(Line))
        Line[Start:End] = Decay * ((1 - Fraction) * Line[Start - Whole:End - Whole] + Fraction * Line[Start - Whole - 1:End - Whole - 1])
    String = Line[1:Total + 1]
    Time = np.arange(int(0.25 * RATE)) / RATE
    Body = np.zeros_like(Time)
    Body[0] = 1.0
    for Hertz, Damping, Gain in ((98.0, 0.035, 0.6), (204.0, 0.03, 0.8), (390.0, 0.02, 0.5), (560.0, 0.015, 0.35), (1050.0, 0.008, 0.2)):
        Body += Gain * 0.02 * np.exp(-Time / Damping) * np.sin(2 * math.pi * Hertz * Time)
    Size = 1 << int(math.ceil(math.log2(len(String) + len(Body))))
    Out = np.fft.irfft(np.fft.rfft(String, Size) * np.fft.rfft(Body, Size), Size)[:len(String)]
    return Out


# -- The sets ----------------------------------------------------------------------------------------------------

def EveryThird(Low, High):
    return list(range(Low, High + 1, 3))


def PianoRecorded():
    """VCSL's Steinway B, sustain pedal down, middle velocity layer (vl3): 42 notes, a whole tone apart."""
    Names = ["A#0"] + ["%s%d" % (Name, Octave) for Octave in range(1, 8) for Name in ("C", "D", "E", "F#", "G#", "A#")
                       if not (Octave == 7 and Name == "A#")]
    for Name in Names:
        Path = Fetch(GitHubRaw("sgossner/VCSL", "Chordophones/Zithers/Grand Piano, Steinway B/Sus/JHPiano_Sus_Close_%s_vl3_rr1.wav" % Name),
                     "VCSL/Steinway/%s.wav" % Name.replace("#", "s"))
        Note = Midi(Name)
        yield Note, Trim(Decode(Path), min(10.0, max(3.0, 12.0 * 2 ** (-(Note - 21) / 30.0)))), None


def ViolinRecorded():
    """VSCO 2 CE's solo violin, arco with vibrato, forte: 15 notes; each gets a crossfaded loop in its sustain."""
    for Name in ("G3", "A3", "C4", "E4", "G4", "A4", "C5", "E5", "G5", "A5", "C6", "E6", "G6", "A6", "C7"):
        Path = Fetch(GitHubRaw("sgossner/VSCO-2-CE", "Strings/Solo Violin/Arco Vib/LLVln_ArcoVib_%s_f.wav" % Name),
                     "VSCO/Violin/%s.wav" % Name)
        Raw = Decode(Path)
        Note = Raw[Onset(Raw):].copy()
        Env = Envelope(Note)
        Steady = float(np.median(Env[int(0.3 * RATE / 512):]))
        Fading = np.nonzero(Env[int(0.5 * RATE / 512):] < 0.5 * Steady)[0]
        Release = (Fading[0] + int(0.5 * RATE / 512)) * 512 / RATE if len(Fading) else len(Note) / RATE
        # The loop's start is tried at a few places after the bow has settled; the best-matching seam is kept.
        Tries = [FindLoop(Note, Earliest, Release - 0.1, min(0.7, max(0.3, Release - Earliest - 0.3)))
                 for Earliest in (0.45, 0.55, 0.65, 0.75)]
        Tries = [Found for Found in Tries if Found is not None]
        if not Tries:
            yield Midi(Name), None, "no room for a loop (release at %.2f s)" % Release
            continue
        Start, End, Score = max(Tries, key=lambda Found: Found[2])
        Crossfade(Note, Start, End, 0.12)
        Clip = Note[:End].copy()
        Clip[:int(0.001 * RATE)] *= np.linspace(0, 1, int(0.001 * RATE))
        yield Midi(Name), Clip, (Start, End, Score)


def GuitarSynth():
    for Note in EveryThird(40, 82):
        yield Note, Trim(SynthGuitar(Note), 6.0), None


# Each instrument's set: its method (the owner's pick), how it is made, its range (MIDI), the release on note-off
# (seconds), its articulation, and where it came from (for CREDITS.md). Each instrument phase adds its own.
SETS = {
    "Piano": ("Recorded", PianoRecorded, (21, 108), 0.35, "Sustain", "VCSL, Grand Piano, Steinway B (CC0)"),
    "Violin": ("Recorded", ViolinRecorded, (55, 100), 0.18, "Arco", "VSCO 2 CE, Solo Violin, Arco Vib (CC0)"),
    "AcousticGuitar": ("Synth", GuitarSynth, (40, 84), 0.25, "Pluck", "synthesised by build_sounds.py"),
}


# -- The build ---------------------------------------------------------------------------------------------------

# The sets whose own tuning is kept: a piano is tuned stretched on purpose (its bass flat, its treble sharp, so its
# octaves sound true against their stiff strings' sharp partials), and the Steinway measured just that, -20 to -38
# cents at the bottom and +7 to +16 at the top (2026-11-14). Correcting each note to equal temperament would undo it;
# instead a smooth curve is fitted through the measurements and only each note's step off that curve is corrected.
STRETCHED = {"Piano"}


def StretchCurve(Notes, Cents):
    """A cubic through (note, cents), refitted once without the points over 2.5 deviations off it: a stray note does not
    bend the piano's curve. Returns the curve as a function of the note."""
    X = np.array(Notes, dtype=float)
    Y = np.array(Cents, dtype=float)
    Keep = np.ones(len(X), dtype=bool)
    for _ in range(2):
        Fit = np.polyfit(X[Keep], Y[Keep], 3)
        Residual = Y - np.polyval(Fit, X)
        Keep = np.abs(Residual) <= 2.5 * (np.std(Residual[Keep]) + 1e-9)
    return lambda Note: float(np.polyval(Fit, Note))


def Build(Instrument, Manifest, Report):
    Method, Maker, Range, Release, Articulation, Source = SETS[Instrument]
    Folder = os.path.join(SOUNDS, Instrument)
    if os.path.isdir(Folder):
        for Name in os.listdir(Folder):
            if Name.endswith(".ogg"):
                os.remove(os.path.join(Folder, Name))
    Samples = []
    Rows = []
    # A bowed note swells for its first few hundred ms: its pitch and its body are read once it has settled. A struck
    # or plucked one is read just after its attack, before it decays.
    Settle = 0.4 if Articulation == "Arco" else 0.1
    Made = []
    for Note, Clip, Loop in Maker():
        if Clip is None:
            Rows.append({"Root": Note, "Problem": Loop})
            print("  ! %s %d: %s" % (Instrument, Note, Loop))
            continue
        At = int(Settle * RATE) if len(Clip) > (Settle + 0.6) * RATE else int(0.03 * RATE)
        Window = 0.5
        if Loop is not None:
            # A vibrato note is read over its whole loop: a 0.5 s window holds part of a vibrato cycle and reads its
            # average off by up to 9 cents (measured on the synthesised violin, exact by construction).
            At, Window = Loop[0], (Loop[1] - Loop[0]) / RATE
        Made.append((Note, Clip, Loop, At, Window, MeasureCents(Clip, Note, At, Window, Loop is not None)))

    Measurements = [(Note, Measured) for Note, _, _, _, _, Measured in Made if Measured is not None]
    Curve = StretchCurve(*zip(*Measurements)) if Instrument in STRETCHED and len(Measurements) >= 6 else (lambda Note: 0.0)

    for Note, Clip, Loop, At, Window, Measured in Made:
        Stretch = round(Curve(Note), 1)
        Problem = None
        if Measured is None or abs(Measured - Stretch) > TUNE_LIMIT:
            Problem = "pitch %s cents from the note" % ("unmeasured" if Measured is None else "%.1f" % Measured)
        Gain = Level(Clip, At / RATE)
        Clip = Clip * Gain
        File = "%s/%d.ogg" % (Instrument, Note)
        Encode(Clip, os.path.join(SOUNDS, *File.split("/")))
        Back = Decode(os.path.join(SOUNDS, *File.split("/")))
        After = MeasureCents(Back, Note, At, Window, Loop is not None)
        Tune = round(Stretch - (Measured or 0.0), 1)
        Entry = {"File": File, "Root": Note, "Tune": Tune, "LoopStart": 0.0, "LoopEnd": 0.0}
        Row = {"File": File, "Root": Note, "Measured": round(Measured, 2) if Measured is not None else None,
               "MeasuredAfter": round(After, 2) if After is not None else None, "Tune": Tune, "Stretch": Stretch,
               "PeakDb": round(Db(float(np.max(np.abs(Back)))), 2), "Seconds": round(len(Back) / RATE, 3),
               "IsClipped": bool(np.max(np.abs(Back)) >= 0.999), "Problem": Problem}
        if Loop is not None:
            Start, End, Score = Loop
            Entry["LoopStart"] = round(Start / RATE, 5)
            Entry["LoopEnd"] = round(End / RATE, 5)
            Row["LoopScore"] = round(Score, 3)
            Row["LoopStart"], Row["LoopEnd"] = Entry["LoopStart"], Entry["LoopEnd"]
        if Problem:
            print("  ! %s: %s" % (File, Problem))
        Samples.append(Entry)
        Rows.append(Row)
        print("  %-34s %+6.1f c  plays %+6.1f c (curve %+5.1f)  peak %6.1f dB  %5.2f s%s" % (
            File, Measured or 0.0, (After or 0.0) + Tune, Stretch, Row["PeakDb"], Row["Seconds"],
            "  loop %.3f-%.3f (%.3f)" % (Entry["LoopStart"], Entry["LoopEnd"], Row["LoopScore"]) if Loop else ""))
    Manifest["Sets"][Instrument] = {"Method": Method, "Range": list(Range), "Release": Release, "Articulation": Articulation,
                                    "Source": Source, "Samples": Samples}
    Report["Sets"][Instrument] = Rows


def Load(Path):
    if not os.path.isfile(Path):
        return {"Version": 1, "Sets": {}}
    with io.open(Path, "r", encoding="utf-8") as Handle:
        return json.load(Handle)


def Save(Path, Data):
    with io.open(Path, "w", encoding="utf-8", newline="\n") as Handle:
        json.dump(Data, Handle, indent=4, ensure_ascii=False)
        Handle.write("\n")


if __name__ == "__main__":
    if FFMPEG is None:
        raise SystemExit("ffmpeg is not on PATH")
    Asked = sys.argv[1:] or list(SETS)
    Manifest = Load(MANIFEST)
    Report = Load(REPORT)
    for Instrument in Asked:
        print(Instrument)
        Build(Instrument, Manifest, Report)
        Save(MANIFEST, Manifest)
        Save(REPORT, Report)
    print("done:", ", ".join(Asked))
