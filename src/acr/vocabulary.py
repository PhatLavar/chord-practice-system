"""Shared complete-chord vocabulary and decomposition; no data/model imports."""

# (triad, extension); inversions collapse to the same root and quality.
COMPONENTS = {
    "maj": ("maj", "none"),
    "min": ("min", "none"),
    "dim": ("dim", "none"),
    "aug": ("aug", "none"),
    "sus2": ("sus2", "none"),
    "sus4": ("sus4", "none"),
    "maj6": ("maj", "6"),
    "min6": ("min", "6"),
    "7": ("maj", "b7"),
    "min7": ("min", "b7"),
    "maj7": ("maj", "7"),
    "minmaj7": ("min", "7"),
    "hdim7": ("dim", "b7"),
    "dim7": ("dim", "bb7"),
}
PITCH = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)


def mapping(symbol):
    if symbol in {"N", "X"}:
        return symbol, "", "", ""
    root, _, quality = symbol.partition("/")[0].partition(":")
    if quality not in COMPONENTS:
        return "OOV", "", "", ""
    pitch = (PITCH[root[0]] + root.count("#") - root.count("b")) % 12
    return quality, pitch, *COMPONENTS[quality]


QUALITIES = list(COMPONENTS)
TRIADS = list(dict.fromkeys(t for t, e in COMPONENTS.values()))
EXTENSIONS = list(dict.fromkeys(e for t, e in COMPONENTS.values()))
CLASSES = ["N"] + [f"{root}:{quality}" for root in range(12) for quality in QUALITIES]


def encode(symbol):
    if symbol == "N":
        return 0
    quality, root, _, _ = mapping(symbol)
    return 1 + root * len(QUALITIES) + QUALITIES.index(quality) if quality in COMPONENTS else -100


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_label(value):
    return "N" if value == 0 else NOTES[(value - 1) // 14] + ":" + QUALITIES[(value - 1) % 14]
