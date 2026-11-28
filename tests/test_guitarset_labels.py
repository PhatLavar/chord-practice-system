import sys
from pathlib import Path
import unittest

try:
    from acr.data.guitarset.labels import map_symbol, progression_signature, fold_role
except ModuleNotFoundError as error:
    if error.name != "mir_eval":
        raise
    map_symbol = None


@unittest.skipIf(map_symbol is None, "Install requirements-eval.txt")
class GuitarSetLabels(unittest.TestCase):
    def test_exact_spelling_and_enharmonic_root(self):
        self.assertEqual(map_symbol("Db:(1,3,5)/5"), map_symbol("C#:maj"))
        self.assertEqual(map_symbol("C:(1,b3,b5,bb7)")["quality"], "dim7")

    def test_ambiguity_stays_masked(self):
        for label in ["C:(1,5)", "C:maj(*3)", "C:9", "C:maj(#9)", "C:maj(*1)"]:
            target = map_symbol(label)
            self.assertEqual(target["complete"], -100, label)
            self.assertEqual(target["triad"], -100, label)
            self.assertEqual(target["extension"], -100, label)
            self.assertEqual(target["root"], 0)
        self.assertEqual(map_symbol("N")["gate"], 1)
        self.assertEqual(map_symbol("X")["gate"], -100)

    def test_progression_invariance_and_difference(self):
        events = lambda labels: [dict(value=v) for v in labels]
        self.assertEqual(
            progression_signature(events(["C:maj", "C:maj", "G:7"])),
            progression_signature(events(["D:maj", "A:7"])),
        )
        self.assertNotEqual(
            progression_signature(events(["C:maj", "G:7"])),
            progression_signature(events(["C:maj", "G:maj"])),
        )

    def test_fold_roles(self):
        for heldout in range(3):
            self.assertEqual(
                {fold_role(g, heldout) for g in range(3)}, {"train", "validation", "test"}
            )
            self.assertEqual(fold_role(heldout, heldout), "test")
