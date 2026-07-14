"""
Tests for the reference-sheet compositor (image_generation/reference_sheet.py)
and the character view selection logic (scene_reference_sheet.py).
"""

import os

import pytest
from PIL import Image

from virtual_streamer.image_generation.reference_sheet import (
    MAX_CELLS,
    ReferenceSheet,
    SheetCell,
    _pack_rows,
    _position_labels,
    build_reference_sheet,
)
from virtual_streamer.video_generation.scene_reference_sheet import select_character_views
from virtual_streamer.video_server.models import LabeledImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(tmp_path, name: str, size=(400, 300)) -> str:
    path = str(tmp_path / name)
    Image.new("RGB", size, (200, 50, 50)).save(path)
    return path


def _cells(tmp_path, specs) -> list:
    """specs: list of (category, description) — one synthetic image each."""
    return [
        SheetCell(
            image_path=_make_image(tmp_path, f"img_{i}.png", size=(300 + 100 * (i % 3), 300)),
            category=cat,
            description=desc,
        )
        for i, (cat, desc) in enumerate(specs)
    ]


# ---------------------------------------------------------------------------
# Layout math
# ---------------------------------------------------------------------------

class TestLayout:
    def test_pack_rows_matches_freshmart_example(self):
        assert _pack_rows(7) == [3, 2, 2]

    @pytest.mark.parametrize("n,expected", [
        (1, [1]), (2, [2]), (3, [3]), (4, [2, 2]), (6, [3, 3]), (9, [3, 3, 3]),
    ])
    def test_pack_rows(self, n, expected):
        assert _pack_rows(n) == expected

    def test_position_labels_three_rows(self):
        labels = _position_labels([3, 2, 2])
        assert labels == [
            "Top Row Left", "Top Row Middle", "Top Row Right",
            "Middle Row Left", "Middle Row Right",
            "Bottom Row Left", "Bottom Row Right",
        ]

    def test_position_labels_single_cell(self):
        assert _position_labels([1]) == ["Top Row"]


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

class TestBuildReferenceSheet:
    def test_canvas_matches_video_resolution(self, tmp_path):
        cells = _cells(tmp_path, [("Setting", "A store"), ("Character", "An owl")])
        sheet = build_reference_sheet(cells, "1280x720", str(tmp_path / "out"))
        assert sheet is not None
        with Image.open(sheet.image_path) as img:
            assert img.size == (1280, 720)

    def test_description_format(self, tmp_path):
        cells = _cells(tmp_path, [
            ("Setting", "A grocery store exterior."),
            ("Character", "An owl mascot."),
            ("Props", "Blue tote bags."),
            ("Logo", "The FreshMart logo."),
        ])
        sheet = build_reference_sheet(cells, "1280x720", str(tmp_path / "out"))
        assert sheet.description.startswith("### Reference Sheet Description")
        assert "**Top Row Left (Setting):** A grocery store exterior." in sheet.description
        assert "(Character):** An owl mascot." in sheet.description
        assert "(Logo):** The FreshMart logo." in sheet.description

    def test_category_ordering(self, tmp_path):
        # Given out of order, Setting must come first and Logo last
        cells = _cells(tmp_path, [
            ("Logo", "logo"), ("Character", "char"), ("Setting", "setting"), ("Props", "prop"),
        ])
        sheet = build_reference_sheet(cells, "1280x720", str(tmp_path / "out"))
        d = sheet.description
        assert d.index("(Setting)") < d.index("(Character)") < d.index("(Props)") < d.index("(Logo)")

    def test_caps_at_max_cells(self, tmp_path):
        cells = _cells(tmp_path, [("Props", f"prop {i}") for i in range(12)])
        sheet = build_reference_sheet(cells, "1280x720", str(tmp_path / "out"))
        assert sheet.cell_count == MAX_CELLS

    def test_missing_images_skipped(self, tmp_path):
        cells = _cells(tmp_path, [("Setting", "ok")])
        cells.append(SheetCell(image_path="/nonexistent.png", category="Props", description="gone"))
        sheet = build_reference_sheet(cells, "1280x720", str(tmp_path / "out"))
        assert sheet.cell_count == 1
        assert "gone" not in sheet.description

    def test_no_cells_returns_none(self, tmp_path):
        assert build_reference_sheet([], "1280x720", str(tmp_path / "out")) is None


# ---------------------------------------------------------------------------
# Character view selection
# ---------------------------------------------------------------------------

class TestSelectCharacterViews:
    def test_prefers_distinct_angles(self):
        character = {
            "labeled_images": [
                {"image_path": "a.png", "angle": "front"},
                {"image_path": "b.png", "angle": "front"},
                {"image_path": "c.png", "angle": "back"},
                {"image_path": "d.png", "angle": "three_quarter"},
            ],
            "identity_images": ["a.png", "b.png", "c.png", "d.png"],
        }
        views = select_character_views(character)
        assert [v.image_path for v in views] == ["a.png", "c.png", "d.png"]

    def test_unlabeled_falls_back_to_first_n(self):
        character = {"identity_images": [f"{i}.png" for i in range(5)], "labeled_images": []}
        views = select_character_views(character)
        assert [v.image_path for v in views] == ["0.png", "1.png", "2.png"]

    def test_fewer_than_max_returns_all(self):
        character = {"identity_images": ["x.png"], "labeled_images": []}
        assert len(select_character_views(character)) == 1

    def test_unlabeled_identity_images_merged_with_labeled(self):
        character = {
            "labeled_images": [{"image_path": "a.png", "angle": "front", "caption": "front view"}],
            "identity_images": ["a.png", "b.png"],
        }
        views = select_character_views(character)
        paths = [v.image_path for v in views]
        assert paths == ["a.png", "b.png"]
        assert views[0].caption == "front view"
