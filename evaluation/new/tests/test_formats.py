import pandas as pd
import pytest

from conftest import FIXTURES
from birdeval.formats import (
    CORNERS, LABELLED, LEGACY, read_corner_tracks, read_gt_tracks, read_labelled_tracks, read_legacy_tracks, tracker_format,
)


def test_gt_reader_converts_corners_and_handles_crlf():
    gt = read_gt_tracks(FIXTURES / "swap_gt.csv")  # written with \r\n line endings
    assert list(gt.columns) == ["frame", "id", "x", "y", "w", "h"]
    assert len(gt) == 20 and set(gt["id"]) == {1, 2}
    first = gt.iloc[0]
    assert (first["frame"], first["id"], first["x"], first["y"], first["w"], first["h"]) == (0, 1, 100.0, 100.0, 10.0, 10.0)


def test_format_detection(tmp_path):
    labelled = tmp_path / "l.csv"
    labelled.write_text("frame_number,track_id,x,y,w,h,row_type,track_confirmed\n0,1,1,1,2,2,matched,True\n")
    corners = tmp_path / "c.csv"
    corners.write_text("frame_number,obj_id,x1,y1,x2,y2,bbox_size,frame_timestamp\n0,5,1,1,3,4,6,2025-01-01\n")
    assert tracker_format(labelled) == LABELLED
    assert tracker_format(FIXTURES / "swap_tracks.csv") == LEGACY
    assert tracker_format(corners) == CORNERS
    c = read_corner_tracks(corners).iloc[0]
    assert (c["x"], c["y"], c["w"], c["h"]) == (1.0, 1.0, 2.0, 3.0)


def test_labelled_reader_keeps_labels_and_allows_degenerate_predicted_boxes(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text(
        "frame_number,track_id,x,y,w,h,row_type,track_confirmed\n"
        "0,1,10,10,5,5,pre_confirmation,True\n"
        "1,1,11,10,5,5,matched,True\n"
        "2,1,-90,10,-0.5,-0.2,predicted,True\n"
        "0,2,50,50,5,5,pre_confirmation,False\n"
    )
    table = read_labelled_tracks(path)
    assert table["track_confirmed"].dtype == bool
    assert list(table.sort_values(["id", "frame"])["row_type"]) == ["pre_confirmation", "matched", "predicted", "pre_confirmation"]


def test_degenerate_detection_boxes_are_rejected(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("frame_number,track_id,x,y,w,h,row_type,track_confirmed\n0,1,10,10,0,5,matched,True\n")
    with pytest.raises(ValueError, match="non-positive"):
        read_labelled_tracks(path)


def test_duplicate_id_in_one_frame_is_rejected(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("frame_number,track_id,x,y,w,h\n0,1,1,1,2,2\n0,1,5,5,2,2\n")
    with pytest.raises(ValueError, match="more than once"):
        read_legacy_tracks(path)


def test_gt_may_contain_two_ids_with_the_same_box(tmp_path):
    # The 1DAC GT has one frame where two birds share an identical box: that is data, not an error.
    path = tmp_path / "gt.csv"
    path.write_text("frame_number,obj_id,x1,y1,x2,y2,bbox_size\r\n5,1,0,0,4,4,16\r\n5,2,0,0,4,4,16\r\n")
    assert len(read_gt_tracks(path)) == 2
    assert isinstance(read_gt_tracks(path), pd.DataFrame)
