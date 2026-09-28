# Copied verbatim from libs/tracker/src/spoortracker/tracked_object_state.py at spoor-main
# commit 145fc6696241ec3ea1e1d3b930164bebf226538b (origin/main). See ../PROVENANCE.md.
from enum import Enum


class TrackedObjectState(str, Enum):
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"
