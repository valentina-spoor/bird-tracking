| Subsystem           | Current Spoor files                                                      |
| ------------------- | ------------------------------------------------------------------------ |
| Motion model        | `kalman_filter.py`, `kalman_tracker_state.py`                            |
| Gating              | `kalman_filter.py`, `linear_assignment.py`                               |
| Association cost    | `nn_matching.py`, `distance_matching.py`                                 |
| Assignment          | `linear_assignment.py`, `matching.py`                                    |
| Track management    | `simple_sort_tracker.py`, `tracked_object.py`, `tracked_object_state.py` |
| Data representation | `shim.py`                                                                |
| Execution/IO        | `run_tracker.py`                                                         |
| Post-processing     | `simple_sort_tracker.py` (`filter_deleted_tracks`, `concatenate_tracks`) |
