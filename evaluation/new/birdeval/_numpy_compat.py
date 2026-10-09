"""Restore the numpy aliases TrackEval still uses (`np.float`, `np.int`, `np.bool`).

numpy 1.24 removed them. TrackEval (JonathonLuiten/TrackEval, last commit 12c8791) calls
`astype(np.int)` in metrics/identity.py on every evaluation, and `np.float` in metrics/hota.py
for empty sequences. The aliases always meant the Python builtins, so restoring them as exactly
those builtins changes no result. Import this module before `trackeval`; it only adds the
aliases if numpy does not have them.
"""

import numpy as np

for _name, _type in (("float", float), ("int", int), ("bool", bool)):
    if _name not in np.__dict__:
        setattr(np, _name, _type)
