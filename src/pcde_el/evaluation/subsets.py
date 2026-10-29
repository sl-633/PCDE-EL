"""IN-D0 / OUT-D0 split of the test queries.

IN-D0 queries already match some D0 name after normalization, so expansion
cannot help them find the entry and can only add competitors. OUT-D0 queries
are where the new names can add coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcde_el.normalization import build_lexical_index, normalize

LEVEL = 1


def in_d0_mask(d0: pd.DataFrame, mentions: pd.Series) -> np.ndarray:
    # Any D0 name counts, not just a name of the gold entry.
    keys = set(build_lexical_index(d0["name"], d0["pid"], level=LEVEL))
    return mentions.map(lambda m: normalize(m, LEVEL) in keys).to_numpy(dtype=bool)


def splits(d0: pd.DataFrame, mentions: pd.Series) -> dict[str, np.ndarray]:
    in_d0 = in_d0_mask(d0, mentions)
    return {"all": np.ones(len(in_d0), dtype=bool), "IN-D0": in_d0, "OUT-D0": ~in_d0}
