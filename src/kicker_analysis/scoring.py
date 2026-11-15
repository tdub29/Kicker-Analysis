"""Shared, explicit kicker scoring. Blocked kicks are misses on Sleeper.

Source: https://support.sleeper.com/en/articles/3998131
Raw outcomes stay separate. An unexplained attempt is not silently a miss.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import numpy as np

MADE_BUCKETS = ("fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
                "fg_made_40_49", "fg_made_50_59", "fg_made_60_")


@dataclass(frozen=True)
class KickerScoring:
    fg_made: tuple[float, ...] = (3., 3., 3., 4., 5., 5.)
    pat_made: float = 1.
    fg_missed: float = 0.
    pat_missed: float = 0.

    def __post_init__(self):
        if len(self.fg_made) != 6 or not all(math.isfinite(x) for x in
                (*self.fg_made, self.pat_made, self.fg_missed, self.pat_missed)):
            raise ValueError("six finite FG bucket scores and finite PAT/miss scores required")

    @property
    def identity(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    @classmethod
    def from_sleeper(cls, settings: dict) -> "KickerScoring":
        # Unsupported active scoring must fail rather than look league-correct.
        keys = {"fgm", "fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49",
                "fgm_50p", "xpm", "fgmiss", "xpmiss"}
        unsupported = [k for k, v in settings.items()
                       if k.startswith(("fg", "xp")) and k not in keys and v != 0]
        if unsupported:
            raise ValueError(f"unsupported kicker scoring: {sorted(unsupported)}")
        base = float(settings.get("fgm", 0))
        values = tuple(base + float(settings.get(k, 0)) for k in
                       ("fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49"))
        # Nonstandard 60+ and yard-based settings require verified mappings first.
        fifty = base + float(settings.get("fgm_50p", 0))
        return cls((*values, fifty, fifty),
                   float(settings.get("xpm", 0)), float(settings.get("fgmiss", 0)),
                   float(settings.get("xpmiss", 0)))

    def points(self, frame: pd.DataFrame) -> pd.Series:
        counts = frame[[*MADE_BUCKETS, "pat_made"]]
        if (not np.isfinite(counts.to_numpy(float)).all() or (counts < 0).any().any()
                or not np.equal(counts, np.floor(counts)).all().all()):
            raise ValueError("made-kick counts must be present and nonnegative")
        if "fg_made" in frame and not frame[list(MADE_BUCKETS)].sum(axis=1).eq(frame.fg_made).all():
            raise ValueError("FG distance buckets do not conserve made kicks")
        failures = {"fg": 0, "pat": 0}
        for prefix, penalty in (("fg", self.fg_missed), ("pat", self.pat_missed)):
            if not penalty:
                continue
            columns = [f"{prefix}_att", f"{prefix}_made", f"{prefix}_missed",
                       f"{prefix}_blocked"]
            absent = set(columns) - set(frame.columns)
            if absent:
                raise ValueError(f"rebuild raw kicking outcomes; missing {sorted(absent)}")
            counts = frame[columns]
            if (not np.isfinite(counts.to_numpy(float)).all() or (counts < 0).any().any()
                    or not np.equal(counts, np.floor(counts)).all().all()):
                raise ValueError(f"invalid {prefix} outcome counts")
            residual = counts.iloc[:, 0] - counts.iloc[:, 1:].sum(axis=1)
            if residual.ne(0).any():
                raise ValueError(f"unclassified {prefix} outcomes; resolve aborted/safety events")
            failures[prefix] = frame[f"{prefix}_missed"] + frame[f"{prefix}_blocked"]
        return self.components([frame[c] for c in MADE_BUCKETS], frame.pat_made,
                               failures["fg"], failures["pat"])

    def components(self, made_by_bucket, pat_made, fg_failed=0, pat_failed=0):
        """The same arithmetic for observed counts and simulated event counts."""
        if len(made_by_bucket) != 6:
            raise ValueError("six distance bucket counts required")
        return (sum(count * points for count, points in zip(made_by_bucket, self.fg_made))
                + pat_made * self.pat_made + fg_failed * self.fg_missed
                + pat_failed * self.pat_missed)


BASE_SCORING = KickerScoring()


def load_scoring(path=None) -> KickerScoring:
    if path is None:
        return BASE_SCORING
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict):
        raise ValueError("league settings must be a JSON object")
    return KickerScoring.from_sleeper(doc.get("scoring_settings", doc))
