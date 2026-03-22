from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class DataLogger:
    output_dir: Path
    rows: List[dict] = field(default_factory=list)

    def log(self, row: dict):
        self.rows.append(row)

    def save_csv(self, filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        pd.DataFrame(self.rows).to_csv(path, index=False)
        return path
