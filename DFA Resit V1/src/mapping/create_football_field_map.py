from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from mapping.floor_map import FloorMap


if __name__ == "__main__":
    project_root = ROOT.parent
    cfg = yaml.safe_load((project_root / "config" / "system_config.yaml").read_text())
    mcfg = cfg["map"]
    fmap = FloorMap.create_football_field(
        width_m=mcfg["width_m"],
        height_m=mcfg["height_m"],
        scale_m_per_cell=mcfg["scale_m_per_cell"],
        sigma_cells=mcfg["sigma_cells"],
        edge_decay=mcfg["edge_decay"],
        center_boost=mcfg["center_boost"],
    )
    out = project_root / mcfg["file"]
    fmap.save(out)
    print(f"Saved map to {out}")
