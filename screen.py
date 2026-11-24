"""Structural screening entry; accepts the archived server data via config."""
import argparse
from pathlib import Path
from src.common import ROOT,load_config,require
from src.structure import analyze

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"config.json")
    p.add_argument("--out",type=Path,default=ROOT/"results/structure_new")
    a=p.parse_args()
    require(not a.out.exists() or not any(a.out.iterdir()),"Choose an empty output directory")
    analyze(load_config(a.config),a.out)
