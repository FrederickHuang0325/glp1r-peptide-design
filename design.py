"""Design entry: archived AI evidence, exhaustive controls, or optional fresh MPNN."""
import argparse
from pathlib import Path
from src.common import ROOT,load_config,require
from src.design import validate_archive,enumerate_space,rerun_mpnn

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["archive","enumerate","mpnn"],default="archive")
    p.add_argument("--out",type=Path,default=ROOT/"results/design_new")
    p.add_argument("--seed",type=int,default=20261003)
    a=p.parse_args();cfg=load_config()
    require(not a.out.exists() or not any(a.out.iterdir()),"Choose an empty output directory")
    a.out.mkdir(parents=True,exist_ok=True)
    if a.mode == "archive":validate_archive(cfg,a.out)
    elif a.mode == "enumerate":enumerate_space(cfg,a.out)
    else:rerun_mpnn(cfg,a.out,a.seed)
