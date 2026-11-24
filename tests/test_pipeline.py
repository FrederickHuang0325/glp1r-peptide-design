"""Lightweight, dependency-minimal unit tests; never calls a remote service."""
import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,load_config,read_csv
from src.structure import kabsch,rmsd,pdb_atoms,contacts,ca
from src.esm_score import validate_scores

class PipelineTests(unittest.TestCase):
    def test_kabsch(self):
        x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        y=x@rotation+np.array([4.,5.,6.])
        r,t=kabsch(x,y)
        self.assertLess(rmsd(x@r+t,y),1e-10)
        self.assertAlmostEqual(np.linalg.det(r),1.)
    def test_reference(self):
        cfg=load_config();atoms=pdb_atoms(ROOT/cfg["reference_pdb"])
        self.assertEqual(len(ca(atoms,"R")),384)
        self.assertEqual(len(contacts(atoms,"R","P")),109)
    def test_sequences(self):
        cfg=load_config();wt=cfg["sequences"]["WT"]
        for group,seq in cfg["sequences"].items():
            self.assertEqual(len(seq),30)
            self.assertTrue(all(a == b if i not in [16,23] else a in "AQ" for i,(a,b) in enumerate(zip(seq,wt))))
    def test_ESM_archive(self):
        cfg=load_config();path=ROOT/cfg["esm_archive_directory"]
        validate_scores(cfg,read_csv(path/"ESM_summary.csv"),read_csv(path/"ESM_per_position.csv"))
    def test_ESM_bad_row_rejected(self):
        cfg=load_config();path=ROOT/cfg["esm_archive_directory"]
        summary=read_csv(path/"ESM_summary.csv");pos=read_csv(path/"ESM_per_position.csv")
        pos[0]["Local_position"]="2"
        with self.assertRaises(ValueError):validate_scores(cfg,summary,pos)

if __name__ == "__main__":unittest.main()
