"""Validate recorded AI samples and expose a constrained design rerun interface."""
import itertools
import time
from collections import Counter
from .common import AA, ROOT, require, read_fasta, write_csv, write_json, sha256

def receptor_coordinate_sequence(pdb):
    residues = {}
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM  ") and line[21] == "R" and line[12:16].strip() == "CA":
            require(line[26] == " ", "Insertion code unsupported in this reference")
            if float(line[54:60]) > 0 and line[16] in " A":
                residues[int(line[22:26])] = AA[line[17:20]]
    return "".join(residues[k] for k in sorted(residues))

def validate_archive(cfg, outdir):
    fasta = ROOT / cfg["recorded_design_fasta"]
    reference = ROOT / cfg["reference_pdb"]
    require(sha256(fasta) == cfg["recorded_design_fasta_sha256"], "Recorded FASTA checksum mismatch")
    require(sha256(reference) == cfg["reference_pdb_sha256"], "Reference PDB checksum mismatch")
    rec = receptor_coordinate_sequence(reference)
    wt = cfg["sequences"]["WT"]
    free = {p-7 for p in cfg["design"]["open_glp1_positions"]}
    counts = Counter()
    samples = []
    for index, (header, sequence) in enumerate(read_fasta(fasta)):
        chains = sequence.split("/")
        require(len(chains) == 2 and len(chains[0]) == 30 and chains[1] == rec, "Unexpected archived sequence/chains")
        pep = chains[0]
        for pos, (a, b) in enumerate(zip(pep, wt)):
            require(a == b if pos not in free else a in "AQ", "Violation of recorded design constraints")
        group = next((g for g, s in cfg["sequences"].items() if s == pep), None)
        require(group is not None, "Unrecognized archived candidate")
        counts[group] += 1
        samples.append({"sample_index": index, "group": group, "peptide_sequence": pep, "receptor_length": len(rec), "original_header": header})
    require(dict(counts) == {"A30Q": 27, "WT": 5}, "Archived sampling counts changed")
    write_csv(outdir / "recorded_design_samples.csv", samples)
    write_json(outdir / "design_audit.json", {
        "mode": "recorded_design_validation_not_new_AI_inference", "samples": len(samples), "counts": dict(counts),
        "fixed_receptor_residues": len(rec), "open_glp1_positions": sorted(p+7 for p in free),
        "mpnn_model": "v_48_020", "colabdesign_install_target": "v1.1.1",
        "original_random_seed": None, "random_seed_note": "Not recorded; do not claim bitwise reproduction of the 32 samples",
    })
    return samples

def enumerate_space(cfg, outdir):
    """Exhaustive two-site controls. This is enumeration, NOT AI sampling."""
    wt = cfg["sequences"]["WT"]
    positions = [p-7 for p in cfg["design"]["open_glp1_positions"]]
    rows = []
    for residues in itertools.product("AQ", repeat=len(positions)):
        seq = list(wt)
        for i, a in zip(positions, residues):
            seq[i] = a
        seq = "".join(seq)
        rows.append({"sequence": seq, "origin": "deterministic_constraint_enumeration_not_AI", "mutation_count": sum(a != b for a, b in zip(seq, wt))})
    write_csv(outdir / "enumerated_design_space.csv", rows)
    return rows

def rerun_mpnn(cfg, outdir, seed):
    """Optional fresh ColabDesign run; must use a separately prepared environment."""
    from colabdesign.mpnn import mk_mpnn_model
    import numpy as np
    started = time.perf_counter()
    model = mk_mpnn_model(model_name="v_48_020", seed=seed)
    model.prep_inputs(pdb_filename=str(ROOT / cfg["reference_pdb"]), chain="P,R", homooligomer=False,
                      fix_pos="P23,P30", inverse=True, rm_aa=cfg["design"]["excluded_amino_acids"], verbose=True)
    require(list(model._lengths) == [30,384], "Parsed chain lengths differ from recorded design")
    free = sorted(set(range(414)) - set(map(int, model._inputs["fix_pos"])))
    require(free == [16,23], "Designable positions differ from intended Q23/A30")
    output = model.sample(num=1, batch=32, temperature=.1, rescore=False)
    outdir.mkdir(parents=True, exist_ok=True)
    fasta = []
    receptor = receptor_coordinate_sequence(ROOT / cfg["reference_pdb"])
    for i, seq in enumerate(output["seq"]):
        seq = str(seq)
        pep, rec = seq.split("/")
        require(rec == receptor, "Fresh ProteinMPNN changed receptor sequence")
        wt = cfg["sequences"]["WT"]
        require(all(a == b if j not in [16,23] else a in "AQ" for j,(a,b) in enumerate(zip(pep,wt))), "Fresh constraints failed")
        fasta.extend([f'>sample:{i}_score:{output["score"][i]:.6f}_seqid:{output["seqid"][i]:.6f}', seq])
    (outdir / "new_mpnn_samples.fasta").write_text("\n".join(fasta)+"\n", encoding="utf-8")
    write_json(outdir / "new_mpnn_run.json", {"seed": seed, "seconds": time.perf_counter()-started,
        "mode": "fresh_AI_sampling_not_the_original_run", "numpy": np.__version__, "free_indices_zero_based": free,
        "original_sampling_seed_unknown": True})
