"""Main submission entry: source checks -> design audit -> AF3 -> ESM -> results.csv."""
import argparse
import importlib.metadata
import platform
import sys
import time
from pathlib import Path
from src.common import ROOT, HYDRO, load_config, require, read_json, sha256, write_csv, write_json
from src.design import validate_archive,enumerate_space
from src.structure import analyze
from src.esm_score import archived,infer
from src import __version__

def check_inputs():
    manifest=read_json(ROOT/"data/input_manifest.json")
    for row in manifest:
        path=ROOT/row["path"]
        require(path.is_file() and sha256(path) == row["sha256"],f'Input missing/changed: {row["path"]}')
    return len(manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--out",type=Path,default=ROOT/"results/reproduced")
    parser.add_argument("--esm-mode",choices=["archive","infer"],default="archive")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    parser.add_argument("--track",help="Exact registered competition track name")
    args=parser.parse_args()
    started=time.perf_counter()
    cfg=load_config(args.config)
    out=args.out.resolve()
    # Never overwrite source data or a previous successful run.
    require(not out.exists() or not any(out.iterdir()),"Output folder is not empty; choose a new --out directory")
    count=check_inputs()
    out.mkdir(parents=True,exist_ok=True)
    validate_archive(cfg,out);enumerate_space(cfg,out)
    models,groups,seeds=analyze(cfg,out)
    if args.esm_mode == "infer":
        require(args.checkpoint is not None,"--checkpoint is required for fresh ESM inference")
        esm=infer(cfg,out,args.checkpoint,args.device)
    else:
        esm=archived(cfg,out)
    em={r["Group"]:r for r in esm};gm={r["group"]:r for r in groups}
    ordering=["A30Q","Q23A","Q23A_A30Q","WT"]
    candidates=[]
    for group in ordering:
        seq=cfg["sequences"][group];structure=gm[group]
        tier=0 if group == "WT" else 2 if group == "Q23A_A30Q" else 1
        candidates.append({"candidate_id":cfg["candidate_ids"][group],"competition_track":args.track or cfg["competition_track"],
            "group":group,"sequence":seq,"length":len(seq),"intended_C_terminus":"amide",
            "structure_C_terminus":"unamidatedApprox; no explicit amide modification in AF3 input",
            "origin":cfg["origins"][group],"role":"mother_control" if group == "WT" else "candidate",
            "validation_priority_tier":tier,"priority_note":"A30Q and Q23A are co-priority; not predicted efficacy ranking" if tier == 1 else "combination_control" if tier == 2 else "reference_control",
            "mutation_count":sum(a!=b for a,b in zip(seq,cfg["sequences"]["WT"])),"GRAVY":round(sum(HYDRO[a] for a in seq)/30,6),
            "AF3_models":15,"AF3_seeds":"17;29;43",
            "peptide_RMSD_median_A":round(structure["peptide_RMSD_A_median"],6),
            "peptide_RMSD_min_A":round(structure["peptide_RMSD_A_min"],6),"peptide_RMSD_max_A":round(structure["peptide_RMSD_A_max"],6),
            "Nterm_RMSD_median_A":round(structure["peptide_Nterm_RMSD_A_median"],6),
            "contact_retention_median_pct":round(structure["contact_retention_pct_median"],6),
            "fixed_interface_PAE_median_A":round(structure["fixed_interface_PAE_A_median"],6),
            "peptide_CA_pLDDT_median":round(structure["peptide_CA_pLDDT_median"],6),
            "ESM_mean_masked_log_probability":round(float(em[group]["Mean_masked_log_probability"]),6),
            "ESM_delta_mean_vs_WT":round(float(em[group]["Delta_mean_vs_WT"]),6),
            "affinity_prediction":"NA","cAMP_activity_prediction":"NA","beta_arrestin_bias_prediction":"NA",
            "model_versions":"ColabDesign install target v1.1.1/ProteinMPNN v_48_020; AlphaFold Server model version undisclosed; ESM-2 esm2_t33_650M_UR50D",
            "pipeline_version":__version__,"structure_file":structure["representative_structure"],
            "notes":"Post-hoc exploratory screening; no training/fine-tuning; no pharmacological inference from structure/ESM"})
    write_csv(out/"results.csv",candidates)
    versions={}
    for p in ["numpy","torch","fair-esm"]:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]="not installed; not required for archive mode" 
    log={"pipeline_version":__version__,"time_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "seconds":time.perf_counter()-started,"input_files_verified":count,"python":platform.python_version(),"platform":platform.platform(),
        "packages":versions,"command":sys.argv,"esm_mode":args.esm_mode,"AI_design_mode":"archive_audit_plus_constraint_enumeration",
        "AF3_mode":"recorded_output_reanalysis_not_new_server_inference","config_sha256":sha256(args.config),
        "competition_track_confirmed":(args.track or cfg["competition_track"]) != "UNSPECIFIED",
        "random_seeds":{"AF3":[17,29,43],"original_ProteinMPNN":None,"ESM":17},
        "limitations":["15 AF3 samples are not biological replicates","AF3 server/MPNN historical seeds and software revisions limit exact regeneration",
                       "Archived ESM scores are verified but not recomputed unless --esm-mode infer","No wet-lab simulation data included"]}
    write_json(out/"run_log.json",log)
    print(f"SUCCESS: {out/'results.csv'}",flush=True)
    if not log["competition_track_confirmed"]:print("Before submission supply the registered track via --track; current value is UNSPECIFIED.")

if __name__ == "__main__":main()
