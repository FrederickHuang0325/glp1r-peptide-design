"""ESM-2 masked pseudo-log-likelihood: archived verification or fresh inference."""
import csv
import math
import time
from pathlib import Path
from .common import ROOT, require, read_csv, write_csv, write_json, sha256

def validate_scores(cfg, summaries, positions):
    require(len(summaries) == 4 and len(positions) == 120,"ESM dataset must contain 4 sequences x 30 residues")
    require({r["Group"] for r in summaries} == set(cfg["sequences"]),"ESM group names mismatch")
    wt_mean = next(float(r["Mean_masked_log_probability"]) for r in summaries if r["Group"] == "WT")
    for row in summaries:
        group,seq = row["Group"],cfg["sequences"][row["Group"]]
        require(row["Sequence"] == seq and int(row["Length"]) == 30,"ESM sequence mismatch")
        sample = [p for p in positions if p["Group"] == group]
        require(len(sample) == 30 and sorted(int(p["Local_position"]) for p in sample) == list(range(1,31)),"ESM duplicate/missing positions")
        total = sum(float(p["Masked_log_probability"]) for p in sample)
        require(abs(total-float(row["Sum_masked_log_probability"])) < 1e-8,"ESM sum inconsistent")
        require(abs(total/30-float(row["Mean_masked_log_probability"])) < 1e-8,"ESM mean inconsistent")
        require(abs(total/30-wt_mean-float(row["Delta_mean_vs_WT"])) < 1e-8,"ESM delta inconsistent")
        for p in sample:
            i=int(p["Local_position"])-1
            lp=float(p["Masked_log_probability"])
            require(int(p["GLP1_position"]) == i+7 and p["Residue"] == seq[i] and p["WT_residue"] == cfg["sequences"]["WT"][i],"ESM residue numbering mismatch")
            require(math.isfinite(lp) and lp <= 0 and abs(math.exp(lp)-float(p["Masked_probability"])) < 1e-7,"Invalid ESM probability")

def archived(cfg,outdir):
    source = ROOT/cfg["esm_archive_directory"]
    summaries,positions=read_csv(source/"ESM_summary.csv"),read_csv(source/"ESM_per_position.csv")
    validate_scores(cfg,summaries,positions)
    write_csv(outdir/"ESM_summary.csv",summaries)
    write_csv(outdir/"ESM_per_position.csv",positions)
    write_json(outdir/"esm_mode.json",{"mode":"archived_scores_verified_not_fresh_inference",
        "input_sha256":{n:sha256(source/n) for n in ["ESM_summary.csv","ESM_per_position.csv"]}})
    return summaries

def infer(cfg,outdir,checkpoint,device="cuda"):
    import torch
    import esm
    started=time.perf_counter()
    checkpoint=Path(checkpoint).resolve()
    require(checkpoint.is_file() and checkpoint.stem == "esm2_t33_650M_UR50D","Expected official ESM-2 650M checkpoint")
    # The weights_only=False legacy loader is restricted to hash-verified weights.
    require(sha256(checkpoint) == cfg["esm_checkpoint_sha256"],"Checkpoint hash differs; do not unpickle untrusted weights")
    regression=checkpoint.with_name(checkpoint.stem+"-contact-regression.pt")
    require(regression.is_file() and sha256(regression) == cfg["esm_regression_sha256"],"Regression checkpoint missing/hash mismatch")
    require(device != "cuda" or torch.cuda.is_available(),"CUDA unavailable; use --device cpu explicitly")
    torch.manual_seed(17)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(str(checkpoint),map_location="cpu",weights_only=False)
    regression_data=torch.load(str(regression),map_location="cpu",weights_only=False)
    model,alphabet=esm.pretrained.load_model_and_alphabet_core(checkpoint.stem,data,regression_data)
    del data,regression_data
    model=model.float().eval().to(device)
    convert=alphabet.get_batch_converter()
    rows,positions=[],[]
    with torch.inference_mode():
        for group,seq in cfg["sequences"].items():
            _,_,tokens=convert([(group,seq)]);tokens=tokens.to(device)
            lps,llrs=[],[]
            for i,aa in enumerate(seq):
                masked=tokens.clone();masked[0,i+1]=alphabet.mask_idx
                output=model(masked,return_contacts=False)
                logs=torch.log_softmax(output["logits"][0,i+1].float(),dim=-1)
                lp=logs[alphabet.get_idx(aa)].item();lps.append(lp)
                wt=cfg["sequences"]["WT"][i]
                llr=lp-logs[alphabet.get_idx(wt)].item() if aa != wt else ""
                if llr != "": llrs.append(llr)
                positions.append({"Group":group,"Local_position":i+1,"GLP1_position":i+7,"WT_residue":wt,"Residue":aa,
                    "Masked_log_probability":lp,"Masked_probability":math.exp(lp),"Mutation_LLR_in_candidate_context":llr})
                del output,logs,masked
            rows.append({"Group":group,"Sequence":seq,"Length":30,"Mutation_count":sum(a!=b for a,b in zip(seq,cfg["sequences"]["WT"])),
                "Sum_masked_log_probability":sum(lps),"Mean_masked_log_probability":sum(lps)/30,"Delta_mean_vs_WT":0.,
                "Mutation_LLR_sum_in_candidate_context":sum(llrs) if llrs else "","Origin":cfg["origins"][group]})
            print(f"ESM completed {group}",flush=True)
    baseline=rows[0]["Mean_masked_log_probability"]
    for row in rows:row["Delta_mean_vs_WT"]=row["Mean_masked_log_probability"]-baseline
    validate_scores(cfg,rows,positions)
    write_csv(outdir/"ESM_summary.csv",rows);write_csv(outdir/"ESM_per_position.csv",positions)
    write_json(outdir/"esm_mode.json",{"mode":"fresh_ESM_inference","seconds":time.perf_counter()-started,"seed":17,
        "device":device,"torch":torch.__version__,"checkpoint_sha256":cfg["esm_checkpoint_sha256"],
        "method":"30 separate masked passes per peptide; float32; full-vocabulary log-softmax; no amidation token"})
    return rows
