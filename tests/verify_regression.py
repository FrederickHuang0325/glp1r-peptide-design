"""Compare rebuilt metrics to the supplied Colab/ESM tables, not invented values."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read_csv,require

def verify(folder):
    old=read_csv(ROOT/"data/recorded/all_60_metrics.csv")
    new=read_csv(folder/"all_models.csv")
    def name(g):return g.replace("/","_")
    n={(r["group"],int(r["seed"]),int(r["model_index"])):r for r in new}
    require(len(n)==60 and len(old)==60,"Wrong model count")
    fields={"对齐CA数量":"aligned_CA_count","受体拟合RMSD_A":"receptor_fit_RMSD_A","肽整体RMSD_A":"peptide_RMSD_A",
        "肽N端RMSD_A":"peptide_Nterm_RMSD_A","参考接触保留数":"reference_contacts_retained","接触保持率_pct":"contact_retention_pct",
        "新增接触数":"new_contacts","Q23位点接触数":"Q23_contacts","A30位点接触数":"A30_contacts",
        "固定界面PAE_受体至肽_A":"fixed_interface_PAE_R_to_P_A","固定界面PAE_肽至受体_A":"fixed_interface_PAE_P_to_R_A",
        "固定界面PAE_双向平均_A":"fixed_interface_PAE_A","肽平均pLDDT_CA":"peptide_CA_pLDDT","肽N端平均pLDDT_CA":"peptide_Nterm_CA_pLDDT"}
    deviations={k:0. for k in fields.values()}
    for r in old:
        q=n[(name(r["组别"]),int(r["种子"]),int(r["模型编号"]))]
        for a,b in fields.items():deviations[b]=max(deviations[b],abs(float(r[a])-float(q[b])))
    for key,delta in deviations.items():require(delta < 1e-4,f"Regression mismatch {key}: {delta}")
    os=read_csv(ROOT/"data/recorded/seed_summary.csv");ns=read_csv(folder/"seed_summary.csv")
    lookup={(r["group"],int(r["seed"])):r for r in ns}
    seed_delta=0.
    for r in os:
        q=lookup[(name(r["组别"]),int(r["种子"]))]
        for a,b in {"肽整体RMSD中位数":"peptide_RMSD_A_median","接触保持率中位数":"contact_retention_pct_median","固定界面PAE中位数":"fixed_interface_PAE_A_median"}.items():
            seed_delta=max(seed_delta,abs(float(r[a])-float(q[b])))
    require(seed_delta < 1e-4,"Seed median mismatch")
    es=read_csv(ROOT/"data/esm/ESM_summary.csv");new_es={r["Group"]:r for r in read_csv(folder/"ESM_summary.csv")}
    esm_delta=0.
    for r in es:
        for k in ["Mean_masked_log_probability","Delta_mean_vs_WT"]:
            esm_delta=max(esm_delta,abs(float(r[k])-float(new_es[r["Group"]][k])))
    require(esm_delta < 1e-4,"ESM inference/archive mismatch")
    return {"passed":True,"models_compared":60,"seed_groups_compared":12,"absolute_tolerance":1e-4,
        "maximum_deviation_by_metric":deviations,"seed_summary_maximum_deviation":seed_delta,"ESM_maximum_deviation":esm_delta}

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--results",type=Path,default=ROOT/"results/example")
    print(json.dumps(verify(p.parse_args().results),ensure_ascii=False,indent=2))
