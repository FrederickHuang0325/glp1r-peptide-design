"""Recalculate AF3 structural metrics from recorded outputs; no server submission.

The CIF reader deliberately supports the AF3 atom_site table used in this data,
not arbitrary mmCIF. Chain and residue identities are checked before analysis.
"""
import json
import re
import shlex
import zipfile
from collections import Counter
import numpy as np
from .common import AA, ROOT, require, sha256, write_csv, write_json

def pdb_atoms(path):
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM  ") or line[21] not in "RP" or line[16] not in " A":
            continue
        if float(line[54:60]) <= 0:
            continue
        require(line[26] == " ", "Reference insertion codes not supported")
        atoms.append({"chain": line[21], "res": int(line[22:26]), "rn": line[17:20], "name": line[12:16].strip(),
            "el": line[76:78].strip(), "xyz": np.array([float(line[k:k+8]) for k in [30,38,46]], dtype=np.float32),
            "occ": float(line[54:60]), "b": float(line[60:66])})
    return atoms

def cif_atoms(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "loop_" and i+1 < len(lines) and lines[i+1].startswith("_atom_site."):
            cols, atoms, j = [], [], i+1
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols.append(lines[j].strip().split(".",1)[1]); j += 1
            while j < len(lines) and not lines[j].startswith(("#","loop_","_")):
                values = shlex.split(lines[j]); j += 1
                if not values:
                    continue
                require(len(values) == len(cols), "Unsupported multiline AF3 atom_site row")
                d = dict(zip(cols, values))
                if d.get("label_alt_id", ".") not in (".","?","A") or float(d["occupancy"]) <= 0:
                    continue
                require(d["label_comp_id"] in AA, "Nonstandard residue requires a new parser/mapping")
                atoms.append({"chain": d["label_asym_id"], "res": int(d["label_seq_id"]), "rn": d["label_comp_id"],
                    "name": d["label_atom_id"], "el": d["type_symbol"], "occ": float(d["occupancy"]),
                    "b": float(d["B_iso_or_equiv"]),
                    "xyz": np.array([float(d[k]) for k in ["Cartn_x","Cartn_y","Cartn_z"]], dtype=np.float32)})
            require(bool(atoms), "Empty AF3 atom table")
            return atoms
    raise ValueError("No AF3 atom_site table")

def ca(atoms, chain):
    return {a["res"]:a for a in atoms if a["chain"] == chain and a["name"] == "CA"}

def sequence(atoms, chain):
    residues = ca(atoms, chain)
    return "".join(AA[residues[k]["rn"]] for k in sorted(residues))

def contacts(atoms, receptor, peptide, allowed_receptor=None, cutoff=5.):
    r = [a for a in atoms if a["chain"] == receptor and a["el"] not in ("H","D") and
         (allowed_receptor is None or a["res"] in allowed_receptor)]
    p = [a for a in atoms if a["chain"] == peptide and a["el"] not in ("H","D")]
    require(bool(r) and bool(p), "Empty contact atom selection")
    # Float64 distances avoid moving pairs across the cutoff through float32 rounding.
    x = np.asarray([a["xyz"] for a in r], dtype=float)
    y = np.asarray([a["xyz"] for a in p], dtype=float)
    d2 = np.sum((x[:,None,:]-y[None,:,:])**2, axis=2)
    ii,jj = np.where(d2 <= cutoff**2)
    return {(r[i]["res"],p[j]["res"]) for i,j in zip(ii,jj)}

def kabsch(moving, fixed):
    """Fit receptor only. Apply this SAME transform to the peptide."""
    x, y = np.asarray(moving,dtype=float), np.asarray(fixed,dtype=float)
    xm,ym = x.mean(axis=0),y.mean(axis=0)
    u,_,vt = np.linalg.svd((x-xm).T @ (y-ym))
    correction = np.eye(3); correction[-1,-1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    return rotation, ym-xm @ rotation

def rmsd(x,y):
    return float(np.sqrt(np.mean(np.sum((np.asarray(x)-np.asarray(y))**2,axis=1))))

def one_member(z, suffix):
    matches = [n for n in z.namelist() if n.endswith(suffix)]
    require(len(matches) == 1, f"Expected one {suffix}; found {len(matches)}")
    return matches[0]

def analyze(cfg, outdir):
    ref = pdb_atoms(ROOT / cfg["reference_pdb"])
    rca,pca = ca(ref,"R"),ca(ref,"P")
    rec = cfg["receptor_sequence"]
    offset = cfg["receptor_auth_offset"]
    # The offset is an explicitly documented, sequence-validated mapping for R440.
    for number, atom in rca.items():
        require(0 <= number-offset-1 < len(rec) and AA[atom["rn"]] == rec[number-offset-1], "Reference receptor mapping mismatch")
    require(sequence(ref,"P") == cfg["sequences"]["WT"], "Reference peptide differs from WT")
    require(sorted(pca) == list(range(7,37)), "Reference peptide CA incomplete")
    anchors = sorted(n for n in rca if cfg["anchor_auth_range"][0] <= n <= cfg["anchor_auth_range"][1])
    require(len(anchors) == 191 and len(rca) == 384, "Reference anchor/modelled receptor differs")
    ref_contacts = contacts(ref,"R","P",cutoff=cfg["contact_cutoff_A"])
    require(len(ref_contacts) == 109, "Unexpected reference contact count")
    fixed_interface = sorted({r for r,p in ref_contacts})
    require(len(fixed_interface) == 55, "Unexpected fixed interface size")
    outdir.mkdir(parents=True,exist_ok=True)
    structure_dir = outdir / "structures"; structure_dir.mkdir(exist_ok=True)
    rows, manifest, contact_sets = [], [], {}
    seen = set()
    for archive in sorted((ROOT/cfg["af3_archive_directory"]).glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            require(z.testzip() is None, f"Broken archive: {archive.name}")
            reqs = json.loads(z.read(one_member(z,"job_request.json")))
            require(len(reqs) == 1, "Expected one recorded server job")
            req = reqs[0]
            require(len(req["sequences"]) == 2, "Unexpected entities")
            entities = [e["proteinChain"] for e in req["sequences"]]
            receptor_entries = [e for e in entities if e["sequence"] == rec]
            require(len(receptor_entries) == 1, "Unexpected receptor sequence")
            peptide_entries = [e for e in entities if len(e["sequence"]) == 30]
            require(len(peptide_entries) == 1, "Unexpected peptide entity")
            pep = peptide_entries[0]["sequence"]
            group = next((g for g,s in cfg["sequences"].items() if s == pep),None)
            require(group is not None, "Unknown AF3 peptide")
            require(all(e["count"] == 1 and e.get("useStructureTemplate") is False and not e.get("modifications") and not e.get("glycans") for e in entities), "Server input settings differ")
            require(len(req["modelSeeds"]) == 1, "One seed per archive required")
            seed = int(req["modelSeeds"][0])
            require(seed in cfg["af3_seeds"] and (group,seed) not in seen, "Duplicate or unexpected job/seed")
            seen.add((group,seed))
            model_ids = sorted(int(re.search(r"_model_(\d+)\.cif$",n).group(1)) for n in z.namelist() if re.search(r"_model_(\d+)\.cif$",n))
            require(model_ids == list(range(5)), "Expected five AF3 models")
            manifest.append({"group":group,"seed":seed,"archive":archive.name,"sha256":sha256(archive),
                "job_name":req["name"],"request_format_version":req.get("version"),"model_version":"not disclosed in recorded outputs",
                "use_templates":False,"amidation_encoded":False,"models":5})
            for model_id in model_ids:
                cname = one_member(z,f"_model_{model_id}.cif")
                text = z.read(cname).decode()
                atoms = cif_atoms(text)
                seqs = {c:sequence(atoms,c) for c in sorted({a["chain"] for a in atoms})}
                require(sorted(map(len,seqs.values())) == [30,440], "AF3 chain lengths changed")
                cr = next(c for c,s in seqs.items() if s == rec)
                cp = next(c for c,s in seqs.items() if s == pep)
                require(sorted(ca(atoms,cr)) == list(range(1,441)) and sorted(ca(atoms,cp)) == list(range(1,31)), "Noncontiguous predicted chain")
                full = json.loads(z.read(one_member(z,f"_full_data_{model_id}.json")))
                conf = json.loads(z.read(one_member(z,f"_summary_confidences_{model_id}.json")))
                tc = np.array(full["token_chain_ids"])
                tr = np.array(full["token_res_ids"])
                require(len(tc) == 470 and np.array_equal(tr[tc == cr],np.arange(1,441)) and np.array_equal(tr[tc == cp],np.arange(1,31)), "PAE token identity/order mismatch")
                pae = np.asarray(full["pae"],dtype=float)
                require(pae.shape == (470,470) and np.isfinite(pae).all(), "Invalid PAE")
                fi = np.where((tc == cr) & np.isin(tr+offset,fixed_interface))[0]
                pi = np.where(tc == cp)[0]
                require(len(fi) == 55, "PAE interface mapping failed")
                forward = float(pae[np.ix_(fi,pi)].mean())
                reverse = float(pae[np.ix_(pi,fi)].mean())
                predicted_r,predicted_p = ca(atoms,cr),ca(atoms,cp)
                x = [predicted_r[n-offset]["xyz"] for n in anchors]
                y = [rca[n]["xyz"] for n in anchors]
                rotation,translation = kabsch(x,y)
                pfit = np.array([predicted_p[i]["xyz"] @ rotation+translation for i in range(1,31)])
                pref = np.array([pca[i]["xyz"] for i in range(7,37)])
                plddt = [predicted_p[i]["b"] for i in range(1,31)]
                chain_order = list(dict.fromkeys(map(str,tc)))
                ir,ip = chain_order.index(cr),chain_order.index(cp)
                pair = conf.get("chain_pair_iptm")
                pair_f = float(pair[ir][ip]) if pair is not None else None
                pair_r = float(pair[ip][ir]) if pair is not None else None
                # Convert residue numbers ONLY after validating chain sequences/tokens.
                for atom in atoms:
                    atom["res"] += offset if atom["chain"] == cr else 6
                cur_contacts = contacts(atoms,cr,cp,set(rca),cfg["contact_cutoff_A"])
                uid = f"{group}_seed{seed}_model{model_id}"
                contact_sets[uid] = cur_contacts
                kept = len(cur_contacts & ref_contacts)
                structure_file = structure_dir / f"{uid}.cif"
                structure_file.write_text(text,encoding="utf-8")
                rows.append({"group":group,"seed":seed,"model_index":model_id,"model_id":uid,"aligned_CA_count":len(anchors),
                    "receptor_fit_RMSD_A":rmsd(np.asarray(x) @ rotation+translation,y),
                    "peptide_RMSD_A":rmsd(pfit,pref),"peptide_Nterm_RMSD_A":rmsd(pfit[:9],pref[:9]),
                    "reference_contact_count":len(ref_contacts),"reference_contacts_retained":kept,
                    "contact_retention_pct":100*kept/len(ref_contacts),"new_contacts":len(cur_contacts-ref_contacts),
                    "Q23_contacts":sum(p == 23 for r,p in cur_contacts),"A30_contacts":sum(p == 30 for r,p in cur_contacts),
                    "fixed_interface_PAE_R_to_P_A":forward,"fixed_interface_PAE_P_to_R_A":reverse,
                    "fixed_interface_PAE_A":(forward+reverse)/2,"peptide_CA_pLDDT":float(np.mean(plddt)),
                    "peptide_Nterm_CA_pLDDT":float(np.mean(plddt[:9])),"pair_ipTM_R_to_P":pair_f if pair_f is not None else "NA",
                    "pair_ipTM_P_to_R":pair_r if pair_r is not None else "NA","structure_file":structure_file.relative_to(outdir).as_posix()})
                print(f"Analyzed {uid}",flush=True)
    expected = {(g,s) for g in cfg["sequences"] for s in cfg["af3_seeds"]}
    require(seen == expected and len(rows) == 60, "Incomplete four-group/three-seed dataset")
    write_csv(outdir / "all_models.csv",rows)
    numeric = ["peptide_RMSD_A","peptide_Nterm_RMSD_A","contact_retention_pct","fixed_interface_PAE_A","peptide_CA_pLDDT"]
    summaries,seeds = [],[]
    for group in cfg["sequences"]:
        subset = [r for r in rows if r["group"] == group]
        record = {"group":group,"n_models":len(subset)}
        for metric in numeric:
            values = [r[metric] for r in subset]
            record[metric+"_median"] = float(np.median(values))
            record[metric+"_min"] = float(np.min(values))
            record[metric+"_max"] = float(np.max(values))
        # Representative is nearest the group median, not the best-scoring model.
        selected = min(subset,key=lambda r:(abs(r["peptide_RMSD_A"]-record["peptide_RMSD_A_median"]),r["seed"],r["model_index"]))
        record["representative_structure"] = selected["structure_file"]
        summaries.append(record)
        for seed in cfg["af3_seeds"]:
            sample = [r for r in subset if r["seed"] == seed]
            seeds.append({"group":group,"seed":seed,"n_models":len(sample),**{k+"_median":float(np.median([r[k] for r in sample])) for k in numeric}})
    write_csv(outdir/"group_summary.csv",summaries)
    write_csv(outdir/"seed_summary.csv",seeds)
    frequencies = []
    union = set(ref_contacts).union(*contact_sets.values())
    for r,p in sorted(union):
        frequencies.append({"receptor_auth_position":r,"peptide_GLP1_position":p,"in_reference":(r,p) in ref_contacts,
            **{g+"_pct":100*sum((r,p) in cs for uid,cs in contact_sets.items() if uid.startswith(g+"_seed"))/15 for g in cfg["sequences"]}})
    write_csv(outdir/"contact_frequency.csv",frequencies)
    write_json(outdir/"af3_manifest.json",manifest)
    return rows,summaries,seeds
