"""Small standard-library I/O helpers. No implicit downloads or installation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AA = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(), "ARNDCQEGHILKMFPSTWYV"))
HYDRO = dict(zip("ARNDCQEGHILKMFPSTWYV", [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-.8,-.7,-.9,-1.3,4.2]))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    require(bool(rows), "Refusing to write an empty results table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_fasta(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(">"):
            records.append([line[1:], ""])
        elif line.strip():
            require(bool(records), "FASTA sequence without header")
            records[-1][1] += line.strip()
    return records

def load_config(path=None):
    return read_json(path or ROOT / "config.json")
