"""Write exact recorded AlphaFold Server requests; no API/account interaction."""
import argparse
import json
import zipfile
from pathlib import Path
from src.common import ROOT,load_config,require,write_json
from src.structure import one_member

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"results/server_inputs")
    a=p.parse_args();cfg=load_config()
    require(not a.out.exists() or not any(a.out.iterdir()),"Choose an empty output directory")
    a.out.mkdir(parents=True,exist_ok=True)
    requests=[]
    for archive in sorted((ROOT/cfg["af3_archive_directory"]).glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            req=json.loads(z.read(one_member(z,"job_request.json")))
            require(len(req) == 1,"Expected one request per archive")
            write_json(a.out/(archive.stem+"_request.json"),req)
            requests.extend(req)
    require(len(requests) == 12,"Expected twelve real server requests")
    write_json(a.out/"all_12_recorded_requests.json",requests)
    print("Requests saved; manually verify and upload to AlphaFold Server. No jobs were submitted.")
