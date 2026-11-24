"""Execute local notebook Python cells without installing/changing Jupyter.

This checks Python-cell execution, not browser rendering or kernel installation.
Use Jupyter normally to display the notebooks interactively.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def run_one(path):
    nb=json.loads(path.read_text(encoding="utf-8"))
    namespace={"__name__":"__main__"}
    os.chdir(ROOT)
    for i,cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":continue
        source="".join(cell["source"])
        code=compile(source,f"{path.name}:cell{i}","exec")
        exec(code,namespace)
    return {"notebook":path.name,"python_cells_executed":sum(c["cell_type"]=="code" for c in nb["cells"]),
            "validation":"Python-cell execution only; no Jupyter kernel installation",
            "fresh_neural_inference":"optional cells disabled by default"}

if __name__ == "__main__":
    if len(sys.argv)>1:print(json.dumps(run_one(Path(sys.argv[1]).resolve()),ensure_ascii=False))
    else:
        for path in sorted((ROOT/"notebooks").glob("*.ipynb")):
            subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),str(path)],check=True)
