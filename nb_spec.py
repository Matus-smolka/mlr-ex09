"""Print the notebook cells of one task, verbatim, into a UTF-8 file.

The tasks in ex09_student.ipynb are a markdown cell with the title, a markdown cell with the
specification table to fill in, a data cell, a markdown cell saying what to show, and then the empty
cells for our own code. Reading them straight out of the .ipynb JSON is the only way to be sure the
URLs, the split indices and the reference numbers quoted in the report are the ones the notebook
actually gives (they have been mis-quoted before).

The console on this machine is cp1250 and the notebook has non-ASCII in it, so the output goes to a
file rather than to stdout.

Usage:  py -I nb_spec.py <notebook.ipynb> "<title to look for>" <out.txt> [extra cells before/after]
"""
import json
import re
import sys

nb_path, needle, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
pad = int(sys.argv[4]) if len(sys.argv) > 4 else 6

nb = json.loads(open(nb_path, encoding="utf-8").read())
cells = ["".join(c["source"]) for c in nb["cells"]]
hits = [i for i, c in enumerate(cells) if re.search(needle, c, re.I)]
if not hits:
    sys.exit(f"no cell matches {needle!r} in {nb_path}")
start, stop = max(0, hits[0] - 1), min(len(cells), hits[0] + pad)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"# cells {start}..{stop - 1} of {nb_path}, matched {needle!r} at cell {hits[0]}\n\n")
    for i in range(start, stop):
        kind = nb["cells"][i]["cell_type"]
        f.write(f"--- cell {i} [{kind}] {len(cells[i])} chars ---\n{cells[i]}\n\n")
print(f"{len(cells)} cells, match at {hits[0]}, wrote cells {start}..{stop - 1} to {out_path}")
