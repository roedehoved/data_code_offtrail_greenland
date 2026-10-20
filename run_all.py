"""
Reproduce every result in the paper and its Supporting Information.

    python run_all.py

Runs, in order:
  code/1_process_data.py   raw GPX, trails and DEM -> data/processed/      (~2 min)
  code/2_main_analysis.py  Tables 1-4, Figures 1-4 -> outputs/            (~5 min)
  code/3_si_robustness.py  SI tables and figures -> outputs/              (~10 min)
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for script in ("1_process_data.py", "2_main_analysis.py", "3_si_robustness.py"):
    t0 = time.time()
    print(f"\n#### {script}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "code", script)], check=True,
                   cwd=os.path.join(HERE, "code"))
    print(f"#### {script} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
print("\nAll outputs are in outputs/tables and outputs/figures.")
