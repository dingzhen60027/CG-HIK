"""Finish only the already fixed evaluation: aggregation, FK replay and probes.

No retries, changed settings, new paths or result-dependent state selection.
This coordinator may wait for the two committed full-trajectory jobs to finish.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
from .study import ROOT


def call(module,args,cpus):
    subprocess.run(['taskset','-c',cpus,sys.executable,'-m',module,*args],cwd=ROOT,check=True)


def main():
    root=ROOT/'outputs/correction_reserve_ik'
    folders=[str(root/'formal'/robot) for robot in ('panda','ur5e')]
    while not all((Path(folder)/'completed.json').exists() for folder in folders):
        time.sleep(10)
    call('confik.correction_reserve.reporting',['--folders',*folders,'--out',str(root/'formal_reports')],'16,17')
    with ThreadPoolExecutor(max_workers=2) as pool:
        audit=pool.submit(call,'confik.correction_reserve.audit',
            ['--folders',*folders,'--out',str(root/'formal_verification')],'16,17')
        probes=pool.submit(call,'confik.correction_reserve.mechanism',
            ['--folders',*folders,'--out',str(root/'formal_mechanism')],'0,2')
        probes.result();audit.result()
    call('confik.correction_reserve.figures',['--report',str(root/'formal_reports'),
         '--out',str(root/'formal_figures'),'--mechanism',str(root/'formal_mechanism'),'--runs',*folders],'16,17')
    print('Fixed evaluation and read-only command replay complete; no new algorithm launched.',flush=True)


if __name__=='__main__':main()
