#!/usr/bin/env python3
"""One Phase 2 entry point; the fixed 24 development scenes only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/process-scan-mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    import argparse
    from confik.process_scan.phase2 import prepare,run,integration
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['integration','prepare','run','report','render','verify'])
    p.add_argument('--slot',action='append');a=p.parse_args()
    if a.stage=='integration':integration(a.slot[0] if a.slot else 'panda_plane_p0_u')
    elif a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.slot)
    else:
        from confik.process_scan import phase2_reporting as reporting
        getattr(reporting,a.stage)()
