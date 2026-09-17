#!/usr/bin/env python3
"""Isolated Phase 1.5 records; never writes the frozen Phase 1 root."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/process-scan-mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from confik.process_scan.phase15 import repair_references,calibrate_execution,calibrate_b2,prepare_core,run_core
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['references','execution-calibration','b2-calibration','prepare','run'])
    a=p.parse_args()
    {'references':repair_references,'execution-calibration':calibrate_execution,'b2-calibration':calibrate_b2,'prepare':prepare_core,'run':run_core}[a.stage]()
