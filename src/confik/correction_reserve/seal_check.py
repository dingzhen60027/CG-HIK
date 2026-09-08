"""Read-only check required before the one fixed fresh evaluation."""
import json
from pathlib import Path
import subprocess
import yaml
from .study import ROOT,sha,code_hashes


def main():
    folder=ROOT/'outputs/correction_reserve_ik/formal_protocol'
    seal=json.loads((folder/'selection_seal.json').read_text())
    assert seal['configuration']==yaml.safe_load((ROOT/'configs/correction_reserve.yaml').read_text())
    assert seal['code_hashes']==code_hashes(),'implementation changed after identity freeze'
    for name,digest in seal['files'].items():assert sha(folder/name)==digest,name
    assert sha(ROOT/'src/confik/correction_reserve/data.py')==seal['generation_source_sha256']
    paths=['src/confik/correction_reserve','configs/correction_reserve.yaml','docs/CRIK_ALGORITHM_PROTOCOL.md']
    for flag in ([],['--cached']):
        diff=subprocess.check_output(['git','diff',*flag,'--name-only','--',*paths],cwd=ROOT,text=True)
        assert not diff,'commit code before evaluation'
    assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*paths],cwd=ROOT,text=True)
    print('Configuration, numerical implementation and fresh identities match the pre-outcome seal.')


if __name__=='__main__':main()
