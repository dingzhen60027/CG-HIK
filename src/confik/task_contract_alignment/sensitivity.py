"""Same preselected witnesses at all three contractual scales."""
from .study import Study
from .point_study import run_points
from .contract import SENSITIVITY_METHODS


def main():
    s=Study()
    assert (s.out/'02_point_study/completed.json').exists()
    run_points(s,s.out/'03_contract_sensitivity',SENSITIVITY_METHODS,s.cfg['sensitivity_scales'],True)


if __name__=='__main__':main()
