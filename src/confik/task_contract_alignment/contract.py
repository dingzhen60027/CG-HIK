"""Public norm-ball contract and conservative native mapping, kept separate."""
from dataclasses import asdict, replace
import numpy as np
from ..solvers.verifier import SolutionVerifier, VerifierConfig
from ..continuation_mechanism.observation import representable_interior

STRICT = 1e-5
METHODS = ('trac_strict_5ms', 'trac_position_5ms', 'trac_orientation_5ms',
           'trac_task_5ms', 'trac_strict_20ms', 'trac_task_20ms',
           'dls_strict', 'dls_task')
TRAJECTORY_METHODS = tuple(m for m in METHODS if 'position' not in m and 'orientation' not in m)
SENSITIVITY_METHODS = ('trac_strict_5ms', 'trac_task_5ms', 'dls_task')


def verifier_for(kin, source, scale=1.):
    c = VerifierConfig(**source['verifier'])
    return SolutionVerifier(kin, replace(c, position_tolerance=c.position_tolerance*scale,
                                       orientation_tolerance=c.orientation_tolerance*scale))


def native_mapping(method, verifier):
    c = verifier.config
    b = np.zeros(6)
    if '_task_' in method or '_position_' in method:
        b[:3] = np.nextafter(c.position_tolerance/np.sqrt(3.), 0.)
    if '_task_' in method or '_orientation_' in method:
        b[3:] = np.nextafter(c.orientation_tolerance/np.sqrt(3.), 0.)
    effective = np.maximum(b, STRICT)
    if np.linalg.norm(effective[:3]) > c.position_tolerance or np.linalg.norm(effective[3:]) > c.orientation_tolerance:
        raise ValueError('internal epsilon or component box exceeds public norm ball')
    return dict(bounds=b.tolist(), effective_component_limits=effective.tolist(),
        internal_epsilon=STRICT, public_contract=asdict(c),
        frame='target frame', position='R_target.T @ (p_actual - p_target)',
        rotation='Log(R_target.T @ R_actual), axis-angle vector, not RPY',
        units=['m']*3+['rad']*3,
        formula='aligned: nextafter(epsilon/sqrt(3),0); strict: zero additional bound',
        conservativeness='inscribed component box, NOT equivalent to full norm ball; max(bound, internal epsilon), not their sum')


def task_interval(kin, query, verifier):
    return representable_interior(kin, query, verifier)
