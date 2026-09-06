"""Export completed supplementary measurements; never execute an IK method."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/revision_compute_allocation'
PAPER=ROOT/'paper'
DEST=PAPER/'revision_generated'
SOURCE=PAPER/'revision_source_data'
FIG=PAPER/'revision_figures'
NAMES={'always_hard':'Always-hard','geometry_threshold':'Geometry rule','reject_only_hard':'Reject-only + hard',
       'routing_only':'Routing-only','p50_selection':'P50-selection','full_cghik':r'\method'}
ROBOTS={'panda':'Panda','ur5e':'UR5e'}
FAMILIES={'smooth':'Smooth','near_singular':'Near-singular','joint_limit_return':'Joint-limit return','high_curvature':'High-curvature'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):return list(csv.DictReader((OUT/name).open()))
def number(x,n=3):return f'{float(x):.{n}f}' if x not in ['',None] else '--'
def percent(x,n=2):return number(100*float(x),n)
def ratio_ci(r,metric='latency_ratio'):
    return f"{number(r[metric],4)} [{number(r[metric+'_ci_low'],4)}, {number(r[metric+'_ci_high'],4)}]"
def write(name,rr):
    with (DEST/name).open('x') as f:
        f.write('% Generated only from completed supplementary records.\n')
        for row in rr:f.write(' & '.join(str(x) for x in row)+r' \\'+'\n')

def main():
    if DEST.exists() or SOURCE.exists() or FIG.exists():raise FileExistsError('Supplementary paper exports already exist')
    DEST.mkdir();SOURCE.mkdir();FIG.mkdir()
    files=['reports/point_main_table.csv','reports/point_paired_comparisons.csv','reports/oracle_crossfit_summary.csv',
           'reports/trajectory_main_table.csv','reports/trajectory_paired_comparisons.csv','reports/feasible_trajectory_savings_decomposition.csv',
           'reports/branch_divergence_diagnostics.csv','reports/external_verifier_rejections.csv',
           '01_existing_result_decomposition/trajectory_savings_decomposition.csv','01_existing_result_decomposition/same_population_routing_comparison.csv']
    inputs={}
    for name in files:
        path=OUT/name;inputs[name]=sha(path);shutil.copy2(path,SOURCE/path.name)
    figures=json.loads((OUT/'reports/analysis_delivery_manifest.json').read_text())['figures']
    for name,value in figures.items():
        path=OUT/'reports/publication_figures'/name
        assert sha(path)==value
        if path.suffix=='.pdf':shutil.copy2(path,FIG/name)
    points=rows(files[0]);paired=rows(files[1]);oracle=rows(files[2]);traj=rows(files[3]);tpairs=rows(files[4])
    point=lambda robot,method:next(r for r in points if r['robot']==robot and r['subset']=='feasible' and r['method']==method)
    trajectory=lambda robot,method,family='all':next(r for r in traj if r['robot']==robot and r['family']==family and r['method']==method)
    write('point_internal_rows.tex',[
        [ROBOTS[robot],NAMES[method],percent(r['verified_success_rate']),number(r['mean_fev']),number(r['p50_ms']),number(r['p95_ms']),number(r['p99_ms']),percent(r['accepted_within_20ms_rate'])]
        for robot in ROBOTS for method in NAMES for r in [point(robot,method)]])
    write('point_external_rows.tex',[
        [ROBOTS[robot],budget,percent(r['verified_success_rate'],3),number(float(r['total_latency_ns'])/float(r['calls'])/1e6),number(r['p50_ms']),number(r['p95_ms']),number(r['p99_ms']),percent(r['accepted_within_20ms_rate'],3)]
        for robot in ROBOTS for budget in [5,20,100,400] for r in [point(robot,f'trac_ik_{budget}ms')]])
    write('point_paired_rows.tex',[
        [ROBOTS[r['robot']],NAMES[r['method']]+' / '+NAMES[r['reference']],ratio_ci(r),ratio_ci(r,'fev_ratio')]
        for r in paired if r['subset']=='feasible' and (r['method']=='full_cghik' and r['reference'] in ['always_hard','geometry_threshold','reject_only_hard','p50_selection'] or r['method']=='routing_only' and r['reference']=='always_hard')])
    write('p95_ablation_rows.tex',[
        [ROBOTS[r['robot']],ratio_ci(r,'p95_ratio'),ratio_ci(r,'p99_ratio')]
        for r in paired if r['subset']=='feasible' and r['method']=='full_cghik' and r['reference']=='p50_selection'])
    write('trajectory_rows.tex',[
        [ROBOTS[robot],NAMES.get(method,'TRAC-IK (5 ms)'),r['completed']+'/40',r['deadline_complete']+'/40',number(float(r['total_latency_ns'])/1e9),number(r['mean_fev']),number(r['p50_ms']),number(r['p95_ms']),number(r['p99_ms'])]
        for robot in ROBOTS for method in ['always_hard','routing_only','full_cghik','trac_ik_5ms'] for r in [trajectory(robot,method)]])
    write('trajectory_paired_rows.tex',[
        [ROBOTS[r['robot']],NAMES[r['method']]+' / '+NAMES[r['reference']],ratio_ci(r),ratio_ci(r,'fev_ratio'),
         f"{percent(r['completion_difference'],1)} [{percent(r['completion_difference_ci_low'],1)}, {percent(r['completion_difference_ci_high'],1)}]"]
        for r in tpairs if r['family']=='all' and r['reference'] in ['always_hard','routing_only']])
    write('family_rows.tex',[
        [ROBOTS[robot],FAMILIES[family],h['completed']+' / '+g['completed'],number(float(h['total_latency_ns'])/1e9)+' / '+number(float(g['total_latency_ns'])/1e9),
         number(h['mean_fev'])+' / '+number(g['mean_fev']),t['completed']+' / '+number(float(t['total_latency_ns'])/1e9)]
        for robot in ROBOTS for family in FAMILIES for h,g,t in [(trajectory(robot,'always_hard',family),trajectory(robot,'full_cghik',family),trajectory(robot,'trac_ik_5ms',family))]])
    write('oracle_rows.tex',[
        [ROBOTS[r['robot']],r['preference_stability_query_count'],percent(r['preference_stability']),percent(r['entry_mean_range_at_most_015ms_rate']),
         f"{number(r['crossfit_mean_latency_ratio'],4)} [{number(r['crossfit_mean_latency_ratio_ci_low'],4)}, {number(r['crossfit_mean_latency_ratio_ci_high'],4)}]"] for r in oracle])
    labels={'both_complete':'Both complete','only_cghik_complete':'Only CG-HIK','only_hard_complete':'Only hard','neither_complete':'Neither complete'}
    write('old_decomposition_rows.tex',[
        [ROBOTS[r['robot']],labels[r['group']],r['trajectory_count'],number(float(r['hard_total_latency_ns'])/1e9),number(float(r['cghik_total_latency_ns'])/1e9),
         r['hard_total_fev']+' / '+r['cghik_total_fev'],percent(r['fraction_of_all_saved_latency'])]
        for r in rows(files[8])])
    names={'fixed_easy':'Fixed easy','fixed_medium':'Fixed medium','fixed_hard':'Fixed hard','frozen_cghik':r'\method','empirical_oracle':'Empirical oracle'}
    write('same_population_rows.tex',[
        [ROBOTS[r['robot']],names.get(r['method'],r['method'].replace('_',' ')),r['query_count'],number(r['numerical_verifier_mean_ms']),number(r['measured_label_pipeline_mean_ms']),number(r['mean_noisy_query_empirical_p95_ms']),'--']
        for r in rows(files[9])])
    manifest={'role':'supplementary evaluation, exported after findings report and before manuscript rewrite',
              'findings_sha256':sha(ROOT/'docs/REVISION_EXPERIMENT_FINDINGS.md'),'inputs':inputs,
              'figures':{p.name:sha(p) for p in FIG.iterdir()},'tables':{p.name:sha(p) for p in DEST.iterdir()},
              'old_source_data_unchanged':True,'no_solver_calls':True}
    with (DEST/'evidence_manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')

if __name__=='__main__':main()
