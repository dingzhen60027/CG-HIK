#!/usr/bin/env python3
"""Generate manuscript numbers/tables from frozen evidence; no IK/model imports."""
import csv,hashlib,json,math,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper'
EVIDENCE=ROOT/'outputs/task_contract_alignment'
COMMIT='089a3fe7ec0456002aa0f3b9661119b99072fad7'
LABELS={'trac_strict_5ms':'TRAC strict 5 ms','trac_position_5ms':'TRAC position 5 ms','trac_orientation_5ms':'TRAC orientation 5 ms','trac_task_5ms':'TRAC aligned 5 ms','trac_strict_20ms':'TRAC strict 20 ms','trac_task_20ms':'TRAC aligned 20 ms','dls_strict':'DLS strict','dls_task':'DLS aligned'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    manifest=load(EVIDENCE/'delivery_manifest.json')
    for p,h in manifest['files'].items():assert digest(ROOT/p)==h,p
    gen=PAPER/'generated';data=PAPER/'source_data';gen.mkdir(exist_ok=True);data.mkdir(exist_ok=True)
    names=['point_main','point_families','point_paired','sensitivity_main','sensitivity_families','sensitivity_paired','trajectory_main','trajectory_families','trajectory_paired','completion_uid_sets','first_failures','dls_excess_iterations','native_mapping_table']
    sources={};tables={}
    for n in names:
        p=EVIDENCE/'05_aggregate'/f'{n}.csv';shutil.copyfile(p,data/f'task_{n}.csv')
        sources[str(p.relative_to(ROOT))]=digest(p)
        jp=p.with_suffix('.json')
        if jp.exists():tables[n]=load(jp);sources[str(jp.relative_to(ROOT))]=digest(jp)
    historic={}
    for n,rel in {
        'allocation_points':'reports/point_main_table.csv',
        'allocation_point_paired':'reports/point_paired_comparisons.csv',
        'allocation_trajectories':'reports/trajectory_main_table.csv',
        'allocation_trajectory_paired':'reports/trajectory_paired_comparisons.csv',
        'allocation_decomposition':'01_existing_result_decomposition/trajectory_savings_decomposition.csv',
        'allocation_same_population':'01_existing_result_decomposition/same_population_routing_comparison.csv',
        'allocation_oracle':'reports/oracle_crossfit_summary.csv'}.items():
        p=ROOT/'outputs/revision_compute_allocation'/rel
        shutil.copyfile(p,data/f'task_{n}.csv');sources[str(p.relative_to(ROOT))]=digest(p)
        historic[n]=list(csv.DictReader(p.open()))
    protocol=load(EVIDENCE/'01_protocol/protocol.json');k=load(EVIDENCE/'05_aggregate/key_findings.json')
    for p in [EVIDENCE/'01_protocol/protocol.json',EVIDENCE/'05_aggregate/key_findings.json',EVIDENCE/'05_aggregate/verification.json']:
        sources[str(p.relative_to(ROOT))]=digest(p)
    nums={}
    def add(n,v,dec=3):nums[n]=v if isinstance(v,str) else f'{v:.{dec}f}'
    def ci(n,v,scale=1):add(n,f'{v[0]*scale:.3f} [{v[1]*scale:.3f}, {v[2]*scale:.3f}]')
    c=protocol['public_contract']
    add('contract.ep_mm',c['position_tolerance']*1000,0);add('contract.er_deg',math.degrees(c['orientation_tolerance']),1)
    add('contract.dt_ms',protocol['config']['dt']*1000,0);add('contract.ev',c['velocity_tolerance'],4)
    nominal={r['nominal'] for r in protocol['witness_validation'].values()}
    sensitivity={r['sensitivity'] for r in protocol['witness_validation'].values()}
    inexecutable={r['units'] for r in tables['point_main'] if not r['witness_feasible']}
    assert len(nominal)==len(sensitivity)==len(inexecutable)==1
    for n,v in [('point',next(iter(nominal))),('inexecutable',next(iter(inexecutable))),('sensitivity',next(iter(sensitivity))),('trajectory',len(load(EVIDENCE/'01_protocol/ur5e_trajectory_identities.json'))),('frames',protocol['config']['frames'])]:add('sample.'+n,v,0)
    add('verification.violations',load(EVIDENCE/'05_aggregate/verification.json')['accepted_contract_violations'],0)
    for pop in ['point','trajectory','sensitivity']:
        for r in tables[pop+'_main']:
            if pop=='point' and not r['witness_feasible']:continue
            pre=f'{pop}.{r["robot"]}.{r["method"]}'
            if pop=='sensitivity':pre+=f'.{r["scale"]:g}'
            for ep in ['verified_success','internal_success','accepted_within_20ms']:add(pre+'.'+ep,100*r[ep],2)
            for q in ['p50','p95','p99','mean']:add(pre+'.'+q,r['latency_'+q+'_ms'])
            for err,scale in [('position',1000),('orientation',180/math.pi)]:
                for q in ['p50','p95','p99','max']:add(pre+'.'+err+'_'+q,r['accepted_'+err][q]*scale,4)
            for cell in ['internal_success__task_accept','internal_success__task_reject','internal_failure__task_accept','internal_failure__task_reject']:add(pre+'.'+cell,r[cell],0)
            if pop=='trajectory':
                add(pre+'.completion','/'.join(map(str,r['completion_counts'])))
                lo,hi=min(r['completion_counts']),max(r['completion_counts'])
                add(pre+'.completion_range',str(lo) if lo==hi else f'{lo}--{hi}')
                add(pre+'.deadline','/'.join(map(str,r['deadline_completion_counts'])))
                add(pre+'.cumulative_s',r['cumulative_latency_ns_per_sweep']/1e9)
            if pop in ['point','sensitivity']:add(pre+'.miss_uid',r['any_repeat_missed_uids'],0)
    for robot,kk in k.items():
        for pop in ['point','trajectory']:
            for method,r in kk[pop+'_paired'].items():
                pre=f'{pop}.{robot}.{method}.contrast'
                ci(pre+'.time_ratio',r['latency_ratio']);ci(pre+'.success_pp',r['success_difference'],100)
                add(pre+'.saving_pct',100*(1-r['latency_ratio'][0]),2)
                add(pre+'.gained',len(r['gained_uids']),0);add(pre+'.lost',len(r['lost_uids']),0)
        for n,v in kk['dls_observation'].items():add('dls.'+robot+'.'+n,v,0)
        add('dls.'+robot+'.mismatch_pct',100*kk['point']['dls_strict']['native_failure_task_accept']/next(iter(nominal)),2)
    for robot in ['panda','ur5e']:
        for r in tables['trajectory_paired']:
            if r['robot']==robot and r['method']=='trac_task_5ms' and 'family' in r:
                add(f'family.{robot}.{r["family"]}.gained',len(r['gained_uids']),0)
        rr=[r for r in historic['allocation_point_paired'] if r['robot']==robot and r['subset']=='feasible']
        for method,ref,n in [('full_cghik','always_hard','full_hard'),('routing_only','always_hard','routing_hard'),('full_cghik','p50_selection','p95_p50'),('full_cghik','geometry_threshold','full_geometry')]:
            r=next(r for r in rr if r['method']==method and r['reference']==ref)
            for metric in ['latency_ratio','fev_ratio','p95_ratio','p99_ratio']:
                if r[metric]:ci(f'allocation.{robot}.{n}.{metric}',[float(r[metric]),float(r[metric+'_ci_low']),float(r[metric+'_ci_high'])])
            add(f'allocation.{robot}.{n}.time_increase_pct',100*(float(r['latency_ratio'])-1),2)
            add(f'allocation.{robot}.{n}.fev_reduction_pct',100*(1-float(r['fev_ratio'])),2)
        r=next(r for r in historic['allocation_decomposition'] if r['robot']==robot and r['group']=='neither_complete')
        add(f'allocation.{robot}.joint_failure_share',100*float(r['fraction_of_all_saved_latency']),2)
    def texrows(n,rows):(gen/(n+'.tex')).write_text('\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n'+r'\bottomrule'+'\n')
    for pop in ['point','trajectory']:
        rows=[]
        rs=[r for r in tables[pop+'_main'] if r.get('witness_feasible',True)]
        for r in sorted(rs,key=lambda x:(x['robot'],list(LABELS).index(x['method']))):
            if pop=='point':v=[f'{r["internal_success"]*100:.2f}',f'{r["verified_success"]*100:.2f}',r['any_repeat_missed_uids'],f'{r["latency_mean_ms"]:.3f}']
            else:v=['/'.join(map(str,r['completion_counts'])),'/'.join(map(str,r['deadline_completion_counts'])),f'{r["verified_success"]*100:.2f}',f'{r["cumulative_latency_ns_per_sweep"]/1e9:.3f}']
            rows.append(['Panda' if r['robot']=='panda' else 'UR5e',LABELS[r['method']],*v,*[f'{r["latency_"+q+"_ms"]:.3f}' for q in ['p50','p95','p99']]])
        texrows('task_'+pop+'_rows',rows)
    rows=[]
    for r in tables['point_main']:
        if r['witness_feasible']:rows.append([r['robot'],LABELS[r['method']],*[r[c] for c in ['internal_success__task_accept','internal_success__task_reject','internal_failure__task_accept','internal_failure__task_reject']]])
    texrows('task_status_rows',rows)
    rows=[]
    for r in tables['trajectory_paired']:
        if r['method']=='trac_task_5ms' and 'family' in r:
            rs={x['method']:x for x in tables['trajectory_families'] if x['robot']==r['robot'] and x['family']==r['family']}
            a,b=rs['trac_task_5ms'],rs['trac_strict_5ms'];v=r['latency_ratio']
            rows.append([r['robot'],r['family'].replace('_',' '),'/'.join(map(str,b['completion_counts'])),'/'.join(map(str,a['completion_counts'])),f'{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]',f'{len(r["gained_uids"])}/{len(r["lost_uids"])}'])
    texrows('task_family_rows',rows)
    rows=[]
    for robot in ['panda','ur5e']:
        for method,ref,label in [('full_cghik','always_hard','Full / hard'),('routing_only','always_hard','Routing-only / hard'),('full_cghik','geometry_threshold','Full / geometry'),('full_cghik','reject_only_hard','Full / reject-only'),('full_cghik','p50_selection','P95 / P50 selection')]:
            r=next(r for r in historic['allocation_point_paired'] if r['robot']==robot and r['subset']=='feasible' and r['method']==method and r['reference']==ref)
            vals=[f'{float(r[n]):.3f} [{float(r[n+"_ci_low"]):.3f}, {float(r[n+"_ci_high"]):.3f}]' for n in ['latency_ratio','fev_ratio']]
            rows.append([robot,label,*vals,f'{float(r["success_difference"])*100:.2f}'])
    texrows('task_allocation_rows',rows)
    mapping=load(EVIDENCE/'01_protocol/native_mappings.json');rows=[]
    for m in list(LABELS)[:6]:
        r=next(x for x in mapping if x['robot']=='panda' and x['scale']==1 and x['method']==m)
        b=r['bounds'];rows.append([LABELS[m],f'{b[0]*1000:.6f}',f'{b[3]:.8f}','Component box',('20' if '20ms' in m else '5')+' ms'])
    rows += [['DLS strict','0.010000','0.00001000','Separate norms','25 iter.'],['DLS aligned',f'{c["position_tolerance"]*1000:.6f}',f'{c["orientation_tolerance"]:.8f}','Separate norms','25 iter.']]
    texrows('task_mapping_rows',rows)
    with (data/'task_mapping_main.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['setting','position_component_mm_or_norm','orientation_component_rad_or_norm','shape','budget']);w.writerows(rows)
    (gen/'paper_numbers.tex').write_text('% Generated from frozen evidence; do not edit.\n'+'\n'.join(r'\expandafter\def\csname ev@'+n+r'\endcsname{'+v+'}' for n,v in sorted(nums.items()))+'\n')
    write(gen/'evidence_snapshot.json',dict(evidence_commit=COMMIT,sources=sources,protocol=protocol,tables=tables,historical_allocation=historic,numbers=nums,data_files={str(p.relative_to(PAPER)):digest(p) for p in data.glob('task_*.csv')}))
    print(f'Built {len(nums)} evidence macros and six table fragments; no solver calls.')
if __name__=='__main__':main()
