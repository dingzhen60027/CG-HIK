"""Read-only reduction of Phase 1.5; all four methods/all scenes retained."""
import json
import gzip
from pathlib import Path
import numpy as np
import h5py
from scipy.interpolate import BSpline
from .phase15 import OUT,OLD,ROOT,code_hashes
from .study import write,sha
from .reporting import csv_write,scene_rows,summary,paired,bridge,numerical_work,failure_breakdown
from .models import context
from .task import ScanTask
from .planning import ray_center_function

METHODS=('B0','B1','G','B2')


def bottlenecks(rows):
    answer=[]
    for row in rows:
        path=OUT/'runs'/row['slot']/f"{row['method']}_r0/planned_path.npz"
        if not path.exists():continue
        identity=json.loads((OUT/'inputs'/row['slot']/'identity.json').read_text())
        task=ScanTask(**identity['task']);adapter,_=context(row['robot']);raw=np.load(path)
        C=raw['coeff'];basis=BSpline(raw['knots'],np.eye(len(C)),3)
        s=(raw['s'][:-1]+raw['s'][1:])/2;x=(raw['x'][:-1]+raw['x'][1:])/2;dt=np.diff(raw['t'])
        q=basis(s)@C;qs=basis(s,nu=1)@C;qss=basis(s,nu=2)@C
        vr=np.max(abs(qs)*np.sqrt(x)[:,None]/(.5*adapter.public.limits.velocity),axis=1)
        ar=np.max(abs(qss*x[:,None]+qs*raw['u'][:,None]),axis=1)/2
        ray=ray_center_function(task,adapter)
        sr=np.array([np.linalg.norm(np.asarray(ray(qq)[1])@dd)*np.sqrt(xx)/.198 for qq,dd,xx in zip(q,qs,x)])
        scan=task.references(s)[4];total=float(sum(dt))
        answer.append(dict(slot=row['slot'],robot=row['robot'],family=row['family'],method=row['method'],
            planned_total_s=total,planned_scan_s=float(sum(dt[scan])),planned_turn_s=float(sum(dt[~scan])),
            actual_scan_s=row.get('scan_duration_s'),actual_turn_s=row.get('transition_duration_s'),actual_settle_s=row.get('settling_duration_s'),
            near_public_joint_speed_fraction=float(sum(dt[vr>=.95])/total),
            near_public_acceleration_fraction=float(sum(dt[ar>=.95])/total),
            near_reserved_acceleration_fraction=float(sum(dt[ar/.90**2>=.95])/total),
            near_reserved_sampling_fraction=float(sum(dt[(sr/.90>=.95)&scan])/total),
            actual_acceleration_max=row.get('acceleration_utilization_max'),actual_tracking_max_rad=row.get('tracking_error_max_rad'),
            torque_saturation_fraction=row.get('torque_saturation_fraction'),
            constraint_occupancy_is_nonexclusive=True,quality_completed=row['quality_completed']))
        dest=OUT/'reports/planned_derivatives'/f"{row['slot']}_{row['method']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,s=s,dt=dt,q=q,q_dot=qs*np.sqrt(x)[:,None],q_ddot=qss*x[:,None]+qs*raw['u'][:,None],
                            scan=scan,sample_utilization=sr,speed_utilization=vr,acceleration_utilization=ar)
    return answer


def calibration_tables():
    margin=[];b2=[];reference=[]
    for p in sorted((OUT/'execution_calibration').glob('*/*/execution_metrics.json')):
        m=json.loads(p.read_text());t=json.loads((p.parent/'timing.json').read_text())
        margin.append(dict(path=p.parent.parent.name,setting=p.parent.name,planned_s=t['duration'],**m))
    for p in sorted((OUT/'b2_calibration').glob('*/summary.json')):
        d=json.loads(p.read_text());b2.append(dict(problem=p.parent.name,status=d['status'],initial_s=d['initial_s'],
            known_faster_s=d['known_faster_s'],adopted_s=d['adopted_s'],
            initial_actual_s=d['metrics']['initial']['simulated_completion_time_s'],
            adopted_actual_s=d['metrics']['adopted']['simulated_completion_time_s'],
            known_faster_actual_s=d['metrics']['known_faster']['simulated_completion_time_s'],
            **d['derivative_check']))
    for p in sorted((OUT/'inputs').glob('*/identity.json')):
        d=json.loads(p.read_text());reference.append(dict(slot=d['slot'],scene_uid=d['scene_uid'],
            available=d['reference_geometry_feasible'],selected_tilt_deg=d['selected_tilt_deg'],reference_source=d['reference_source']))
    csv_write(OUT/'reports/execution_calibration.csv',margin);csv_write(OUT/'reports/b2_calibration.csv',b2)
    csv_write(OUT/'reports/reference_audit.csv',reference)
    return margin,b2,reference


def figures(main,rows,bottle,margin,b2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,
        'axes.titlesize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    root=OUT/'reports/figures';root.mkdir(parents=True,exist_ok=True)
    write(root/'contract.json',dict(conclusion='Distinguish verified baseline capability, realized quality, and execution bottlenecks.',
        archetype='quantitative grid',backend='Python matplotlib',unit='scene; timing repeats averaged within scene',
        paired_intervals='six CAD-placement clusters per robot; directions nested',
        plots='all scene values or explicitly qualified paired subsets; no plotted inferential bars',
        limits='Missing/failing paths count against all-scene yield; no formal or proposed-method result',
        export='editable SVG, vector PDF, 300 dpi PNG; minimum 6 pt',source='reports CSV and immutable recorded execution'))
    colors=['#467a9b','#49a191','#a88eb8','#c7884f']
    def save(fig,name):
        for ext in ('svg','pdf','png'):fig.savefig(root/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9,5.8),layout='constrained')
    for i,robot in enumerate(('panda','ur5e')):
        for j,method in enumerate(METHODS):
            a=next(r for r in main if r['robot']==robot and r['method']==method)
            axes[i,0].bar(j,a['quality_completed_scenes'],color=colors[j])
            axes[i,0].text(j,a['quality_completed_scenes']+.15,str(a['quality_completed_scenes']),ha='center',fontsize=7)
            for ax,key in ((axes[i,1],'quality_execution_s'),(axes[i,2],'total_plan_wall_s')):
                values=[r[key] for r in rows if r['robot']==robot and r['method']==method and r.get(key) is not None]
                ax.scatter(np.full(len(values),j),values,s=12,color=colors[j],alpha=.7)
                if values:ax.plot([j-.2,j+.2],[np.median(values)]*2,color='black',lw=1)
        axes[i,0].set(ylim=(0,13),yticks=[0,3,6,9,12],ylabel=f'{robot.upper()}\nQuality-complete / 12')
        axes[i,1].set_ylabel('Qualified actual cycle time (s)');axes[i,2].set_ylabel('All measured planning work (s)')
        for ax in axes[i]:ax.set_xticks(range(4),METHODS);ax.grid(axis='y',alpha=.15)
    for ax,title in zip(axes[0],('a  All scheduled scenes','b  Qualified scenes only','c  All timed scenes')):ax.set_title(title,loc='left')
    save(fig,'core_baselines')
    fig,axes=plt.subplots(1,3,figsize=(9,3),layout='constrained')
    names=sorted(set(r['path'] for r in margin))
    for label,color in (('before','#888888'),('after','#467a9b')):
        vals=[next(r for r in margin if r['path']==name and r['setting']==label) for name in names]
        axes[0].plot(range(len(names)),[r['acceleration_utilization_max'] for r in vals],'o-',label=label,color=color)
        axes[1].plot(range(len(names)),[1000*r['max_hole_diameter_upper_bound_m'] for r in vals],'o-',label=label,color=color)
    axes[0].axhline(1,color='black',ls='--',lw=.7);axes[1].axhline(2,color='black',ls='--',lw=.7)
    for ax in axes[:2]:ax.set_xticks(range(len(names)),[n.replace('panda','P').replace('ur5e','U').replace('_','\n') for n in names],fontsize=6)
    axes[0].set(title='a  Isolated margin calibration',ylabel='Actual acceleration / public limit');axes[0].legend(fontsize=7)
    axes[1].set(title='b  Coverage holes',ylabel='Hole bound (mm)')
    for k,r in enumerate(b2):
        axes[2].plot([0,1,2],[r['initial_actual_s'],r['adopted_actual_s'],r['known_faster_actual_s']],'o-',label=r['problem'])
    axes[2].set_xticks([0,1,2],['Slow input','B2 adopted','Known faster'],fontsize=7)
    axes[2].set(title='c  Independent B2 validation',ylabel='Quality-qualified actual cycle (s)');axes[2].legend(fontsize=7)
    save(fig,'calibration_closure')
    fig,axes=plt.subplots(2,2,figsize=(8,5.5),layout='constrained')
    for i,robot in enumerate(('panda','ur5e')):
        for j,method in enumerate(METHODS):
            rr=[r for r in bottle if r['robot']==robot and r['method']==method]
            if rr:
                turn=[r['planned_turn_s']/r['planned_total_s'] for r in rr]
                axes[i,0].scatter(np.full(len(rr),j),turn,color=colors[j],s=14)
        a={r['slot']:r for r in rows if r['robot']==robot and r['method']=='B1'}
        b={r['slot']:r for r in rows if r['robot']==robot and r['method']=='B2'}
        keys=[k for k in a if a[k]['quality_execution_s'] is not None and b[k]['quality_execution_s'] is not None]
        axes[i,1].scatter(range(len(keys)),[b[k]['quality_execution_s']-a[k]['quality_execution_s'] for k in keys],color=colors[3],s=18)
        axes[i,1].axhline(0,color='black',lw=.6);axes[i,1].set(xlabel=f'Jointly quality-complete scenes (n={len(keys)})',ylabel='Actual B2 − B1 cycle time (s)')
        axes[i,0].set_xticks(range(4),METHODS);axes[i,0].set_ylabel(f'{robot.upper()}\nPlanned transition-time fraction')
    axes[0,0].set_title('a  All available paths',loc='left');axes[0,1].set_title('b  Paired physical comparison',loc='left')
    save(fig,'bottlenecks_and_paired_times')


def report():
    raw=[json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/*/metrics.json'))]
    if len(raw)!=288:raise RuntimeError(f'Expected 288 core conditions, found {len(raw)}')
    rows=scene_rows(raw,METHODS);main=[];families=[]
    for robot in ('panda','ur5e'):
        for method in METHODS:
            sub=[r for r in rows if r['robot']==robot and r['method']==method]
            main.append(dict(robot=robot,method=method,**summary(sub)))
            for family in ('plane','cylinder','saddle'):
                families.append(dict(robot=robot,method=method,family=family,**summary([r for r in sub if r['family']==family])))
    bridges=bridge(rows,OUT);work=numerical_work(raw,OUT);fails=failure_breakdown(rows,OUT);bottle=bottlenecks(rows);intervals=paired(rows)
    for name,values in (('all_conditions',raw),('scene_results',rows),('baseline_main',main),('baseline_families',families),
                        ('bridge_tb_vs_gn',bridges),('bottlenecks',bottle),('failure_breakdown',fails),('numerical_work',work),('paired_intervals',intervals)):
        csv_write(OUT/'reports'/f'{name}.csv',values)
    margin,b2,refs=calibration_tables();figures(main,rows,bottle,margin,b2)
    hashes=json.loads((OUT/'inputs/seal.json').read_text())['code_hashes']
    write(OUT/'reports/integrity.json',dict(conditions=len(raw),expected_conditions=288,
        source_unchanged=hashes==code_hashes(),common_references=sum(r['available'] for r in refs),
        physical_runs=sum(r.get('physical_run',False) for r in raw),formal_scenes_run=False,
        raw_record_index=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((OUT/'runs').rglob('*')) if p.is_file()]))
    write(OUT/'reports/summary.json',dict(main=main,b2_calibration=b2,reference_count=sum(r['available'] for r in refs),
        failure_rows=fails,paired_intervals=intervals))
    print(json.dumps(main,indent=2),flush=True)


if __name__=='__main__':report()
