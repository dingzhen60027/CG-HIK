"""Evidence-only figures and tables. No numerical solver is imported or run here."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..revision_compute_allocation.common import json_write,digest
from .study import Study,now

LABELS={'trac_strict_5ms':'Strict 5','trac_position_5ms':'Position 5',
    'trac_orientation_5ms':'Orientation 5','trac_task_5ms':'Aligned 5',
    'trac_strict_20ms':'Strict 20','trac_task_20ms':'Aligned 20',
    'dls_strict':'DLS strict','dls_task':'DLS aligned'}
COLORS={'trac_strict_5ms':'#777777','trac_position_5ms':'#997BBD',
    'trac_orientation_5ms':'#C18D53','trac_task_5ms':'#247F91',
    'trac_strict_20ms':'#AAAAAA','trac_task_20ms':'#64A8B5',
    'dls_strict':'#94755E','dls_task':'#CB9B7E'}


def read(s,name):
    return json.loads((s.out/'05_aggregate'/f'{name}.json').read_text())


def save(fig,folder,name):
    for ext in ['pdf','svg','png']:
        path=folder/f'{name}.{ext}'
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=300)
    plt.close(fig)


def figures(s,folder):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,
        'legend.frameon':False,'figure.constrained_layout.use':True})
    point=read(s,'point_main');sens=read(s,'sensitivity_main');traj=read(s,'trajectory_main')
    fig,ax=plt.subplots(figsize=(7.2,3.7));ax.axis('off')
    cells=[['Internal success\nTask accepts\nAdmissible numerical return',
            'Internal success\nTask rejects\nReturn code is not acceptance'],
           ['Internal failure\nTask accepts\nAdmissible despite nonconvergence',
            'Internal failure\nTask rejects\nNo accepted command returned']]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j*.5,1-(i+1)*.4),.48,.36,facecolor='#EAF2F3' if j==0 else '#F0ECE8',edgecolor='white'))
            ax.text(j*.5+.24,1-(i+1)*.4+.18,cells[i][j],ha='center',va='center',linespacing=1.6)
    ax.text(.5,.04,'Witness-confirmed miss: task rejection + independently verified command\nfor the identical target, previous state, dt and task contract.',ha='center',va='center')
    ax.set(xlim=(0,1),ylim=(0,1));save(fig,folder,'figure1_taxonomy')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.5))
    for i,robot in enumerate(['panda','ur5e']):
        rs=[r for r in point if r['robot']==robot and r['witness_feasible']]
        rs=sorted(rs,key=lambda r:list(LABELS).index(r['method']));x=np.arange(len(rs))
        for j,key in enumerate(['verified_success','latency_p95_ms']):
            ax=axes[i,j];vals=[r[key]*(100 if j==0 else 1) for r in rs]
            ax.bar(x,vals,color=[COLORS[r['method']] for r in rs]);ax.set_xticks(x,[LABELS[r['method']] for r in rs],rotation=45,ha='right',rotation_mode='anchor')
            ax.set_title(f'{robot.upper()} — '+('admissibility' if j==0 else 'whole-call tail'))
            ax.set_ylabel('Verified (%)' if j==0 else 'P95 (ms)')
            if j==0:ax.set_ylim(0,105)
    save(fig,folder,'figure2_point_alignment')
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.5))
    for i,robot in enumerate(['panda','ur5e']):
        for method in ['trac_strict_5ms','trac_task_5ms','dls_task']:
            rs=sorted([r for r in sens if r['robot']==robot and r['method']==method],key=lambda r:r['scale'])
            for j in range(3):
                y=[(1-r['verified_success'])*100 if j==0 else r['latency_p95_ms'] if j==1 else r['accepted_position']['p95']/(.001*r['scale']) for r in rs]
                axes[i,j].plot([r['scale'] for r in rs],y,'o-',color=COLORS[method],label=LABELS[method])
        for j in range(3):
            axes[i,j].set_xticks([.5,1,2],['0.5×','1×','2×']);axes[i,j].set_xlabel('Pose contract scale')
            axes[i,j].set_title(robot.upper());axes[i,j].set_ylabel(['Missed calls (%)','P95 latency (ms)','P95 position / tolerance'][j])
        axes[i,0].legend(fontsize=7)
    save(fig,folder,'figure3_contract_sensitivity')
    fig,axes=plt.subplots(4,2,figsize=(7.2,7.6))
    for i,robot in enumerate(['panda','ur5e']):
        rs=sorted([r for r in traj if r['robot']==robot],key=lambda r:list(LABELS).index(r['method']))
        for j in range(4):
            ax=axes[j,i]
            for x,r in enumerate(rs):
                vals=r['completion_counts'] if j==0 else [r['cumulative_latency_ns_per_sweep']/1e9] if j==1 else [r['verified_success']*100] if j==2 else [r['latency_p50_ms'],r['latency_p95_ms'],r['latency_p99_ms']]
                if j<3:
                    ax.bar(x,np.mean(vals),color=COLORS[r['method']])
                    if j==0:ax.plot([x]*len(vals),vals,'k.',markersize=3)
                else:
                    ax.plot([x]*3,vals,'-',color=COLORS[r['method']]);ax.scatter([x]*3,vals,color=COLORS[r['method']],s=[8,16,30])
            ax.set_xticks(range(len(rs)),[LABELS[r['method']] for r in rs] if j==3 else ['']*len(rs),rotation=45,ha='right',rotation_mode='anchor')
            if j==0:ax.set_title(robot.upper())
            ax.set_ylabel(['Completed / 40','Cumulative time (s)','Frame verified (%)','P50 / P95 / P99 (ms)'][j])
            if j==0:ax.set_ylim(0,42)
            if j==2:ax.set_ylim(0,105)
    save(fig,folder,'figure4_trajectories')
    paired=read(s,'trajectory_paired')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4))
    for i,robot in enumerate(['panda','ur5e']):
        rs=[r for r in paired if r['robot']==robot and r['method']=='trac_task_5ms' and 'family' in r]
        ax=axes[i]
        for j,r in enumerate(rs):
            c,lo,hi=r['latency_ratio'];ax.errorbar(c,j,xerr=[[c-lo],[hi-c]],fmt='o',color='#247F91',capsize=3)
            ax.text(1.,j+.22,f'+{len(r["gained_uids"])} / −{len(r["lost_uids"])} UIDs',transform=ax.get_yaxis_transform(),ha='right',fontsize=7)
        ax.set_yticks(range(len(rs)),[r['family'].replace('_','\n') for r in rs]);ax.axvline(1,color='#999999',lw=.8)
        ax.set_xlabel('Aligned / strict cumulative time\nPaired trajectory bootstrap 95% CI');ax.set_title(robot.upper())
    save(fig,folder,'figure5_family_effects')
    traces=read(s,'dls_excess_iterations')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
    for ax,robot in zip(axes,['panda','ur5e']):
        rs=[r for r in traces if r['robot']==robot and r['method']=='dls_strict']
        vals=[r['excess_iterations_after_task_admissibility'] for r in rs if r['excess_iterations_after_task_admissibility'] is not None]
        ax.hist(vals,bins=np.arange(-.5,26.5),color='#94755E');ax.set_xlabel('Excess iterations after task admissibility');ax.set_ylabel('Queries')
        ax.set_title(f'{robot.upper()} — {len(vals)} observed admissible iterates')
    save(fig,folder,'supplement_dls_excess_iterations')


def table_text(rows,trajectory=False):
    lines=['| Robot | Setting | '+('Complete / 40' if trajectory else 'Verified %')+' | P50 / P95 / P99 ms | '+('Cumulative s' if trajectory else 'Mean ms')+' |','|---|---|---:|---|---:|']
    for r in rows:
        success='/'.join(map(str,r['completion_counts'])) if trajectory else f'{100*r["verified_success"]:.3f}'
        cost=r['cumulative_latency_ns_per_sweep']/1e9 if trajectory else r['latency_mean_ms']
        lines.append(f'| {r["robot"]} | {LABELS[r["method"]]} | {success} | {r["latency_p50_ms"]:.3f} / {r["latency_p95_ms"]:.3f} / {r["latency_p99_ms"]:.3f} | {cost:.3f} |')
    return '\n'.join(lines)


def main():
    s=Study();folder=s.out/'reports';folder.mkdir(exist_ok=True)
    assert (s.out/'05_aggregate/completed.json').exists()
    json_write(folder/'figure_contract.json',dict(
        backend='Python matplotlib',width_inches=7.2,exports=['PDF editable text','SVG editable text','PNG preview'],
        figures={'1':'schematic: four internal/task cells and exact-input witness condition',
            '2':'quantitative grid: all prespecified point settings, feasibility and P95 shown separately',
            '3':'quantitative grid: every scale, missed calls, P95, and accepted position relative to contract',
            '4':'quantitative grid: both robots, completion/repeat spread, full cumulative cost, latency quantiles',
            '5':'paired effects: all families, cumulative cost ratio/95% interval and gained/lost UIDs',
            'S1':'observed DLS iteration distribution; no hypothetical TRAC trace'},
        uncertainty='Fig5 family-stratified paired trajectory percentile bootstrap, 4000 resamples; Fig4 completion dots are planned within-trajectory repeats; other panels empirical full-call distributions, descriptive not independent-frame inference',
        omissions='none from prescribed populations; constructed inexecutable point results reported separately in tables',
        source_data='05_aggregate JSON and CSV, directly read without hand-maintained numeric constants'))
    figures(s,folder)
    p=[r for r in read(s,'point_main') if r['witness_feasible']];t=read(s,'trajectory_main')
    with (folder/'MAIN_TABLES.md').open('x') as f:
        f.write('# Main results\n\n## Solver-native contract mapping\n\nSee `01_protocol/native_mappings.json` and the fixed protocol mapping table.\n\n## Witness-feasible points\n\n'+table_text(p)+'\n\n## Complete trajectories\n\n'+table_text(t,True)+'\n')
    json_write(folder/'report_manifest.json',dict(created_utc=now(),files={str(p.relative_to(s.root)):digest(p) for p in folder.iterdir() if p.is_file()}))


if __name__=='__main__':main()
