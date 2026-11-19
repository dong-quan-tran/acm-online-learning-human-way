"""Plot fixed held-out TTT accuracy. Usage: python plot_heldout_ttt.py input.csv --out plots"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED=['experiment','condition','seed','source_checkpoint','code_version','adaptation_stream_condition','heldout_condition','adaptation_images_processed','heldout_image_count','correct_prediction_count','heldout_accuracy','adaptation_update_count','adaptation_time_seconds','heldout_evaluation_time_seconds','config_path','split_file_path']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('csv',nargs='+',help='One or more result CSV files')
    ap.add_argument('--out',default='plots')
    ap.add_argument('--reference-csv',help='Optional frozen baseline CSV with columns experiment,seed,heldout_condition,method,accuracy')
    args=ap.parse_args()
    df=pd.concat([pd.read_csv(f) for f in args.csv],ignore_index=True)
    missing=set(REQUIRED)-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    numeric=['adaptation_images_processed','heldout_image_count','correct_prediction_count','heldout_accuracy','adaptation_update_count','adaptation_time_seconds','heldout_evaluation_time_seconds']
    for col in numeric: df[col]=pd.to_numeric(df[col],errors='raise')
    if df[numeric].isna().any().any(): raise ValueError('Missing numeric values')
    if (df.heldout_image_count<=0).any() or (df.correct_prediction_count>df.heldout_image_count).any() or (df.correct_prediction_count<0).any(): raise ValueError('Invalid prediction counts')
    calculated=df.correct_prediction_count/df.heldout_image_count
    if ((df.heldout_accuracy-calculated).abs()>0.0011).any(): raise ValueError('heldout_accuracy must be a fraction (0 to 1) matching correct/total')
    keys=['experiment','seed','adaptation_stream_condition','heldout_condition']
    if df.duplicated(keys+['adaptation_images_processed']).any(): raise ValueError('Duplicate evaluation points for same run')
    ref=pd.read_csv(args.reference_csv) if args.reference_csv else None
    if ref is not None:
        needed={'experiment','seed','heldout_condition','method','accuracy'}
        if not needed.issubset(ref.columns): raise ValueError(f'Reference CSV requires {sorted(needed)}')
        if not ref.method.isin(['joint_frozen','supervised_frozen']).all(): raise ValueError('Reference method must be joint_frozen or supervised_frozen')
        if ref.duplicated(['experiment','seed','heldout_condition','method']).any(): raise ValueError('Duplicate frozen reference')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for (exp,seed,stream),sub in df.groupby(['experiment','seed','adaptation_stream_condition']):
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
        for ax,held in zip(axes,['clean','noisy']):
            s=sub[sub.heldout_condition==held].sort_values('adaptation_images_processed')
            if s.empty: ax.set_title(f'{held}: no data');continue
            if s.adaptation_images_processed.iloc[0]!=0: raise ValueError(f'Missing step 0 for {exp} seed {seed} {stream} {held}')
            x=s.adaptation_images_processed.to_numpy();y=s.heldout_accuracy.to_numpy()*100
            ax.plot(x,y,marker='o',label='Online TTT')
            start=float(y[0]);ax.axhline(start,linestyle=':',color='gray',label=f'Online start ({start:.2f}%)')
            joint=None
            if ref is not None:
                r=ref[(ref.experiment.astype(str)==str(exp))&(ref.seed.astype(str)==str(seed))&(ref.heldout_condition==held)]
                for method,style in [('joint_frozen','--'),('supervised_frozen','-.')]:
                    v=r[r.method==method]
                    if len(v):
                        value=float(v.accuracy.iloc[0])*100
                        ax.axhline(value,linestyle=style,label=f'{method.replace("_"," ")} ({value:.2f}%)')
                        if method=='joint_frozen':joint=value
            ax.set_title(f'Held-out: {held}')
            ax.set_xlabel('Adaptation images processed')
            ax.set_ylabel('Held-out classification accuracy (%)')
            ax.grid(alpha=.25);ax.legend(fontsize=8)
            for _,row in s.iterrows():
                summary.append({'experiment':exp,'seed':seed,'adaptation_stream_condition':stream,'heldout_condition':held,'adaptation_images_processed':int(row.adaptation_images_processed),'accuracy_pct':row.heldout_accuracy*100,'change_from_online_start_pp':(row.heldout_accuracy*100-start),'change_from_joint_frozen_pp':(row.heldout_accuracy*100-joint) if joint is not None else None,'adaptation_time_seconds':row.adaptation_time_seconds,'heldout_evaluation_time_seconds':row.heldout_evaluation_time_seconds})
        fig.suptitle(f'Experiment {exp} | Seed {seed} | Adaptation stream: {stream}')
        fig.tight_layout()
        name=f'exp_{exp}_seed_{seed}_stream_{stream}.png';fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig)
        print('Saved',out/name)
    pd.DataFrame(summary).to_csv(out/'accuracy_changes.csv',index=False)
    print('Saved',out/'accuracy_changes.csv')

if __name__=='__main__':main()
