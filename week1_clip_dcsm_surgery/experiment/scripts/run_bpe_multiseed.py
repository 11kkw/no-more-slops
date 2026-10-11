"""Durable, locked sequential 3-seed retraining and evaluation supervisor."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/bpe_multiseed'
PYTHON=Path('/home/jovyan/no-more-slops/week1_clip_dcsm_surgery/experiment/.venv/bin/python')
SEEDS=[15,16,17]
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'runner.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (OUT/'state.json').exists(): raise RuntimeError('Existing run state; refusing duplicate execution or overwrite')
state={'status':'running','pid':os.getpid(),'started_unix':time.time(),'seeds':SEEDS,'completed_seeds':[],'stage':'starting','seed':None,'child_pid':None,'commands':[]}

def save():
 p=OUT/'state.tmp'; p.write_text(json.dumps(state,indent=2)); p.replace(OUT/'state.json')

env=dict(os.environ,HF_HOME='/home/jovyan/no-more-slops/week1_clip_dcsm_surgery/experiment/hf_cache',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}
(OUT/'source_snapshot.json').write_text(json.dumps(sources,indent=2))

def run(seed,stage,args,log):
 for name,digest in sources.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('Source changed during run: '+name)
 cmd=[str(PYTHON),*args]; state.update(stage=stage,seed=seed,child_pid=None); state['commands'].append({'seed':seed,'stage':stage,'argv':cmd,'log':str(log),'started_unix':time.time()}); save()
 with log.open('w') as f:
  p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  state['child_pid']=p.pid; save(); rc=p.wait()
 state['commands'][-1].update(exit_code=rc,ended_unix=time.time()); state['child_pid']=None; save()
 if rc: raise RuntimeError(f'{stage} seed {seed} failed with exit {rc}; see {log}')

save()
try:
 for seed in SEEDS:
  d=OUT/f'seed_{seed}'; d.mkdir(exist_ok=False)
  train=d/'train'; evaluate=d/'evaluation'; adaptation=d/'adaptation'
  run(seed,'training',['scripts/train_dcsm_scorer.py','--data_root','data/dcsm_train','--spatial_dir','objaverse_train_cutmix_newrelations_withbg','--epochs','27','--batch_size','8','--lr','0.001','--workers','24','--seed',str(seed),'--out',str(train)],d/'train.log')
  init=json.loads((train/'initialization.json').read_text())
  assert len({x['scorer'] for x in init['initial_hashes'].values()})==1
  assert len({x['embedding'] for x in init['initial_hashes'].values()})==1
  ckpts=[str(train/f'dcsm_{r}_retrain.pt') for r in ['naive','maskclip','clearclip']]
  run(seed,'full_compositional_evaluation',['scripts/compositional_eval.py','--queries','data/comp_queries/comp_queries.json','--coco_root','data/coco','--dcsm_ckpts',*ckpts,'--out',str(evaluate)],d/'evaluation.log')
  run(seed,'adaptation_and_controls',['scripts/evaluate_bpe_adaptation.py','--ckpt_dir',str(train),'--out',str(adaptation)],d/'adaptation.log')
  state['completed_seeds'].append(seed); save()
 import numpy as np
 from scipy.stats import t
 ss=[json.loads((OUT/f'seed_{seed}/adaptation/transfer_controls_summary.json').read_text()) for seed in SEEDS]
 aggregate={'seeds':SEEDS,'n_spatial_each':[s['n_spatial'] for s in ss],'note':'Mean/SD and t intervals are across training seeds, not image-level bootstrap. 3 seeds provide limited precision.','top1':{},'paired_adaptation_gain':{}}
 def stats(v):
  v=np.array(v,dtype=float); mean=float(v.mean()); sd=float(v.std(ddof=1)); half=float(t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v)))
  return {'per_seed':v.tolist(),'mean':mean,'sample_std':sd,'mean_t95_interval':[mean-half,mean+half]}
 for k in ss[0]['adaptation']: aggregate['top1'][k]=stats([s['adaptation'][k]['top1'] for s in ss])
 for r in ['maskclip','clearclip']: aggregate['paired_adaptation_gain'][r]=stats([s['adaptation']['retrained_'+r]['top1']-s['adaptation']['ordinary_scorer_on_'+r]['top1'] for s in ss])
 (OUT/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
 state.update(status='completed',stage='done',finished_unix=time.time(),seed=None); save()
except BaseException as e:
 state.update(status='failed',error=str(e),traceback=traceback.format_exc(),finished_unix=time.time()); save(); raise
