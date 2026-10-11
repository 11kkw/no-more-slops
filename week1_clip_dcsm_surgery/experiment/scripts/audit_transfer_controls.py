"""Diagnostic evaluation of EXISTING misaligned checkpoints; not corrected retraining evidence."""
import json, time, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from compositional_eval import DCSMScorer, crop, pad_box, union_box, IMG_SEQ, TEXT_SEQ, SPECIAL_WORDS
from probe_localization import dense_features
from train_dcsm_scorer import insert_special, ObjaverseSpatial

torch.set_num_threads(4)
OUT=Path('results/validation_audit'); OUT.mkdir(exist_ok=True,parents=True)
clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True).to('cuda').eval()
tok=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True)
proc=CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True)
scorers={r:DCSMScorer('results/retrain/dcsm_'+r+'_retrain.pt',clip,tok,proc,'cuda') for r in ['naive','maskclip','clearclip']}
items=[q for q in json.loads(Path('data/comp_queries/comp_queries.json').read_text())['items'] if q['subset']=='spatial']
# Deterministic relation-stratified control subset, chosen without performance selection.
control_idx=set(); byrel=defaultdict(list)
for i,q in enumerate(items): byrel[q['rel']].append(i)
for ids in byrel.values(): control_idx.update(ids)

def text_vec(q):
 tt=tok([q],padding='max_length',max_length=TEXT_SEQ,truncation=True,return_tensors='pt').to('cuda')
 return torch.nn.functional.normalize(clip.text_projection(clip.text_model(**tt).last_hidden_state),dim=-1)

def raw_cos(feat,txt,q,scorer):
 cos=torch.einsum('bie,lte->blit',feat,txt)
 return insert_special(cos,[q],scorer.emb,'cuda')

def score(cos,scorer,mode='batch',fixed=None):
 if mode=='batch': c=(cos-cos.mean())/cos.std()
 elif mode=='per_candidate': c=(cos-cos.mean(dim=(1,2,3),keepdim=True))/cos.std(dim=(1,2,3),keepdim=True)
 else: c=(cos-fixed[0])/fixed[1]
 return scorer.net(c).flatten().cpu().tolist()

# Fixed normalization calibration uses training source ONLY, not evaluation candidate sets.
sp=ObjaverseSpatial('data/dcsm_train/objaverse_train_cutmix_newrelations_withbg',0)
calib=[]
for i in np.linspace(0,len(sp)-1,16,dtype=int):
 im,_,q,_=sp[int(i)]; calib.append((im,q))
fixed={}
with torch.inference_mode():
 pv=proc(images=[x[0] for x in calib],return_tensors='pt')['pixel_values'].to('cuda')
 feats,_=dense_features(clip,pv,224)
 tt=tok([x[1] for x in calib],padding='max_length',max_length=TEXT_SEQ,truncation=True,return_tensors='pt').to('cuda')
 txt=torch.nn.functional.normalize(clip.text_projection(clip.text_model(**tt).last_hidden_state),dim=-1)
 for r,s in scorers.items():
  c=insert_special(torch.einsum('bie,lte->blit',feats[r],txt),[x[1] for x in calib],s.emb,'cuda')
  fixed[r]=(float(c.mean()),float(c.std()))
(OUT/'fixed_normalization_calibration.json').write_text(json.dumps({'source':'16 deterministic evenly spaced Objaverse training samples; 16x16 matrix, no eval calibration','stats':fixed},indent=2))

opp={'to the left of':'to the right of','to the right of':'to the left of','above':'below','below':'above'}
records=[]; start=time.time()
with torch.inference_mode():
 for i,it in enumerate(items):
  img=Image.open(Path('data/coco/val2017')/it['file_name']).convert('RGB'); W,H=img.size
  crops=[crop(img,pad_box(union_box(c,it['B_box']),W,H)) for c in it['candidates']]
  pv=proc(images=crops,return_tensors='pt')['pixel_values'].to('cuda'); feats,_=dense_features(clip,pv,224); txt=text_vec(it['query'])
  sc={}
  for r in scorers:
   sc['retrained_'+r]=score(raw_cos(feats[r],txt,it['query'],scorers[r]),scorers[r])
   if r!='naive': sc['ordinary_scorer_on_'+r]=score(raw_cos(feats[r],txt,it['query'],scorers['naive']),scorers['naive'])
  rec={'index':i,'image_id':it['image_id'],'query':it['query'],'rel':it['rel'],'target':it['target_idx'],'n':len(crops),'scores':sc,'pred':{k:int(np.argmax(v)) for k,v in sc.items()}}
  if i in control_idx:
   ctl={}
   for r,s in scorers.items():
    c=raw_cos(feats[r],txt,it['query'],s)
    ctl[r+'_per_candidate_norm']=score(c,s,'per_candidate')
    ctl[r+'_fixed_train_norm']=score(c,s,'fixed',fixed[r])
   for tag,q in [('relation_removed',it['query'].replace(it['rel'],' ')),('relation_reversed',it['query'].replace(it['rel'],opp[it['rel']]))]:
    tv=text_vec(q)
    for r,s in scorers.items(): ctl[r+'_'+tag]=score(raw_cos(feats[r],tv,q,s),s)
   box=it['B_box']
   for c in it['candidates']: box=union_box(box,c)
   common=crop(img,pad_box(box,W,H)); cpv=proc(images=[common]*len(crops),return_tensors='pt')['pixel_values'].to('cuda'); cf,_=dense_features(clip,cpv,224)
   for r,s in scorers.items(): ctl[r+'_common_crop']=score(raw_cos(cf[r],txt,it['query'],s),s)
   rec['control_scores']=ctl; rec['control_pred']={k:int(np.argmax(v)) for k,v in ctl.items()}
  records.append(rec)
  if (i+1)%50==0:
   (OUT/'transfer_control_per_item.partial.json').write_text(json.dumps(records)); print('progress',i+1,'/',len(items),'seconds',round(time.time()-start,1),flush=True)

summary={'status':'complete','interpretation':'Historical diagnostics on misaligned checkpoints. Does NOT establish correct relation-token adaptation or a corrected multi-seed result.','n_spatial':len(records),'n_control':sum('control_pred' in x for x in records),'control_relation_counts':dict(Counter(x['rel'] for x in records if 'control_pred' in x)),'adaptation':{},'controls':{},'elapsed_seconds':time.time()-start}
for k in records[0]['pred']:
 summary['adaptation'][k]={'correct':sum(x['pred'][k]==x['target'] for x in records),'n':len(records),'top1':float(np.mean([x['pred'][k]==x['target'] for x in records]))}
ctrl=[x for x in records if 'control_pred' in x]
for k in ctrl[0]['control_pred']:
 r=k.split('_')[0]; base='retrained_'+r
 summary['controls'][k]={'n':len(ctrl),'same_original_target_accuracy':float(np.mean([x['control_pred'][k]==x['target'] for x in ctrl])),'prediction_change_rate':float(np.mean([x['control_pred'][k]!=x['pred'][base] for x in ctrl])),'score_tie_rate':float(np.mean([np.ptp(x['control_scores'][k])<1e-6 for x in ctrl]))}
summary['controls_note']='Reversed relation is measured as prediction sensitivity, NOT correct reversed-label accuracy: opposite direction can have multiple correct candidates. Common crop with no candidate-specific representation makes the inputs identical and ties structurally inevitable.'
for r in scorers:
 summary['controls'][r+'_original_on_control']={'top1':float(np.mean([x['pred']['retrained_'+r]==x['target'] for x in ctrl])),'n':len(ctrl)}
(OUT/'transfer_control_per_item.json').write_text(json.dumps(records))
(OUT/'transfer_controls_summary.json').write_text(json.dumps(summary,indent=2))
examples={}
for r in scorers:
 examples[r]={typ:[{'image_id':x['image_id'],'query':x['query'],'target':x['target'],'prediction':x['pred']['retrained_'+r]} for x in records if (x['pred']['retrained_'+r]==x['target'])==(typ=='success')][:5] for typ in ['success','failure']}
(OUT/'success_failure_examples.json').write_text(json.dumps(examples,indent=2))
print(json.dumps(summary,indent=2),flush=True)
