"""Read-only audit of existing DCSM token replacement, initialization and data provenance."""
import json, hashlib, random, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import CLIPTokenizer
from compositional_eval import SPECIAL_WORDS, IMG_SEQ, TEXT_SEQ, CrossModalConvNetwork, tolerant_torch_load
from train_dcsm_scorer import insert_special, init_special_embedding, ObjaverseSpatial, ObjaverseColor

OUT=Path('results/validation_audit'); OUT.mkdir(parents=True,exist_ok=True)
tok=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True)
queries=json.loads(Path('data/comp_queries/comp_queries.json').read_text())['items']
ck=tolerant_torch_load('results/retrain/dcsm_naive_retrain.pt'); args=ck['args']; root=Path(args['data_root'])
sp=ObjaverseSpatial(str(root/args['spatial_dir']),0); co=ObjaverseColor(str(root/args['color_dir']),0)
neg=json.loads((root/args['neg_json']).read_text())
train_text=[]
from train_dcsm_scorer import REL_TEXT, REL_OPP, NEG_WORDS
for f in sp.files:
 a,r,b=Path(f).stem.split('_')
 train_text.extend([f'A picture of a {a} {REL_TEXT[r]} a {b}',f'A picture of a {b} {REL_TEXT[r]} a {a}',f'A picture of a {b} {REL_TEXT[REL_OPP[r]]} a {a}',f'A picture of a {a} {REL_TEXT[REL_OPP[r]]} a {b}'])
for _,f in co.items:
 c1,a,c2,b=Path(f).stem.split('_')[:4]
 train_text.extend([f'A picture of a {c1} {a} and a {c2} {b}',f'A picture of a {c2} {a} and a {c1} {b}',f'A picture of a {c2} {b} and a {c1} {a}',f'A picture of a {c1} {b} and a {c2} {a}'])
for n in neg:
 for v in (0,1):
  a,b=n['img1_obj'][v],n['img2_obj'][v]
  for w in NEG_WORDS: train_text.extend([f'An image of a {a} {w} a {b}',f'An image of a {b} {w} a {a}'])

emb=torch.nn.Embedding(len(SPECIAL_WORDS),IMG_SEQ)
with torch.no_grad():
 for i in range(len(SPECIAL_WORDS)): emb.weight[i].fill_(1000+i)

def inspect(texts):
 counts=Counter(); examples=[]
 for text in sorted(set(texts)):
  sent=text.split(' '); ids=tok(text,truncation=True,max_length=TEXT_SEQ)['input_ids']; tokens=tok.convert_ids_to_tokens(ids)
  for w in SPECIAL_WORDS:
   if w not in sent: continue
   intended=[i for i,t in enumerate(tokens) if t==w+'</w>']
   loc=sent.index(w); counts['special_word_occurrences']+=sent.count(w); counts['unique_text_word_cases']+=1
   match=(intended==[loc]); counts['aligned' if match else 'misaligned']+=1
   if len(examples)<12 and not match: examples.append({'text':text,'word':w,'implemented_column':loc,'actual_token_columns':intended,'replaced_token':tokens[loc] if loc<len(tokens) else None,'tokens':tokens})
 return {'counts':dict(counts),'examples':examples}

tests=[]
for text in ['A picture of a cat above a dog','A picture of a cat to the left of a dog','a picture of a traffic light below a fire hydrant','a picture of a hippopotamus below a dog','a picture of a cat above a dog above a bird']:
 cos=torch.arange(TEXT_SEQ,dtype=torch.float32).view(1,1,1,TEXT_SEQ).expand(1,1,IMG_SEQ,TEXT_SEQ).clone()
 out=insert_special(cos,[text],emb,'cpu'); ids=tok(text)['input_ids']; tokens=tok.convert_ids_to_tokens(ids)
 changed=(out[0,0,0]!=cos[0,0,0]).nonzero().flatten().tolist()
 expected=[i for i,t in enumerate(tokens[:TEXT_SEQ]) if t.removesuffix('</w>') in SPECIAL_WORDS]
 tests.append({'text':text,'changed_columns':changed,'expected_columns':expected,'pass':changed==expected,'tokens':tokens})

# Exact initialization recipe currently used: a single seed but sequential independent initializations.
torch.manual_seed(args['seed']); nets={r:CrossModalConvNetwork(IMG_SEQ,TEXT_SEQ) for r in args['readouts']}; embs={r:init_special_embedding('cpu') for r in args['readouts']}
base=args['readouts'][0]
init={r:{'scorer_matches_naive':all(torch.equal(v,nets[r].state_dict()[k]) for k,v in nets[base].state_dict().items()),'embedding_matches_naive':torch.equal(embs[base].weight,embs[r].weight)} for r in args['readouts']}

train_paths=set()
for f in sp.files: train_paths.update([root/args['spatial_dir']/f,root/args['spatial_dir']/('CUTMIX_'+f)])
for d,f in co.items: train_paths.update([Path(d)/f,Path(d)/('MIRROR_'+f)])
for n in neg:
 for k in ['img1_path','img2_path']: train_paths.add(root/n[k].replace('\\','/'))
train_paths=sorted({p.resolve() for p in train_paths}); eval_paths=sorted({(Path('data/coco/val2017')/q['file_name']).resolve() for q in queries})
missing=[str(p) for p in train_paths if not p.exists()]
train_coco={int(p.stem) for p in train_paths if p.suffix.lower()=='.jpg' and p.stem.isdigit()}; eval_ids={q['image_id'] for q in queries}
manifest={'checkpoint_args':args,'dataset_sizes':{'spatial':len(sp),'color_smallbig':len(co),'negation':len(neg)},'scope':'Available pool reconstructed from current data and checkpoint arguments; epoch-level actually sampled subset was NOT logged. Released checkpoints provenance is NOT covered.','training_files':[str(p) for p in train_paths],'evaluation_files':[str(p) for p in eval_paths],'missing_training_files':missing,'coco_id_overlap':sorted(train_coco&eval_ids),'exact_path_overlap':sorted(str(p) for p in set(train_paths)&set(eval_paths)),'subset_counts':dict(Counter(q['subset'] for q in queries)),'image_counts_by_subset':{s:len({q['image_id'] for q in queries if q['subset']==s}) for s in sorted({q['subset'] for q in queries})}}
(OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
report={'training_caption_alignment':inspect(train_text),'evaluation_caption_alignment':inspect([q['query'] for q in queries]),'actual_insert_special_tests':tests,'initialization':init,'checkpoint_seed':args['seed'],'human_verified_independent_test':'No such dataset identified in the current training/evaluation configuration; do not claim human validation.'}
(OUT/'alignment_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'alignment':{k:v['counts'] for k,v in report.items() if k.endswith('_alignment')},'tests':tests,'initialization':init,'dataset_sizes':manifest['dataset_sizes'],'training_files':len(train_paths),'missing':len(missing),'evaluation_files':len(eval_paths),'coco_id_overlap':manifest['coco_id_overlap'],'subset_counts':manifest['subset_counts']},indent=2),flush=True)
# Hash ALL recovered train and evaluation files; detect byte-identical images independent of names.
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
eval_hash={}
for p in eval_paths: eval_hash.setdefault(sha(p),[]).append(str(p))
hits=[]; done=0
for p in train_paths:
 if p.exists():
  digest=sha(p); done+=1
  if digest in eval_hash: hits.append({'training':str(p),'evaluation':eval_hash[digest],'sha256':digest})
 if done and done%10000==0: print('hashed',done,flush=True)
hash_report={'hashed_training_files':done,'hashed_evaluation_files':len(eval_paths),'missing_training_files':missing,'byte_identical_overlap':hits,'limitation':'SHA256 detects identical files, not resized/reencoded/derived copies. Current pool does not prove historical epoch-level membership.'}
(OUT/'hash_overlap.json').write_text(json.dumps(hash_report,indent=2)); print('HASH_RESULT',json.dumps(hash_report),flush=True)
