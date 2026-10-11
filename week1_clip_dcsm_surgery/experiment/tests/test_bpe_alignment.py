import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPModel, CLIPImageProcessor
from dcsm_surgery.bpe_alignment import replace_special_bpe, ALIGNMENT_VERSION
from dcsm_surgery.constants import IMG_SEQ, TEXT_SEQ, SPECIAL_WORDS
from dcsm_surgery.scorer import CrossModalConvNetwork, DCSMScorer, insert_special

class AlignmentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tok=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True)
 def emb(self,words=SPECIAL_WORDS):
  e=torch.nn.Embedding(len(words),IMG_SEQ)
  with torch.no_grad():
   for i in range(len(words)): e.weight[i].fill_(1000+i)
  return e
 def cos(self,length=TEXT_SEQ):
  return torch.arange(length,dtype=torch.float32).view(1,1,1,length).expand(2,1,IMG_SEQ,length).clone()
 def changed(self,a,b): return (a[0,0,0]!=b[0,0,0]).nonzero().flatten().tolist()
 def test_actual_training_function_bos_multibpe_prefix_repeated_words(self):
  for text,expected in [('A picture of a cat above a dog',[6]),('A picture of a cat to the left of a dog',[8]),('a picture of a traffic light below a fire hydrant',[7]),('a picture of a hippopotamus below a dog',[8]),('a picture of a cat above a dog above a bird',[6,9])]:
   with self.subTest(text=text):
    c=self.cos(); original=c.clone(); out=insert_special(c,[text],self.emb(),'cpu',tokenizer=self.tok)
    self.assertEqual(self.changed(c,out),expected); self.assertTrue(torch.equal(c,original)); self.assertEqual(out.shape,c.shape)
 def test_multibpe_lookup_replaces_every_column(self):
  text='a hippopotamus'; words=['hippopotamus']; c=self.cos(); out=replace_special_bpe(c,[text],self.emb(words),self.tok,words)
  piece=self.tok('hippopotamus',add_special_tokens=False)['input_ids']; self.assertGreater(len(piece),1)
  self.assertEqual(self.changed(c,out),list(range(2,2+len(piece))))
 def test_repeated_multibpe_lookup(self):
  text='hippopotamus and hippopotamus'; words=['hippopotamus']; c=self.cos(); out=replace_special_bpe(c,[text],self.emb(words),self.tok,words)
  self.assertEqual(len(self.changed(c,out)),2*len(self.tok('hippopotamus',add_special_tokens=False)['input_ids']))
 def test_partial_truncation_skips_whole_occurrence(self):
  text='a hippopotamus'; c=self.cos(4); out=replace_special_bpe(c,[text],self.emb(['hippopotamus']),self.tok,['hippopotamus'])
  self.assertTrue(torch.equal(c,out))
 def test_fully_truncated_lookup_skipped(self):
  text='above '+'cat '*40+'below'; c=self.cos(); out=insert_special(c,[text],self.emb(),'cpu',tokenizer=self.tok)
  self.assertEqual(self.changed(c,out),[1])
 def test_lookup_just_before_eos_replaced_without_touching_eos_padding_bos(self):
  text='cat '*27+'above'; c=self.cos(); ids=self.tok(text,padding='max_length',max_length=TEXT_SEQ,truncation=True)['input_ids']
  out=insert_special(c,[text],self.emb(),'cpu',tokenizer=self.tok); self.assertEqual(self.changed(c,out),[28]); self.assertEqual(ids[29],self.tok.eos_token_id)
  c=self.cos(); out=insert_special(c,['above'],self.emb(),'cpu',tokenizer=self.tok); self.assertEqual(self.changed(c,out),[1]); self.assertTrue(torch.equal(c[:,:,:,2:],out[:,:,:,2:])); self.assertTrue(torch.equal(c[:,:,:,0],out[:,:,:,0]))
 def test_case_punctuation_word_boundaries(self):
  text='ABOVE, nothing not! aboveboard'; c=self.cos(); out=insert_special(c,[text],self.emb(),'cpu',tokenizer=self.tok)
  ids=self.tok.convert_ids_to_tokens(self.tok(text)['input_ids']); changed=self.changed(c,out)
  self.assertEqual(len(changed),2); self.assertEqual([ids[i] for i in changed],['above</w>','not</w>'])
 def test_mismatched_input_ids_fail_loudly(self):
  c=self.cos(); ids=self.tok(['above'],padding='max_length',max_length=TEXT_SEQ,truncation=True,return_tensors='pt')['input_ids']; ids[0,1]=0
  with self.assertRaisesRegex(ValueError,'input_ids'): replace_special_bpe(c,['above'],self.emb(),self.tok,SPECIAL_WORDS,ids)
 def test_gradient_reaches_each_occurrence_embedding_not_replaced_cosine(self):
  c=self.cos().requires_grad_(); e=self.emb(); out=insert_special(c,['above cat above'],e,'cpu',tokenizer=self.tok); out.sum().backward()
  self.assertTrue(torch.all(e.weight.grad[SPECIAL_WORDS.index('above')]==4)); self.assertTrue(torch.all(c.grad[:,:,:,1]==0)); self.assertTrue(torch.all(c.grad[:,:,:,3]==0)); self.assertTrue(torch.all(c.grad[:,:,:,2]==1))
 def test_overlapping_lookup_rejected(self):
  with self.assertRaisesRegex(ValueError,'Overlapping'): replace_special_bpe(self.cos(),['to the left of'],self.emb(['left','to the left of']),self.tok,['left','to the left of'])
 def test_actual_eval_score_matches_training_replacement(self):
  device='cuda' if torch.cuda.is_available() else 'cpu'; torch.set_num_threads(4)
  clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True).to(device).eval(); proc=CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'fixture.pt'; e=self.emb(); net=CrossModalConvNetwork(IMG_SEQ,TEXT_SEQ)
   torch.save({'model_state_dict':net.state_dict(),'external_logic_token_embeddings':e,'special_slicing':ALIGNMENT_VERSION,'readout':'naive'},p)
   scorer=DCSMScorer(str(p),clip,self.tok,proc,device); capture=[]; hook=scorer.net.register_forward_pre_hook(lambda _,inp:capture.append(inp[0].detach().clone()))
   crops=[Image.new('RGB',(224,224),'red'),Image.new('RGB',(224,224),'blue')]; text='a hippopotamus above a cat above a bird'
   scores=scorer.score(crops,text); hook.remove(); self.assertTrue(np.isfinite(scores).all())
   with torch.no_grad():
    pv=proc(images=crops,return_tensors='pt')['pixel_values'].to(device); vo=clip.vision_model(pixel_values=pv)
    im=torch.nn.functional.normalize(clip.visual_projection(clip.vision_model.post_layernorm(vo.last_hidden_state)),dim=-1)
    tt=self.tok([text],padding='max_length',max_length=TEXT_SEQ,truncation=True,return_tensors='pt').to(device); txt=torch.nn.functional.normalize(clip.text_projection(clip.text_model(**tt).last_hidden_state),dim=-1)
    cos=torch.einsum('bie,lte->blit',im,txt); expected=insert_special(cos,[text],scorer.emb,device,tokenizer=self.tok,input_ids=tt['input_ids']); expected=(expected-expected.mean())/expected.std()
   torch.testing.assert_close(capture[0],expected); self.assertEqual(capture[0].shape,(2,1,IMG_SEQ,TEXT_SEQ))
   # Newly tagged fixture exercises the corrected path, not an old checkpoint relabelled as trained.

if __name__=='__main__': unittest.main(verbosity=2)
