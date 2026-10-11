"""Regression test for the BPE-aligned Functional-Row replacement (failed on the original whitespace-index rule)."""
import unittest
import torch
from transformers import CLIPTokenizer
from dcsm_surgery.constants import IMG_SEQ, TEXT_SEQ, SPECIAL_WORDS
from dcsm_surgery.scorer import insert_special

class AlignmentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tok=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True)
  cls.emb=torch.nn.Embedding(len(SPECIAL_WORDS),IMG_SEQ)
  with torch.no_grad():
   for i in range(len(SPECIAL_WORDS)): cls.emb.weight[i].fill_(1000+i)
 def test_replacement_targets_actual_bpe_columns_and_all_occurrences(self):
  cases=['A picture of a cat above a dog','A picture of a cat to the left of a dog','a picture of a traffic light below a fire hydrant','a picture of a hippopotamus below a dog','a picture of a cat above a dog above a bird']
  for text in cases:
   with self.subTest(text=text):
    cos=torch.arange(TEXT_SEQ,dtype=torch.float32).view(1,1,1,TEXT_SEQ).expand(1,1,IMG_SEQ,TEXT_SEQ).clone()
    out=insert_special(cos,[text],self.emb,'cpu')
    tokens=self.tok.convert_ids_to_tokens(self.tok(text)['input_ids'])
    expected=[i for i,t in enumerate(tokens[:TEXT_SEQ]) if t in [w+'</w>' for w in SPECIAL_WORDS]]
    changed=(out[0,0,0]!=cos[0,0,0]).nonzero().flatten().tolist()
    self.assertEqual(changed,expected)
if __name__=='__main__': unittest.main(verbosity=2)
