import ast
import types
import unittest
from pathlib import Path
import torch
import train_dcsm_scorer as train
from compositional_eval import CrossModalConvNetwork, IMG_SEQ, TEXT_SEQ

class SharedInitializationTests(unittest.TestCase):
 def test_actual_initialization_same_for_all_readouts(self):
  torch.manual_seed(15)
  readouts=['naive','maskclip','clearclip']
  if hasattr(train,'initialize_shared'):
   nets,embs=train.initialize_shared(readouts,'cpu')
  else:
   # Execute the exact current initialization statements from main, not a mock.
   tree=ast.parse(Path(train.__file__).read_text()); main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
   nodes=[x for x in main.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('nets','embs') for t in x.targets)]
   ns=dict(vars(train),args=types.SimpleNamespace(readouts=readouts),device='cpu')
   exec(compile(ast.Module(body=nodes,type_ignores=[]),str(train.__file__),'exec'),ns); nets,embs=ns['nets'],ns['embs']
  for r in readouts:
   for k,v in nets['naive'].state_dict().items(): self.assertTrue(torch.equal(v,nets[r].state_dict()[k]),(r,k))
   self.assertTrue(torch.equal(embs['naive'].weight,embs[r].weight),r)
   if r!='naive':
    self.assertNotEqual(next(nets['naive'].parameters()).data_ptr(),next(nets[r].parameters()).data_ptr())
    self.assertNotEqual(embs['naive'].weight.data_ptr(),embs[r].weight.data_ptr())
if __name__=='__main__': unittest.main(verbosity=2)
