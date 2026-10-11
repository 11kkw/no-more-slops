import unittest
import torch
from dcsm_surgery import scorer as train

class SharedInitializationTests(unittest.TestCase):
 def test_actual_initialization_same_for_all_readouts(self):
  torch.manual_seed(15)
  readouts=['naive','maskclip','clearclip']
  nets,embs=train.initialize_shared(readouts,'cpu')  # the initialisation train_dcsm_scorer.py uses
  for r in readouts:
   for k,v in nets['naive'].state_dict().items(): self.assertTrue(torch.equal(v,nets[r].state_dict()[k]),(r,k))
   self.assertTrue(torch.equal(embs['naive'].weight,embs[r].weight),r)
   if r!='naive':
    self.assertNotEqual(next(nets['naive'].parameters()).data_ptr(),next(nets[r].parameters()).data_ptr())
    self.assertNotEqual(embs['naive'].weight.data_ptr(),embs[r].weight.data_ptr())
if __name__=='__main__': unittest.main(verbosity=2)
