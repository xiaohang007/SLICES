raceback (most recent call last):
  File "train_prior.py", line 73, in <module>
    pretrain()
  File "train_prior.py", line 53, in pretrain
    log_p, _ = Prior.likelihood(seqs, energies)
  File "/home/admin/Desktop/Hang/SCILES/20230613/property_optimization/3_conditional_beta/model.py", line 81, in likelihood
    logits, h = self.rnn(x[:, step], energy, h)
  File "/home/admin/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/admin/Desktop/Hang/SCILES/20230613/property_optimization/3_conditional_beta/model.py", line 27, in forward
    e = self.dense(e)
  File "/home/admin/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/admin/miniconda3/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 103, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x256 and 1x64)
