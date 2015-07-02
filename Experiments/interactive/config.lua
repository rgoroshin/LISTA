--==misc==
exp_name='interactive'
small_exp=true
--==dataset==
dataset='CIFAR_CN'
--==loss==
l1w=0.5
--==learning==
learn_rate = 7e-8
epochs=20
bsz=16
repeat_exp=5
--==architectures==
arch={
 untied_weights = true,
 nlayers = 3,
 recurrent = false,
 name = "FISTA",
 niter = 100,
 repeat_exp = 1, 
 fix_decoder = false,
}

