--==misc==
exp_name='Interactive'
small_exp=true
--==dataset==
dataset='CIFAR_CN'
--==loss==
l1w=0.5
--==learning==
learn_rate = 2e-7 
epochs=3
bsz=16
repeat_exp=2
--==architectures==
arch={
 untied_weights = false,
 nlayers = 0,
 fix_decoder = true,
 name = "LISTA",
}

