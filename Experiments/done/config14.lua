--==misc==
exp_name='EXP1/config14'
small_exp=false
--==dataset==
dataset='CIFAR_CN'
--==loss==
l1w=0.5
--==learning==
epochs=30
bsz=16
repeat_exp=10
--==architectures==
arch={
 untied_weights = true,
 nlayers = 0,
 fix_decoder = true,
 name = "ReLU",
}

