--==misc==
exp_name='EXP1/config11'
small_exp=true
--==dataset==
dataset='CIFAR_CN'
--==loss==
l1w=0.5
--==learning==
epochs=3
bsz=16
repeat_exp=2
--==architectures==
arch={
 untied_weights = false,
 nlayers = 1,
 fix_decoder = true,
 name = "ReLU",
}

