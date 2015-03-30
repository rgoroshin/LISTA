--==misc==
exp_name='EXP1/config8'
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
 fix_decoder = true,
 nloops = 3,
 name = "LISTA",
}

