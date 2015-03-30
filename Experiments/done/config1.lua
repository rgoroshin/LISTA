--==misc==
exp_name='EXP1/config1'
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
 repeat_exp = 1,
 name = "FISTA",
 niter = 1000,
}

