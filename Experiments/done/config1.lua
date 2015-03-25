--==misc==
exp_name='EXP1/config1'
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
 repeat_exp = 1,
 name = "FISTA",
 niter = 10,
}

