--==misc==
exp_name='autoencoder_exp1/config1'
small_exp=false
--==dataset==
dataset='CIFAR_CN'
--==loss==
l1w=0.5
--==learning==
epochs=20
bsz=16
repeat_exp=5
--==architectures==
arch={
 repeat_exp = 1,
 name = "FISTA",
 niter = 1000,
}

