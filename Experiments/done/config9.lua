--==misc==
exp_name='autoencoder_exp1/config9'
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
 untied_weights = true,
 nlayers = 5,
 recurrent = false,
 name = "LISTA",
 fix_decoder = false,
}

