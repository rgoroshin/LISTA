--==misc==
exp_name='autoencoder_exp1/config11'
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
 untied_weights = false,
 nlayers = 1,
 recurrent = false,
 name = "ReLU",
 fix_decoder = false,
}

