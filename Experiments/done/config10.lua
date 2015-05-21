--==misc==
exp_name='compare_all/config10'
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
 nlayers = 0,
 fix_decoder = true,
 name = "ReLU",
}

