--=====List of configs to compare:===== 
--1-FISTA 
--2-LISTA (0-5 loops) 
--3-Untied weight LISTA (0-5 loops)
--4-Deep relu networks with and without tied weights (0-5 layers)
--==options==
cutorch.setDevice(1)
torch.setnumthreads(8)
math.randomseed(123)
cutorch.manualSeed(123) 
torch.manualSeed(123)
exp_name = 'EXP1' 
save_dir = './Results/Experiments/'..exp_name..'/' 
--loss 
l1w = 0.5 
--training batch size  
bsz = 16
--learning rate 
learn_rate = nil 
--learning rate hyper-optimization 
resolution = 10
depth = 2
--epochs 
train_epochs = 30 
--repeat experiments 
repeat_exp = 5 
--configs 
configs = {} 
configs[1] = {name='FISTA',repeat_exp=1,niter=500,l1w=l1w} 
----LISTA with untied weights 
configs[2]  = {name='LISTA',nloops=0,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[3]  = {name='LISTA',nloops=1,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[4]  = {name='LISTA',nloops=3,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[5]  = {name='LISTA',nloops=0,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[6]  = {name='LISTA',nloops=1,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[7]  = {name='LISTA',nloops=3,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[8]  = {name='ReLU',nlayers=0,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[9]  = {name='ReLU',nlayers=1,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[10] = {name='ReLU',nlayers=3,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[11] = {name='ReLU',nlayers=0,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[12] = {name='ReLU',nlayers=1,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
configs[13] = {name='ReLU',nlayers=3,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir,repeat_exp=repeat_exp} 
