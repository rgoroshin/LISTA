--=====List of configs to compare:===== 
--1-FISTA 
--2-LISTA (0-5 loops) 
--3-Untied weight LISTA (0-5 loops)
--4-Deep relu networks with and without tied weights (0-5 layers)
--==options==
cutorch.setDevice(2)
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
--configs 
configs = {} 
--configs[1] = {name='FISTA',niter=10,l1w=l1w} 
--LISTA with untied weights 
--configs[1] = {name='LISTA',nloops=0,untied_weights=false,l1w=l1w,learn_rate=0.1,epochs=1} 
--configs[1] = {name='LISTA',nloops=3,untied_weights=false,l1w=l1w,learn_rate=0.01,epochs=3,save_dir=save_dir} 
configs[1] = {name='LISTA',nloops=3,untied_weights=true,l1w=l1w,learn_rate=0.01,epochs=3,save_dir=save_dir} 
--configs[4] = {name='LISTA',nloops=3,untied_weights=false} 
--configs[5] = {name='LISTA',nloops=5,untied_weights=false} 
----LISTA with untied weights 
--configs[6] = {name='LISTA',nloops=0,untied_weights=true} 
--configs[7] = {name='LISTA',nloops=1,untied_weights=true} 
--configs[8] = {name='LISTA',nloops=3,untied_weights=true} 
--configs[9] = {name='LISTA',nloops=5,untied_weights=true} 
----Deep ReLU with untied weights 
--configs[10] = {name='ReLU',nloops=0,untied_weights=true} 
--configs[11] = {name='ReLU',nloops=1,untied_weights=true} 
--configs[12] = {name='ReLU',nloops=3,untied_weights=true} 
--configs[13] = {name='ReLU',nloops=5,untied_weights=true} 
----Deep ReLU with tied weights 
--configs[14] = {name='ReLU',nloops=0,untied_weights=true} 
--configs[15] = {name='ReLU',nloops=1,untied_weights=true} 
--configs[16] = {name='ReLU',nloops=3,untied_weights=true} 
--configs[17] = {name='ReLU',nloops=5,untied_weights=true} 
