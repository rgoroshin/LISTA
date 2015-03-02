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
configs[1] = {name='FISTA',niter=100,l1w=l1w} 
----LISTA with untied weights 
configs[2]  = {name='LISTA',nloops=0,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[3]  = {name='LISTA',nloops=1,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[4]  = {name='LISTA',nloops=3,untied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[5]  = {name='LISTA',nloops=0,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[6]  = {name='LISTA',nloops=1,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[7]  = {name='LISTA',nloops=3,untied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[8]  = {name='ReLUnet',nlayers=0,tied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[9]  = {name='ReLUnet',nlayers=1,tied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[10] = {name='ReLUnet',nlayers=3,tied_weights=false,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[11] = {name='ReLUnet',nlayers=0,tied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[12] = {name='ReLUnet',nlayers=1,tied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
configs[13] = {name='ReLUnet',nlayers=3,tied_weights=true,fix_decoder=true,l1w=l1w,learn_rate=nil,epochs=30,save_dir=save_dir} 
