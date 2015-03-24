--=====List of configs to compare:===== 
--1-FISTA 
--2-LISTA (0-5 loops) 
--3-Untied weight LISTA (0-5 loops)
--4-Deep relu networks with and without tied weights (0-5 layers)
--==options==
EXP = {}
EXP.config_dir = './Experiments/config/'
EXP.exp_name = {'EXP1'}
EXP.small_exp = {false}
--dataset 
EXP.dataset = {'CIFAR_CN'}
--loss 
EXP.l1w = {0.5} 
--training batch size  
EXP.bsz = {16} 
--learning rate 
EXP.learn_rate = {nil} 
--learning rate hyper-optimization 
EXP.resolution = {10}
EXP.depth = {2}
EXP.rand_grid = {true} 
--epochs 
EXP.epochs = {30}  
--repeat experiments 
EXP.repeat_exp = {10}  
--configs 
configs = {} 
configs[1] = {name='FISTA',repeat_exp=1,niter=1000,l1w=l1w} 
----LISTA with untied weights 
configs[2]  = {name='LISTA',nloops=0,untied_weights=false,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[3]  = {name='LISTA',nloops=1,untied_weights=false,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[4]  = {name='LISTA',nloops=3,untied_weights=false,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[5]  = {name='LISTA',nloops=5,untied_weights=false,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[6]  = {name='LISTA',nloops=0,untied_weights=true,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[7]  = {name='LISTA',nloops=1,untied_weights=true,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[8]  = {name='LISTA',nloops=3,untied_weights=true,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[9]  = {name='LISTA',nloops=5,untied_weights=true,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[10] = {name='ReLU',nlayers=0,untied_weights=false,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[11] = {name='ReLU',nlayers=1,untied_weights=false,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[12] = {name='ReLU',nlayers=3,untied_weights=false,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[13] = {name='ReLU',nlayers=5,untied_weights=false,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[14] = {name='ReLU',nlayers=0,untied_weights=true,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[15] = {name='ReLU',nlayers=1,untied_weights=true,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[16] = {name='ReLU',nlayers=3,untied_weights=true,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[17] = {name='ReLU',nlayers=5,untied_weights=true,fix_decoder=true,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir}
EXP.configs = configs 
