--=====List of configs to compare:===== 
--1-FISTA 
--2-LISTA (0-5 loops) 
--3-Untied weight LISTA (0-5 loops)
--4-Deep relu networks with and without tied weights (0-5 layers)
--==options==
EXP = {}
EXP.config_dir = './Experiments/config/'
EXP.exp_name = {'autoencoder_exp1'}
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
EXP.resolution = {5} --hard-coded 
EXP.depth = {2} --hard-coded 
EXP.rand_grid = {true} --hard-coded  
--epochs 
EXP.epochs = {20}  
--repeat experiments 
EXP.repeat_exp = {5}  
--configs 
configs = {} 
configs[1] = {name='FISTA',repeat_exp=1,niter=1000} 
----LISTA with untied weights 
configs[2]  = {name='LISTA',nlayers=0,recurrent=false,untied_weights=false,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[3]  = {name='LISTA',nlayers=1,recurrent=false,untied_weights=false,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[4]  = {name='LISTA',nlayers=3,recurrent=false,untied_weights=false,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[5]  = {name='LISTA',nlayers=5,recurrent=false,untied_weights=false,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[6]  = {name='LISTA',nlayers=0,recurrent=false,untied_weights=true,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[7]  = {name='LISTA',nlayers=1,recurrent=false,untied_weights=true,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[8]  = {name='LISTA',nlayers=3,recurrent=false,untied_weights=true,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[9]  = {name='LISTA',nlayers=5,recurrent=false,untied_weights=true,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[10] = {name='ReLU',nlayers=0,recurrent=false,untied_weights=false,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[11] = {name='ReLU',nlayers=1,recurrent=false,untied_weights=false,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[12] = {name='ReLU',nlayers=3,recurrent=false,untied_weights=false,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[13] = {name='ReLU',nlayers=5,recurrent=false,untied_weights=false,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[14] = {name='ReLU',nlayers=0,recurrent=false,untied_weights=true,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[15] = {name='ReLU',nlayers=1,recurrent=false,untied_weights=true,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[16] = {name='ReLU',nlayers=3,recurrent=false,untied_weights=true,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir} 
configs[17] = {name='ReLU',nlayers=5,recurrent=false,untied_weights=true,fix_decoder=false,learn_rate=learn_rate,epochs=train_epochs,save_dir=save_dir}
EXP.configs = configs 
