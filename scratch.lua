dofile('init.lua') 
--cutorch.setDevice(2)
dofile('EXP1/config.lua')
dofile('generateConfig.lua')
generateConfig(EXP)
arg = {1} 
dofile('EXP1/Experiment1.lua') 


