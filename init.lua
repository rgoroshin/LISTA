torch.setdefaulttensortype('torch.FloatTensor')
require 'cunn'
require 'image'
require 'xlua'
require 'optim'
require 'nnx' 
require 'nngraph' 
require 'unsup' 
require 'gnuplot' 
require 'data' 
dofile('./Modules/init.lua') 
dofile('util.lua') 
dofile('generateConfig.lua') 
dofile('inference.lua') 
dofile('pbar.lua') 
