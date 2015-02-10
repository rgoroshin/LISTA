--In this first experiment, we will compare learned versus 
--iterative (FISTA) inference in a *fixed* dictionary. The
--learned encoders will be trained minimizing the lasso w.r.t.
--the parameters of the encoder, i.e.
--s.g.d. minimization of L = ||x-Df(x)|| + l1w*|f(x)| w.r.t. f()  
--=====List of encoders to compare:===== 
--1-FISTA 
--2-LISTA (0-5 loops) 
--3-Untied weight LISTA (0-5 loops)
--4-Deep relu networks with and without tied weights (0-5 layers)
--======================================
--Evaluations criteria 
--1-Lasso loss 
--2-Sparisity, reconstruction error
--=+=+=+=+=++=+=+=+=+=+=+=+=+=+=+=+=+=+=
--==options==
cutorch.setDevice(2)
torch.setnumthreads(8)
math.randomseed(123)
cutorch.manualSeed(123) 
torch.manualSeed(123) 
--loss 
l1w = 0.5 
--training 
bsz = 16 
epochs = 100
learn_rate = 0.1
--encoder 
nloops = 3 

dofile('init.lua') 

--load the data 
if train_data == nil then 
    train_data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
    train_data = train_data.datacn:resize(50000,3,32,32) 
end
if test_data == nil then 
    test_data = torch.load('./Data/CIFAR/CIFAR_CN_test.t7')
    test_data = test_data.datacn:resize(10000,3,32,32) 
end

--data sources 
ds_small = DataSource({dataset = train_data:narrow(1,1,1000), batchSize = bsz})
ds_train = DataSource({dataset = train_data, batchSize = bsz})
ds_test = DataSource({dataset = test_data, batchSize = bsz})

--load a pre-trained decoder (trained on CIFAR-training set, l1w =?, using FISTA) 
decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7') 
--encoder 
inplane = decoder:get(2).weight:size(1)
outplane = decoder:get(2).weight:size(2) 
k = decoder:get(2).kW
padding = (k-1)/2
We = nn.SpatialConvolutionFFT(inplane,outplane,k,k,stride,stride) 
We.weight:copy(flip(decoder:get(2).weight)) 
We.bias:fill(0)
L = 1.05*findMaxEigenValue(decoder)
encoder = construct_LISTA(We,nloops,l1w,L)
--training 
train_encoder_lasso(encoder,decoder,ds_train,l1w,learn_rate,epochs)
