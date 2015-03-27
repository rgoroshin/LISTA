dofile('init.lua') 

n = 3
nsamples=1000
l1w=0.5 
L=600
nloops=1 
niter=100 
learn_rates = torch.Tensor(n) 

if decoder == nil then 
    decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7')
    decoder:cuda()
end 

if all_data == nil then 
    all_data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
end 
data_small = all_data.datacn:resize(50000,3,32,32):narrow(1,1,nsamples)  
ds = DataSource({dataset=data_small,batchSize=16})


for i = 1,n do 
    
    inplane = decoder:get(2).weight:size(1)
    outplane = decoder:get(2).weight:size(2) 
    k = decoder:get(2).kW
    We = nn.SpatialConvolution(inplane,outplane,k,k,stride,stride) 
    We.weight:copy(flip(decoder:get(2).weight)) 
    We.bias:fill(0)
    
    encoder = construct_LISTA(We,nloops,l1w,L,false)
    learn_rate = find_learn_rate(encoder,decoder,true,ds,l1w)
    learn_rates[i] = learn_rate 

end
--Zfista = ConvFISTA(decoder,data,niter,l1w,L)
--eval_fista = eval_sparse_code(data,Zfista,decoder,l1w)
-------------------------------------------------------------

--Wd = torch.load('./Results/Trained_Networks/FISTA_Wd_32x3x9x9.t7')
--I1 = image.toDisplayTensor({input=Wd:transpose(1,2):contiguous(),nrow=8,padding=1}) 
--image.save('./Results/Experiments/decoder1.png',I1) 
--
--k = 9
--padding = (k-1)/2
--pad = nn.SpatialPadding(padding,padding,padding,padding,3,4) 
--decoder = nn.SpatialConvolution(32,3,9,9)
--decoder.bias:fill(0) 
--decoder.weight:copy(Wd)--:clone():transpose(1,2):contiguous()) 
--I2 = image.toDisplayTensor({input=decoder.weight:transpose(1,2):contiguous(),nrow=8,padding=1}) 
--image.save('./Results/Experiments/decoder2.png',I2) 
--
--net = nn.Sequential() 
--net:add(pad)
--net:add(decoder)
--I3 = image.toDisplayTensor({input=net:get(2).weight:transpose(1,2):contiguous(),nrow=8,padding=1}) 
--image.save('./Results/Experiments/decoder3.png',I3) 
--torch.save('./Results/Trained_Networks/FISTA_decoder.t7',net)
