dofile('init.lua') 

l1w=0.5 
L=600
niter=100 

if decoder == nil then 
    decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7')
    decoder:cuda()
end 

if data == nil then 
    data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
    data = data.datacn:resize(50000,3,32,32):narrow(1,1,1000)  
end 

Zfista = ConvFISTA(decoder,data,niter,l1w,L)
eval_fista = eval_sparse_code(data,Zfista,decoder,l1w)
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
