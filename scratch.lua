dofile('init.lua')
cutorch.setDevice(4) 

m = nn.ParallelCriterion() 
x = {} 
for i = 1,3 do 
    x[i] = torch.ones(5) 
    local crit = nn.L1Cost() 
    crit.sizeAverage = false 
    m:add(crit)
end
m.repeatTarget = true 
target = x[1]:clone():zero() 

out = m:forward(x,target) 



--x = nn.Identity()() 
--y1,y2 = nn.ConcatTable():add(nn.Identity()):add(nn.Identity())(x):split(2)
--net = nn.gModule({x},{y1,y2}) 
--
--x = torch.rand(2) 
--y = net:forward(x) 















--x = nn.Identity()()
--x1 = nn.Linear(5,2)(x)
--x2 = nn.Linear(2,10)(x1) 
--m = nn.gModule({x},{x1,x2}) 
--
--x = torch.ones(5)
--y = m:forward(x) 

--infer_ISTA = function(X,decoder,niter,l1w,L)  
--    local inplane = decoder:get(2).weight:size(1)
--    local outplane = decoder:get(2).weight:size(2) 
--    local Zprev = torch.zeros(X:size(1),outplane,X:size(3),X:size(4)):cuda()
--    local Z = Zprev:clone() 
--    local threshold = nn.Threshold(0,0):cuda()  
--    --ISTA inference iterates
--    Zprev:fill(0)
--    for i = 0,niter do 
--        --ISTA
--        Xerr = decoder:forward(Zprev):add(-1,X)  
--        dZ = decoder:backward(Zprev,Xerr)   
--        Zprev:add(-1/L,dZ)  
--        Z:copy(Zprev) 
--        Z:add(-l1w/L) 
--        Z:copy(threshold:forward(Z))
--        Zprev:copy(Z)
--    end
--    return Z 
--end
-----n = 3
--nsamples=1000
--l1w=0.5 
--L=600
--nloops=1 
--niter=100 
--
--if decoder == nil then 
--    decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7')
--    decoder:cuda()
--end 
--inplane = decoder:get(2).weight:size(1)
--outplane = decoder:get(2).weight:size(2) 
--k = decoder:get(2).kW
--We = nn.SpatialConvolution(inplane,outplane,k,k,stride,stride) 
--We.weight:copy(flip(decoder:get(2).weight)) 
--We.bias:fill(0)
--
--
--if all_data == nil then 
--    all_data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
--end 
--data_small = all_data.datacn:resize(50000,3,32,32):narrow(1,1,nsamples)  
--ds = DataSource({dataset=data_small,batchSize=16})
--
----the network 
--net1 = construct_LISTA(We,nloops,l1w,L,false):cuda() 
--net2 = construct_recurrent_LISTA(We,nloops,l1w,L,false):cuda() 
--
----output
--sample = ds:next() 
--out1 = net1:forward(sample) 
--out2 = net2:forward(sample) 
--
--if type(out2) == 'table' then 
--    err = out1:float():add(-1,out2[#out2]:float()):norm()
--else     
--    err = out1:float():add(-1,out2:float()):norm()
--end
----print(err) 
--
--code = infer_ISTA(sample,decoder,nloops,l1w,L)  
----code = iter_infer(decoder,sample,nloops,l1w,L,'ISTA')
--err_out1 = code:float():add(-1,out1:float())
--err_out2 = code:float():add(-1,out2[#out2]:float())
--print(err_out1:norm()) 
--print(err_out2:norm()) 

--[[
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
--torch.save('./Results/Trained_Networks/FISTA_decoder.t7',net)--]]


