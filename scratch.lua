dofile('init.lua') 
cutorch.setDevice(2)

decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7') 

--load the data 
if train_data == nil then 
    train_data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
    train_data = train_data.datacn:resize(50000,3,32,32) 
end
if test_data == nil then 
    test_data = torch.load('./Data/CIFAR/CIFAR_CN_test.t7')
    test_data = test_data.datacn:resize(10000,3,32,32) 
end


--inplane = decoder:get(2).weight:size(1)
--outplane = decoder:get(2).weight:size(2) 
--k = decoder:get(2).kW
--We = nn.SpatialConvolutionFFT(inplane,outplane,k,k,stride,stride) 
--We.weight:copy(flip(decoder:get(2).weight)) 
--We.bias:fill(0)
--LISTA = construct_LISTA(We,config.nloops,config.l1w,config.L,config.untied_weights)
net = construct_deep_net(3,3,32,9,true) 

--data sources 
--config = {nloops=3,l1w=0.5,L=600,untied_weights=false} 
--bsz = 16 
--ds_small = DataSource({dataset = train_data:narrow(1,1,300), batchSize = bsz})
--ds_train = DataSource({dataset = train_data, batchSize = bsz})
--ds_test = DataSource({dataset = test_data, batchSize = bsz})
--
--decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7') 
--inplane = decoder:get(2).weight:size(1)
--outplane = decoder:get(2).weight:size(2) 
--k = decoder:get(2).weight:size(3) 
--We = nn.SpatialConvolutionFFT(inplane,outplane,k,k,stride,stride) 
--We.weight:copy(flip(decoder:get(2).weight)) 
--We.bias:fill(0)
--encoder = construct_LISTA(We,config.nloops,config.l1w,config.L,config.untied_weights)
--
--learn_rate = find_learn_rate(encoder,decoder,ds_small,config.l1w)
--encoder,loss_plot = train_encoder_lasso(encoder,decoder,ds_train,config.l1w,config.learn_rate,config.epochs,config.save_dir)



--X = ds_train:next() 
--Z = ConvFISTA(decoder,ds_train.data[1],30,0.5,600) 
--eval_sparse_code = function(data,codes,decoder,_l1w)
--    local n = 256
--    local ave_loss = 0 
--    local ave_sparsity = 0 
--    local ave_percent_rec_error = 0 
--    local criterion = nn.MSECriterion():cuda() 
--    --data chunk
--    local X = torch.CudaTensor(n,data:size(2),data:size(3),data:size(4))
--    --code chunk 
--    local Z = torch.CudaTensor(n,data:size(2),data:size(3),data:size(4))
--    --evaluation for entire dataset in chunks 
--    for i = 0,data:size(1)-n,n do 
--        progress(i,data:size(1))
--        X:copy(data:narrow(1,i+1,n))
--        Z:copy(codes:narrow(1,i+1,n))
--        local Xr = decoder:forward(Z)
--        local rec_error = criterion:forward(Xr,X)   
--        local percent_rec_error = Xr:clone():add(-1,X):pow(2):mean()/X:clone():pow(2):mean()
--        local sparsity = 1-(Z:float():gt(0):sum()/Z:nElement())
--        local loss = 0.5*rec_error + _l1w*Z:norm(1)/Z:nElement()  
--        ave_percent_rec_error = ave_percent_rec_error + percent_rec_error
--        ave_sparsity = ave_sparsity + sparsity 
--        ave_loss = ave_loss + loss 
--    end
--    --process the remaining data 
--    local a = math.floor(data:size(1)/n)*n+1
--    local b = math.fmod(data:size(1),n)
--    if a < data:size(1) then 
--        X:copy(data:narrow(1,i+1,b))
--        Z:copy(codes:narrow(1,i+1,b))
--        local Xr = decoder:forward(Z)
--        local rec_error = criterion:forward(Xr,X)   
--        local percent_rec_error = Xr:clone():add(-1,X):pow(2):mean()/X:clone():pow(2):mean()
--        local sparsity = 1-(Z:float():gt(0):sum()/Z:nElement())
--        local loss = 0.5*rec_error + _l1w*Z:norm(1)/Z:nElement()  
--        ave_percent_rec_error = ave_percent_rec_error + percent_rec_error
--        ave_sparsity = ave_sparsity + sparsity 
--        ave_loss = ave_loss + loss 
--    end 
--        ave_percent_rec_error = ave_percent_rec_error/data:size(1)
--        ave_sparsity = ave_sparsity/data:size(1) 
--        ave_loss = ave_loss/data:size(1) 
--
--end
