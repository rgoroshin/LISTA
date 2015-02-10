dofile('init.lua') 
cutorch.setDevice(3) 
exp_name = 'Model1/'
save_dir = './Results/Experiments/'..exp_name
os.execute('mkdir -p '..save_dir) 
exp_conf = paths.thisfile() 
os.execute('cp '..exp_conf..' '..save_dir) 

--output text file 
record_file = io.open(save_dir..'output.txt', 'w') 
record_file:write('\n') 
record_file:close()

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
bsz = 16 
ds_small = DataSource({dataset = train_data:narrow(1,1,1000), batchSize = bsz})
ds_train = DataSource({dataset = train_data, batchSize = bsz})
ds_test = DataSource({dataset = test_data, batchSize = bsz})

--decoder/encoder  
Wd = torch.load('./Results/Trained_Networks/FISTA_Wd_32x3x9x9.t7') 
decoder = nn.NormSpatialConvolutionMM(outplane,inplane,k,k,1,1,padding)
encoder = nn.SpatialConvolutionMM(inplane,outplane,k,k,1,1,padding) 
decoder.weight:resize(outplane,inplane,k,k):copy(Wd)
encoder.weight:resize(inplane,outplane,k,k):copy(flip(Wd)):resize(inplane,outplane*k*k) 
decoder.weight:resize(outplane,inplane*k*k)
encoder.bias:zero()
decoder.bias:zero()
decoder:cuda() 

--settings
ds=ds_train
epochs = 5
inplane = 3 
outplane = decoder.nInputPlane 
k = decoder.kW
padding = (k-1)/2
--LISTA loops 
learn_rate = 1e-7
l1w = 0.5
--Learning curves
ave_loss_train = torch.zeros(epochs) 
ave_loss_test = torch.zeros(epochs) 

for _,nloops in ipairs({0,1,3}) do 

    print('number of loops: '..nloops) 
    --LISTA decoder
    --
    --              
    -- input-->LISTA-->(L1)-->NormLinear-->output  
    
    --full network 
    net = nn.Sequential()
    net:add(construct_LISTA(encoder,nloops,l1w,600)) 
    net:add(nn.ModuleL1Penalty(true,l1w,false)) 
    --net:add(nn.Identity())
    net:add(decoder) 
    net:cuda()

    --reconstruction criterion 
    criterion = nn.MSECriterion():cuda() 
    criterion.sizeAverage = false 
    
    for iter = 1,epochs do 
    
        epoch_loss = 0 
        epoch_sparsity = 0 
        epoch_rec_error = 0 
        sys.tic() 
    
        for i = 1, ds:size() do 
    
           progress(i,ds:size()) 
    
           X = ds:next() 
           Xr = net:forward(X)
           rec_error = criterion:forward(Xr,X)   
           Y = net:get(1).output 
          
           net:zeroGradParameters()
           rec_grad = criterion:backward(Xr,X):mul(0.5) --MSE criterion multiplies by 2 
           net:backward(X,rec_grad)
           --fixed decoder 
           net:get(3).gradWeight:fill(0) 
           net:get(3).gradBias:fill(0) 
           net:updateParameters(learn_rate) 
          
          --track loss 
          sample_rec_error = Xr:clone():add(-1,X):pow(2):mean()/X:clone():pow(2):mean()
          sample_sparsity = 1-(Y:float():gt(0):sum()/Y:nElement())
          sample_loss = 0.5*rec_error + l1w*Y:norm(1)/(bsz*outplane*32*32)  
          epoch_rec_error = epoch_rec_error + sample_rec_error
          epoch_sparsity = epoch_sparsity + sample_sparsity 
          epoch_loss = epoch_loss + sample_loss 
    
        end
    
          epoch_rec_error = epoch_rec_error/ds:size() 
          average_loss = epoch_loss/ds:size()  
          average_sparsity = epoch_sparsity/ds:size() 
          print(tostring(iter)..' Time: '..sys.toc()..' %Rec.Error '..epoch_rec_error..' Sparsity:'..average_sparsity..' Loss: '..average_loss) 
          Irec = image.toDisplayTensor({input=Xr,nrow=8,padding=1}) 
          image.save(save_dir..'Irec.png', Irec)
          Ienc = image.toDisplayTensor({input=encoder.weight:clone():resize(outplane,inplane,k,k),nrow=8,padding=1}) 
          image.save(save_dir..'enc.png', Ienc)
          Idec = image.toDisplayTensor({input=flip(decoder.weight:clone():resize(inplane,outplane,k,k)),nrow=8,padding=1}) 
          image.save(save_dir..'dec.png', Idec)
        
    
    end 

end
