dofile('init.lua') 
cutorch.setDevice(1) 
exp_name = 'LISTA_LASSO/'
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

--decoder 
decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7') 

--settings
ds=ds_small
epochs = 5
inplane = 3 
outplane = decoder:get(2).nInputPlane 
k = decoder:get(2).kW
stride = 1
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
    --            (L1) 
    --              |
    -- input-->LISTA-->NormLinear-->output  
    
    --encoder
    encoder = nn.SpatialConvolutionFFT(inplane, outplane, k, k, stride, stride) 
    encoder.weight:copy(flip(decoder:get(2).weight)) 
    encoder.bias:fill(0)
    
    --full network 
    net = nn.Sequential()
    net:add(construct_LISTA(encoder,nloops,l1w,600)) 
    net:add(nn.L1Penalty(true,l1w,false)) 
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
           net:get(3):get(2).gradWeight:fill(0) 
           net:get(3):get(2).gradBias:fill(0) 
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
          Ienc = image.toDisplayTensor({input=encoder.weight:float(),nrow=8,padding=1}) 
          image.save(save_dir..'enc.png', Ienc)
        
    
    end 

end
