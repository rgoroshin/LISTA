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

--data sources 
bsz = 16 
ds_small = DataSource({dataset = train_data:narrow(1,1,1000), batchSize = bsz})
ds_train = DataSource({dataset = train_data, batchSize = bsz})
ds_test = DataSource({dataset = test_data, batchSize = bsz})

X = ds_train:next() 
Z = ConvFISTA(decoder,ds_train.data[1],30,0.5,600) 
