dofile('init.lua') 

bsz = 16
l1w = 0.5 
L = 600
niter = 100 

decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7')
decoder:cuda()

train_data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
train_data = train_data.datacn:resize(50000,3,32,32):narrow(1,1,1000)  

Ztrain = ConvFISTA(decoder,train_data,niter,l1w,L)
eval = eval_sparse_code(train_data,Ztrain,decoder,l1w) 
