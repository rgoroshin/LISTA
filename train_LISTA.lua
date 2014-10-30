--use CUDA by default  
torch.setnumthreads(8)
require 'cunn'
require 'mattorch'
require 'xlua' 
dofile('gpu_lock.lua') 
if nn.istaencoder == nil then 
    dofile('./Modules/LISTA.lua')
end 
--cutorch.setDevice(2)
if gpu_id == nil then 
    gpu_id = set_free_gpu(200) 
end 
torch.setdefaulttensortype('torch.FloatTensor')
---------------------
require 'image'
require 'xlua'
require 'mattorch'
require 'optim'
require 'data' 
dofile('./Modules/init.lua') 
dofile('util.lua') 
dofile('pbar.lua') 
dofile('lushio.lua') 
dofile('networks.lua')

save_path = './Results/Experiments/' 
exp_name = 'LISTA_Decoder/'
epochs = 300 
learn_rate = 0.1 
insz = 2032 
outsz = 1024 
D = 4096 
bsz = 16
l1w = 1e-8
nloops = 4
cost_table={} 
save_path = save_path..exp_name..'D_'..tostring(D)..'_LR_'..tostring(learn_rate)..'_L1_'..tostring(l1w)..'/'
os.execute('mkdir -p '..save_path) 
--copy the experiment config file to the save directory 
exp_conf = paths.thisfile() 
os.execute('cp '..exp_conf..' '..save_path) 

--load and renormalize data 
data1 = data1 or torch.load('./Data/regress_data.t7') 
data2 = data2 or torch.load('./Data/regress_whitened.t7') 

X = data1.X
Y = data2.YW:t() 
Winv = torch.inverse(data2.W:float())  

Xn = X:std(1):add(2):repeatTensor(X:size(1),1)
X = torch.cdiv(X,Xn) 

--partition the data 
ntest = 10000 
idx = torch.randperm(X:size(1)) 
idx_test = idx:narrow(1,1,ntest) 
idx_train = idx:narrow(1,ntest+1,X:size(1)-ntest)
Xtest = torch.zeros(ntest,insz)
Ytest = torch.zeros(ntest,outsz)
Xtrain = torch.zeros(X:size(1)-ntest,insz)
Ytrain = torch.zeros(Y:size(1)-ntest,outsz)
torch.save(save_path..'testdata.t7',{Xtest,Ytest})

--output text file 
record_file = io.open(save_path..'output.txt', 'w') 
record_file:write('\n') 
record_file:close()

--Learning curves
L1_error_plot_train = torch.zeros(epochs) 
rec_error_plot_train = torch.zeros(epochs) 

for i = 1,idx_test:size(1) do 

    Xtest[i]:copy(X[idx_test[i]])    
    Ytest[i]:copy(Y[idx_test[i]])    

end

for i = 1,idx_train:size(1) do 

    Xtrain[i]:copy(X[idx_train[i]])    
    Ytrain[i]:copy(Y[idx_train[i]])    

end

--LISTA decoder
--
--            (L1) 
--              |
-- input-->LISTA-->NormLinear-->output  

net = nn.Sequential()
net.cost_table = {}
net.cost_table.L1 = {} 

net:add(nn.istaencoder(insz,D,nloops)) 
net:add(nn.L1Penalty(l1w,net.cost_table)) 
net:add(nn.NormLinear(D,outsz)) 
net:cuda()

MSE = nn.MSECriterion():cuda()  

for i = 1, epochs do 

    sys.tic() 
    epoch_rec_error = 0 
    epoch_L1_error = 0 

    for j = 1, Xtrain:size(1)-bsz, bsz do 

       progress(j,Xtrain:size(1)) 

       sample = Xtrain:narrow(1,j,bsz):cuda()
       labels = Ytrain:narrow(1,j,bsz):cuda()
      
       net:zeroGradParameters()
       out = net:forward(sample)

       mse_cost = MSE:forward(out,labels)
       mse_grad = MSE:backward(out,labels)

       net:backward(sample,mse_grad)
       net:updateParameters(learn_rate) 

       epoch_rec_error = epoch_rec_error + mse_cost 
       epoch_L1_error = epoch_L1_error + net.cost_table.L1[1] 

    end

    epoch_rec_error = epoch_rec_error / Xtrain:size(1) 
    epoch_L1_error = epoch_L1_error / Xtrain:size(1) 
    
    --Plot error history (throw away first iter for scale) 
    L1_error_plot_train[i] = epoch_L1_error 
    rec_error_plot_train[i] = epoch_rec_error 
    
    --plot the losses  
    gnuplot.plot({'L1', L1_error_plot_train, '+'}, {'rec train', rec_error_plot_train, '+'}) 
    gnuplot.figprint(save_path..'error_history.eps')     

    output = tostring(iter)..' TIME:'..tostring(sys.toc())..'\n'
    output = output..'Rec Error: '..tostring(epoch_rec_error)..'\n' 
    output = output..'L1 Error:  '..tostring(epoch_L1_error)..'\n' 
    record_file = io.open(save_path..'output.txt', 'a') 
    record_file:write(output..'\n') 
    record_file:close() 
   
    --whiten 
    Wd = net:get(3).weight:float()
    Wd = torch.mm(Winv,Wd):t() 
    local dec_image = image_linear(Wd,32,1)  
    image.save(save_path..(init or '')..'decoder'..'.png',dec_image) 
    
    print(output) 

end 
--
--
----sample = X:narrow(1,1,100) 
----rec = transform_data(sample,net) 
----I = rec:reshape(100,32,32) 
----I = image.toDisplayTensor{nrow=10,scaleeach=true,input=I,padding=1,symmetric=true}  
----image.save('./Results/reconstruction.png',I) 
----
----It = Y:narrow(1,1,100):reshape(100,32,32) 
----It = image.toDisplayTensor{nrow=10,scaleeach=true,input=It,padding=1,symmetric=true}  
----image.save('./Results/truth.png',It) 
----
----gnuplot.plot(Xs:squeeze())
----gnuplot.figprint('./Results/std.eps')
--
--
--
--
--
--
--
--
