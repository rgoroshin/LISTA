--use CUDA by default  
torch.setnumthreads(8)
require 'cunn'
require 'nnx' 
require 'xlua' 
--if gpu_id == nil then 
--    gpu_id = set_free_gpu(500) 
--end 
dofile('init.lua') 
cutorch.setDevice(2)
torch.setdefaulttensortype('torch.FloatTensor')
---------------------
require 'image'
require 'xlua'
require 'optim'

reverse = function(x) 
    local n = x:size(1) 
    for i = 1,math.floor(n/2) do 
        local tmp = x[i]
        x[i] = x[n-i+1]
        x[n-i+1] = tmp
    end 
    return x 
end

 flip = function(W) 
   local Wt = W:clone():transpose(1,2)  
    for i = 1,Wt:size(1) do
        for j = 1,Wt:size(2) do  
            for n = 1,Wt:size(3) do 
                reverse(Wt[i][j][n])
            end
            for n = 1,Wt:size(4) do 
                reverse(Wt[i][j]:select(2,n))
            end
        end
    end 
    return Wt  
end 

insz = 10 
inplane = 3 
outplane = 16
k = 5
stride = 1 
pading = (k-1)/2
L = 100

print('==========Test 1==========')
-------------------------------------------------------------
--Check that Sy = y-WWty 
--(Note: this test uses [bsz] = 1 because conv2 doesn't 
--support batches 
-------------------------------------------------------------
encoder = nn.SpatialConvolution(inplane,outplane,k,k) 
We = flip(encoder.weight) 
--dimensions (assume square, odd-sized kernels and stride = 1) 
inplane = We:size(1) 
outplane = We:size(2)
k = We:size(3)
padding = (k-1)/2
pad = nn.SpatialPadding(padding,padding,padding,padding,2,3) 
pad2 = nn.SpatialPadding(2*padding,2*padding,2*padding,2*padding,2,3) 
 
Sw = torch.Tensor(outplane,outplane,2*k-1,2*k-1) 

for i = 1,outplane do
    Sw[i] = torch.conv2(We:select(2,i),flip(We),'F') 
end 

local I = torch.zeros(outplane,outplane,2*k-1,2*k-1) 

for i = 1,outplane do 
    I[i][i][math.ceil(k-0.5)][math.ceil(k-0.5)] = 1 
end

S = nn.Sequential() 
Sw = I - Sw
S:add(pad2:clone()) 
S:add(nn.SpatialConvolution(outplane,outplane,2*k-1,2*k-1))
S:get(2).weight:copy(Sw) 
S:get(2).bias:fill(0) 

encoder.bias:fill(0)
encoder_same = nn.Sequential() 
encoder_same:add(pad:clone()) 
encoder_same:add(encoder) 
encoder = encoder_same 

--Check that the S-operator is correct  
x = torch.rand(inplane,insz,insz)
y = encoder:forward(x) 

--full followed by valid conv
Wty = torch.conv2(y,flip(encoder:get(2).weight),'F') 
WWty = torch.conv2(Wty,encoder:get(2).weight,'V') 
out1 = y - WWty 
--same conv
out2 = S:forward(y) 
print('Discrepancy: '..tostring((out1-out2):norm()))
------------------------------------------------------------
bsz = 16 
nloops1 = 10 
nloops2 = 20
l1w = 0.5 
L = 600
x = torch.rand(bsz,inplane,insz,insz):cuda() 

pad = nn.SpatialPadding(padding,padding,padding,padding,3,4) 
pad2 = nn.SpatialPadding(2*padding,2*padding,2*padding,2*padding,3,4) 

encoder = nn.SpatialConvolution(inplane,outplane,k,k) 
LISTA1 = construct_LISTA(encoder,nloops1,nil,L) 
LISTA2 = construct_LISTA(encoder,nloops2,nil,L) 
LISTA0 = nn.Sequential() 
LISTA0:add(pad:clone())
LISTA0:add(LISTA1.encoder:clone()) 
LISTA0:add(nn.Threshold()) 
LISTA0:cuda() 

decoder = nn.Sequential() 
decoder:add(pad:clone()) 
decoder:add(nn.SpatialConvolution(outplane,inplane,k,k))
decoder:get(2).weight:copy(flip(encoder.weight)) 
decoder:get(2).bias:fill(0) 
decoder:cuda() 

net0 = nn.Sequential() 
net0:add(LISTA0) 
net0:add(decoder:clone()) 
net0:cuda()

net1 = nn.Sequential() 
net1:add(LISTA1) 
net1:add(decoder:clone()) 
net1:cuda()

net2 = nn.Sequential() 
net2:add(LISTA2) 
net2:add(decoder:clone()) 
net2:cuda()

xr0 = net0:forward(x) 
xr1 = net1:forward(x) 
xr2 = net2:forward(x) 

rec_error0 = 0.5*((x:float()-xr0:float()):pow(2):mean()) 
rec_error1 = 0.5*((x:float()-xr1:float()):pow(2):mean()) 
rec_error2 = 0.5*((x:float()-xr2:float()):pow(2):mean()) 

l1_error0 = LISTA0.output:abs():mean() 
l1_error1 = LISTA1.output:abs():mean() 
l1_error2 = LISTA2.output:abs():mean() 

codeSize = LISTA0.output:float():numel() 

error0 = rec_error0 + l1w*l1_error0 
error1 = rec_error1 + l1w*l1_error1
error2 = rec_error2 + l1w*l1_error2

sparsity0 = LISTA0.output:float():gt(0):sum()
sparsity1 = LISTA1.output:float():gt(0):sum()
sparsity2 = LISTA2.output:float():gt(0):sum()

print('Error 0: '.. tostring(error0))  
print('Error 1: '.. tostring(error1))  
print('Error 2: '.. tostring(error2))  

print('Activity 0: '.. tostring(sparsity0))  
print('Activity 1: '.. tostring(sparsity1))  
print('Activity 2: '.. tostring(sparsity2))  

----------------------------------------------------
print('==========Test 2==========')
--verify that n-ISTA iterations 
--produces the same loss as the 
--fprop on an (n-1)-loop LISTA 
--network. There will be precision 
--errors because of S-operator 

--load the data
if data == nil then 
    data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
    data = data.datacn:resize(50000,3,32,32) 
end

bsz = 16
ds_train = DataSource({dataset = data, batchSize = bsz})
ds_small = DataSource({dataset = data:narrow(1,1,800), batchSize = bsz})

ds = ds_train
inplane = 3 
outplane = 16
k = 5
stride = 1
padding = (k-1)/2
--1/(code learning rate)  
L = 600
--dictionary learning rate 
l1w = 0.5
--a sample 
X = ds_train:next() 

for nloops = 1,5 do 
    --=====initialize componenets=====  
    --decoder 
    decoder = nn.Sequential() 
    ConvDec = nn.NormSpatialConvolution(outplane, inplane, k, k, stride, stride) 
    ConvDec.bias:fill(0) 
    decoder:add(nn.SpatialPadding(padding,padding,padding,padding,3,4))
    decoder:add(ConvDec) 
    decoder:cuda() 
    --encoder
    encoder = nn.Sequential() 
    ConvEnc = nn.SpatialConvolution(inplane, outplane, k, k, stride, stride) 
    ConvEnc.weight:copy(flip(ConvDec.weight)) 
    ConvEnc.bias:fill(0)
    encoder:add(nn.SpatialPadding(padding,padding,padding,padding,3,4))
    encoder:add(ConvEnc) 
    encoder:cuda() 
    --LISTA 
    LISTA = construct_LISTA(ConvEnc:clone():float(),nloops,l1w,L)
    LISTAr = construct_recurrent_LISTA(ConvEnc:clone():float(),nloops,l1w,L):cuda()
    --Thresholding-operator 
    threshold = nn.Threshold(0,0):cuda()  
    --Initial code 
    Zprev = torch.zeros(bsz,outplane,32,32):cuda()
    Z = Zprev:clone() 
    --Reconstruction Criterion 
    criterion = nn.MSECriterion():cuda() 
    
    -- ISTA inference 
    for i = 0,nloops do 
        Xerr = decoder:forward(Zprev):add(-1,X)  
        dZ = decoder:backward(Zprev,Xerr)  
        Zprev:add(-1/L,dZ) 
        Z:copy(Zprev) 
        Z:add(-l1w/L) 
        Z:copy(threshold:forward(Z))
        Zprev:copy(Z) 
    end
    --code error 
    Zlista = LISTA:forward(X)
    Zlistar = LISTAr:forward(X)
    --Z = Z:narrow(3,10,10):narrow(4,10,10) 

    code_error = Zlista:float():add(-1,Z:float()):norm()
    if type(Zlistar) == 'table' then 
        code_error2 = Zlistar[#Zlistar]:float():add(-1,Z:float()):norm()
    else 
        code_error2 = Zlistar:float():add(-1,Z:float()):norm()
    end

    Xr = decoder:forward(Z)
    rec_error = criterion:forward(Xr,X) 
    sample_loss = 0.5*rec_error + l1w*Z:abs():mean() 
    print('==========nloops = '..nloops..'==========')
    print(tostring(nloops)..'-L2 code error1  '..tostring(code_error))
    print(tostring(nloops)..'-L2 code error2  '..tostring(code_error2))
    print(tostring(nloops)..'-iter-ISTA Loss  '..tostring(sample_loss))
    
    net = nn.Sequential() 
    net.cost_table = {} 
    net.cost_table.L1 = {} 
    net:add(LISTA)
    net:add(nn.ModuleL1Penalty(true,l1w,true)) 
    net:add(decoder:clone()) 
    net:cuda()
    
    Xr2 = net:forward(X)
    rec_error2 = criterion:forward(Xr2,X) 
    Z2 = LISTA.output:float() 
    sample_loss2 = 0.5*rec_error2 + net:get(2).L1Cost[1] 
    print(tostring(nloops)..'-loop-LISTA Loss '..tostring(sample_loss2))
end
print('==========Test 3==========')
--compare loss obtainted with FISTA 
--iterations and trained LISTA networks 
--this uses the newly training written 
--code for the LISTA project  

--(1) load a pretrained decoder 
--decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7')
--decoder:cuda()
----(2) set params 
--l1w=0.5 
--L=600
--niter=100
--nloops=3
--learn_rate=0.5e-7 
--epochs=20
--data_small = data:narrow(1,1,1000) 
----(3) run FISTA inference & eval codes 
--if eval_fista == nil then 
--    Zfista = ConvFISTA(decoder,data_small,niter,l1w,L)
--    eval_fista = eval_sparse_code(data_small,Zfista,decoder,l1w)
--end 
--print(eval_fista) 
----(4) train LISTA  
--inplane = decoder:get(2).weight:size(1)
--outplane = decoder:get(2).weight:size(2) 
--k = decoder:get(2).kW
--We = nn.SpatialConvolution(inplane,outplane,k,k,stride,stride) 
--We.weight:copy(flip(decoder:get(2).weight)) 
--We.bias:fill(0)
--encoder = construct_LISTA(We,nloops,l1w,L,false)
--ds = DataSource({dataset=data_small,batchSize=16})
--print('training LISTA..') 
--encoder,loss_plot = minimize_lasso_sgd(encoder,decoder,true,ds,l1w,learn_rate,epochs,nil)
--Zlista = transform_data(data_small,encoder)
--eval_lista = eval_sparse_code(data_small,Zlista,decoder,l1w)
--print(eval_lista) 
--[[
--==============================
print('==========Test 3==========')
--Compute the sparse codes using ISTA inference
--for a fixed decoder found using Sparse Coding 
--and regress on the codes with varying number 
--of loops of LISTA. Show that adding more loops
--improves performance

--load the train data and codes 
--if FISTA_train_data == nil then 
--    FISTA_train_data = torch.load('./Results/Trained_Networks/FISTA_data_codes16.t7') 
--end

--load the decoder 
if decoder == nil then 
    decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7')
    decoder:cuda()
end 
--decoder = torch.load('./Results/Trained_Networks/AE_vs_SC/FISTA_decoder16.t7') 
    
--otherwise compute them using FISTA 
if FISTA_train_data == nil then 
   
    --encoder
    encoder = nn.Sequential() 
    ConvEnc = nn.SpatialConvolution(inplane, outplane, k, k, stride, stride) 
    ConvEnc.weight:copy(flip(decoder:get(2).weight)) 
    ConvEnc.bias:fill(0)
    encoder:add(nn.SpatialPadding(padding,padding,padding,padding,3,4))
    encoder:add(ConvEnc) 
    encoder:cuda() 
    
    --Thresholding-operator 
    threshold = nn.Threshold(0,0):cuda()  
    
    --Reconstruction Criterion 
    criterion = nn.MSECriterion():cuda() 
    FISTA_loss = 0 
    min_change = 0.05 
    --initialize FISTA storage 
    Zprev = torch.zeros(bsz,outplane,32,32):cuda()
    Z = Zprev:clone() 
    Y = Zprev:clone() 
    Ynext = Zprev:clone() 
    
    dat = torch.Tensor(50000,3,32,32)
    cod = torch.Tensor(50000,16,32,32) 
    
    local cntr = 0 
    
    for i = 1, ds:size() do 
    
        progress(i,ds:size()) 
    
        --get a sample 
        X = ds:next() 
        
        --FISTA inference (iterate until change in Y is <%min_change)  
        local Ydiff = math.huge 
        local t = 1
        Zprev:fill(0)
        Y:copy(Zprev)
      
        while Ydiff > min_change do 
            --ISTA
            Xerr = decoder:forward(Y):add(-1,X)  
            dZ = encoder:forward(Xerr)  
            Z:copy(Y:add(-1/L,dZ))  
            Z:add(-l1w/L) 
            Z:copy(threshold(Z))
            --FISTA 
            local tnext = (1 + math.sqrt(1+4*math.pow(t,2)))/2 
            Ynext:copy(Z)
            Ynext:add((1-t)/tnext,Zprev:add(-1,Z))
            Ydiff = Y:add(-1,Ynext):norm(2)/Ynext:norm(2) 
            --copies for next iter 
            Y:copy(Ynext)
            Zprev:copy(Z)
            t = tnext
        end
        
            dat:narrow(1,cntr+1,16):copy(X) 
            cod:narrow(1,cntr+1,16):copy(Y) 
            cntr = cntr + 16 
    
            Xr = decoder:forward(Y)
            rec_error = criterion:forward(Xr,X) 
            --track loss 
            sample_loss = 0.5*rec_error + l1w*Z:norm(1)/(bsz*outplane*32*32)  
            FISTA_loss = FISTA_loss + sample_loss 
    end
    
    FISTA_loss = FISTA_loss/ds:size()  
    print('FISTA Loss: '..tostring(FISTA_loss))
end
--save and example to make sure data is aligned 
save_dir = './Results/Experiments/ConvLISTA_test/'
os.execute('mkdir -p '..save_dir) 
exp_conf = paths.thisfile() 
os.execute('cp '..exp_conf..' '..save_dir)
FISTA_train_data = {dat,cod}
torch.save(save_dir..'FISTA_train_data16.t7',FISTA_train_data)

ds_FISTA = DataSource({dataset = FISTA_train_data[1], targets = FISTA_train_data[2] , batchSize = bsz})
sample, target = ds_FISTA:next()
rec = decoder:forward(target) 

Wd = flip(decoder:get(2).weight:float())
Idec = image.toDisplayTensor({input=Wd,nrow=16,padding=1}) 
image.save(save_dir..'dec.png', Idec)
Isample = image.toDisplayTensor({input=sample,nrow=16,padding=1}) 
image.save(save_dir..'sample.png', Isample)
Itarget = image.toDisplayTensor({input=target:clone():resize(16*16,1,32,32),nrow=16,padding=1}) 
image.save(save_dir..'target.png', Itarget)
Irec = image.toDisplayTensor({input=rec,nrow=16,padding=1}) 
image.save(save_dir..'rec.png', Irec)

--load the decoder 
--decoder = torch.load('./Results/Trained_Networks/ISTA_Decoder16.t7') 
learn_rate = 0.01
--MSE reconstruction criterion 
criterion = nn.MSECriterion():cuda()
--save results 
record_file = io.open(save_dir..'output.txt', 'w') 
record_file:write('\n') 
record_file:close()
--Compute FISTA loss 
loss_FISTA = 0 
for i = 1, ds_FISTA:size() do 
    progress(i,ds_FISTA:size()) 
    x,z = ds_FISTA:next()
    xrec_FISTA = decoder:forward(z) 
    sample_rec_error_FISTA = criterion:forward(xrec_FISTA,x)
    sample_loss_FISTA = 0.5*sample_rec_error_FISTA + l1w*z:norm(1)/(bsz*outplane*32*32)--sample_l1_error 
    loss_FISTA = loss_FISTA + sample_loss_FISTA 
end
loss_FISTA = loss_FISTA/ds:size() 
output = 'FISTA Loss: '..tostring(loss_FISTA) 
print(output)
record_file = io.open(save_dir..'output.txt', 'a') 
record_file:write(output..'\n') 
record_file:close()

for i,n in ipairs({0,1,5}) do 
    
    nloops = n
    epochs = 5
    LISTA = construct_LISTA(ConvEnc:clone():float(),nloops,l1w,L):cuda() 
    
    print('#Loops = '..tostring(nloops)) 
    
    for i = 1, epochs do 
    
        epoch_loss = 0
        epoch_code_error = 0 
        sys.tic()
    
        for j = 1, ds_FISTA:size() do 
            
            progress(j,ds_FISTA:size()) 

            x,z = ds_FISTA:next()
            zpred = LISTA:forward(x) 
            sample_code_error = criterion:forward(zpred,z) 
            grad = criterion:backward(zpred,z)
            LISTA:zeroGradParameters() 
            LISTA:backward(x,grad) 
            LISTA:updateParameters(learn_rate)  
            epoch_code_error = epoch_code_error + sample_code_error

            sample_l1_error = LISTA.output:norm(1)
            xrec = decoder:forward(zpred) 
            sample_rec_error = criterion:forward(xrec,x)
            sample_loss = 0.5*sample_rec_error + l1w*sample_l1_error/(bsz*outplane*32*32)--sample_l1_error 
            epoch_loss = epoch_loss + sample_loss
        
        end
    
        epoch_code_error = epoch_code_error/ds:size() 
        epoch_loss = epoch_loss/ds:size() 
        output = tostring(i)..' Time: '..tostring(sys.toc())..' Loss:'..tostring(epoch_loss)..' Code error:'..tostring(epoch_code_error) 
        
        print(output)
        record_file = io.open(save_dir..'output.txt', 'a') 
        record_file:write(output..'\n') 
        record_file:close()

    end

end    

--=====================================
print('==========Test 4==========')
--Next test: verify that learned inference improves 
--with more LISTA loops. Here we load a fixed decoder
--and train LISTA to do sparse inference. 
ds = ds_train
learn_rate = 0.005
test_sample = ds.data:narrow(1,1,bsz):cuda()   

for i,n in ipairs({0,1,5}) do 

    nloops = n
    print('#Loops = '..tostring(nloops)) 
    save_dir = './Results/Experiments/ConvLISTA_Inference_Test'..tostring(nloops)..'/'
    os.execute('mkdir -p '..save_dir) 
    exp_conf = paths.thisfile() 
    os.execute('cp '..exp_conf..' '..save_dir) 
    record_file = io.open(save_dir..'output.txt', 'w') 
    record_file:write('\n') 
    record_file:close()
    
    --load the decoder 
    decoder = torch.load('./Results/Trained_Networks/ISTA_Decoder16.t7') 
--    Idec = image.toDisplayTensor({input=flip(decoder:get(2).weight:float()),nrow=8,padding=1}) 
--    image.save(save_dir..'decoder.png', Idec)
    ConvEnc = nn.SpatialConvolutionBatch(inplane, outplane, k, k, stride, stride) 
    ConvEnc.weight:copy(flip(decoder:get(2).weight)) 
    ConvEnc.bias:fill(0)
    
    --epochs = 5 
    --insz = 3 
    --D = 16
    --k = 5
    --stride = 1
    --pad = (k-1)/2
    --hyper-parameters 
    --l1w = 0.5
    --alpha = -0.5
    --L = 100
--    encoder = nn.SpatialConvolutionBatch(insz, D, k, k, stride, stride) 
--    encoder.weight:copy(flip(decoder:get(2).weight)) 
    
    LISTA = construct_LISTA(ConvEnc:clone():float(),nloops,l1w,L)
    
    net = nn.Sequential()
    net.cost_table = {} 
    net.cost_table.L1 = {} 
    net:add(LISTA) 
    net:add(nn.L1Penalty(l1w,net.cost_table,true)) 
    net:add(decoder) 
    net:cuda() 
    
    criterion = nn.MSECriterion():cuda()
    
    for i = 1, epochs do 
    
        Wd = net:get(3):get(2).weight:clone() 
        
        epoch_loss = 0 
       
        sys.tic()
    
        for j = 1, ds:size() do 
            
            progress(j,ds:size()) 
            sample = ds:next()
            sample_norm = (sample:norm()^2)/(16*3*32*32)
            sample = sample:cuda() 
            rec = net:forward(sample)
    
            sample_l1_error = net.cost_table.L1[1]
            sample_rec_error = criterion:forward(rec,sample)
            sample_loss = 0.5*sample_rec_error + l1w*net:get(1).output:norm(1)/(bsz*outplane*32*32)--sample_l1_error 
    
            grad = criterion:backward(rec,sample) 
            net:zeroGradParameters() 
            net:backward(sample,grad) 
            --fixed decoder 
            net:get(3):get(2).gradWeight:fill(0) 
            net:get(3):get(2).gradBias:fill(0) 
            net:updateParameters(learn_rate)  
    
            epoch_loss = epoch_loss + sample_loss 
        
        end
      
        net:forward(test_sample) 
        sample_act = net:get(1).output[1]:float():squeeze() 
    
        epoch_loss = epoch_loss/ds:size() 
        output = tostring(i)..' Time: '..tostring(sys.toc())..' Loss:'..tostring(epoch_loss)
        
        print(output)
        record_file = io.open(save_dir..'output.txt', 'a') 
        record_file:write(output..'\n') 
        record_file:close()

        --print(Wd:add(-1,net:get(3):get(2).weight):norm())  
    
        We = net:get(1).encoder.weight:float()  
        dWe = net:get(1).encoder.gradWeight:float()  
        Wd = flip(decoder:get(2).weight:float()) 
    
   --     if i==1 or math.mod(i,10) == 0 then 
            test_rec = net:forward(test_sample)
            Irec = image.toDisplayTensor({input=test_rec,nrow=8,padding=1}) 
            image.save(save_dir..'rec'..tostring(i)..'.png', Irec)
            Ienc = image.toDisplayTensor({input=We,nrow=8,padding=1}) 
            image.save(save_dir..'enc'..tostring(i)..'.png', Ienc)
            Idec = image.toDisplayTensor({input=Wd,nrow=8,padding=1}) 
            image.save(save_dir..'dec'..tostring(i)..'.png', Idec)
            --Iact = image.toDisplayTensor({input=ave_act,nrow=8,padding=1}) 
            --image.save(save_dir..'act'..tostring(i)..'.png', Iact)
            Isample_act = image.toDisplayTensor({input=sample_act,nrow=8,padding=1}) 
            image.save(save_dir..'sample_act.png', Isample_act)
  --      end 
    
        --gnuplot.hist(We,100) 
        --gnuplot.figprint(save_dir..(init or '')..'We'..'.eps')  
        --gnuplot.hist(dWe,100) 
        --gnuplot.figprint(save_dir..(init or '')..'dWe'..'.eps')  
        --gnuplot.closeall() 
    
    end 

end --]] 
