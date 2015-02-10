dofile('init.lua') 
cutorch.setDevice(1) 
exp_name = 'FISTA/'
save_dir = './Results/Experiments/'..exp_name
os.execute('mkdir -p '..save_dir) 
exp_conf = paths.thisfile() 
os.execute('cp '..exp_conf..' '..save_dir) 

--load the data
if train_data == nil then 
    train_data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
    data = train_data.datacn 
end

bsz = 16
ds_small = DataSource({dataset = data:narrow(1,1,10000), batchSize = bsz})
ds_large = DataSource({dataset = data, batchSize = bsz})

ds = ds_large
--ds = ds_small
epochs = 100 
insz = 3*32*32 
outsz = 256
--1/(code learning rate)  
L = 50
--inference iters 
niter = 20
--dictionary learning rate 
learn_rate = 5e-5
--L1 weight 
l1w = 1.5
--=====initialize componenets=====  
--Decoder 
decoder = nn.NormLinear(outsz,insz):cuda()
--Transpose Decoder 
encoder = nn.NormLinear(insz,outsz):cuda()  
encoder.weight:copy(decoder.weight:t())
decoder.bias:fill(0) 
encoder.bias:fill(0)
--Thresholding-operator 
threshold = nn.Threshold(0,0):cuda()  
--Initial code 
Zprev = torch.zeros(bsz,outsz):cuda()
--Reconstruction Criterion 
criterion = nn.MSECriterion():cuda() 
criterion.sizeAverage = false 
--initialize FISTA storage 
Z = Zprev:clone() 
Y = Zprev:clone() 
Ynext = Zprev:clone() 
Xerr = torch.CudaTensor(bsz,insz) 
dZ = torch.CudaTensor(bsz,outsz) 

for iter = 1,epochs do 

 epoch_loss = 0 
 epoch_sparsity = 0 
 epoch_rec_error = 0 
 sys.tic() 
  
 for i = 1,ds:size()+1 do 
  
     progress(i,ds:size()) 

     --get a sample 
     X = ds:next() 
     
     --FISTA inference (iterate until change in Y is <%min_change)  
     local t = 1
     Zprev:fill(0)
     Y:fill(0)
     
     for i = 1,niter do 
        --ISTA
          Xerr:copy(decoder:forward(Y):add(-1,X)) 
          dZ:copy(encoder:forward(Xerr):mul(2))   
          Z:copy(Y:add(-1/L,dZ))
          Z:add(-l1w/L) 
          Z:copy(threshold(Z))
          --FISTA 
          local tnext = (1 + math.sqrt(1+4*math.pow(t,2)))/2 
          Ynext:copy(Z)
          Ynext:add((1-t)/tnext,Zprev:add(-1,Z))
         --copies for next iter 
          Y:copy(Ynext)
          Zprev:copy(Z)
          t = tnext
     
      --  sample_loss = Xerr:clone():pow(2):sum() + l1w*Y:norm(1) 
      --  print(sample_loss) 
      --  sys.sleep(0.2) 
     end
      
      --update dictionary
      Xr = decoder:forward(Y)
      rec_error = criterion:forward(Xr,X) 
      rec_grad = criterion:backward(Xr,X) 
      decoder:zeroGradParameters() 
      decoder:backward(Y,rec_grad)
      decoder:updateParameters(learn_rate) 
      
      --sample_loss = Xr:add(-1,X):pow(2):sum() + l1w*Y:norm(1) 
      --print('****')
      --print(sample_loss)
      
      encoder.weight:copy(decoder.weight:t())
      decoder.bias:fill(0) 
      encoder.bias:fill(0)
      --track loss 
      
      sample_loss = Xerr:pow(2):sum() + l1w*Y:norm(1) 
      sample_rec_error = Xerr:sum(2):cdiv(X:pow(2):sum(2)):mean()
      sample_sparsity = 1-(Y:float():gt(0):sum()/Y:nElement())
      epoch_rec_error = epoch_rec_error + sample_rec_error
      epoch_sparsity = epoch_sparsity + sample_sparsity 
      epoch_loss = epoch_loss + sample_loss 
  end
 
  epoch_rec_error = epoch_rec_error/ds:size() 
  average_loss = epoch_loss/ds:size()  
  average_sparsity = epoch_sparsity/ds:size() 
  print(tostring(iter)..' Time: '..sys.toc()..' %Rec.Error '..epoch_rec_error..' Sparsity:'..average_sparsity..' Loss: '..average_loss) 
  Irec = image.toDisplayTensor({input=Xr:float():resize(bsz,3,32,32),nrow=8,padding=1}) 
  image.save(save_dir..'Irec.png', Irec)
  Idec = image.toDisplayTensor({input=encoder.weight:float():resize(outsz,3,32,32),nrow=16,padding=1}) 
  image.save(save_dir..'dec.png', Idec)

end 

torch.save(save_dir..'decoder.t7', decoder)  
