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
    data = train_data.datacn:resize(50000,3,32,32) 
end

bsz = 16
ds_small = DataSource({dataset = data:narrow(1,1,1000), batchSize = bsz})
ds_large = DataSource({dataset = data, batchSize = bsz})

ds = ds_small
epochs = 100 
inplane = 3 
outplane = 32
k = 9
stride = 1
padding = (k-1)/2
--1/(code learning rate)  
L = 100
--inference iters 
niter = 10
--dictionary learning rate 
learn_rate = 0.5
--L1 weight 
l1w = 0.50
--=====initialize componenets=====  
--Decoder 
decoder = nn.Sequential() 
ConvDec = nn.NormSpatialConvolutionFFT(outplane, inplane, k, k, stride, stride) 
ConvDec.bias:fill(0) 
decoder:add(nn.SpatialPadding(padding,padding,padding,padding,3,4))
decoder:add(ConvDec) 
decoder:cuda() 
--Encoder
encoder = nn.Sequential() 
ConvEnc = nn.SpatialConvolutionFFT(inplane, outplane, k, k, stride, stride) 
ConvEnc.weight:copy(flip(ConvDec.weight)) 
ConvEnc.bias:fill(0)
encoder:add(nn.SpatialPadding(padding,padding,padding,padding,3,4))
encoder:add(ConvEnc) 
encoder:cuda() 
--Thresholding-operator 
threshold = nn.Threshold(0,0):cuda()  
--Initial code 
Zprev = torch.zeros(bsz,outplane,32,32):cuda()
--Reconstruction Criterion 
criterion = nn.MSECriterion():cuda() 
--initialize FISTA storage 
Z = Zprev:clone() 
Y = Zprev:clone() 
Ynext = Zprev:clone() 

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
      
      --while Ydiff > min_change do 
     for i = 1,niter do 
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
         --copies for next iter 
          Y:copy(Ynext)
          Zprev:copy(Z)
          t = tnext
     end
      
      --update dictionary
      Xr = decoder:forward(Y)
      rec_error = criterion:forward(Xr,X) 
      rec_grad = criterion:backward(Xr,X) 
      decoder:zeroGradParameters() 
      decoder:backward(Y,rec_grad)
      decoder:updateParameters(learn_rate) 
      ConvDec.bias:fill(0) 
      --copy to encoder  
      ConvEnc.weight:copy(flip(ConvDec.weight))
      --track loss 
      sample_rec_error = Xr:clone():add(-1,X):pow(2):sum(3):sum(4):cdiv(X:pow(2):sum(3):sum(4)):sqrt():mean()
      sample_sparsity = 1-(Y:float():gt(0):sum()/Y:nElement())
      sample_loss = rec_error + l1w*Y:norm(1)/(bsz*outplane*32*32)  
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
  Idec = image.toDisplayTensor({input=flip(ConvDec.weight:float()),nrow=8,padding=1}) 
  image.save(save_dir..'dec.png', Idec)

end 

torch.save(save_dir..'decoder.t7', decoder)  
