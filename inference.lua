findMaxEigenValue = function(decoder) 
    local niter = 100 
    local Wd = decoder:get(2).weight:float()
    local inplane = decoder:get(2).weight:size(1)
    local outplane = decoder:get(2).weight:size(2) 
    local k = decoder:get(2).kW
    local k2 = 2*k-1
    local input = norm_filters(torch.rand(outplane,outplane,k2,k2)) 
    local input_prev = input:clone() 
    local output = input:clone():zero()
    local Sw = torch.Tensor(outplane,outplane,2*k-1,2*k-1) 
    for i = 1,outplane do
        Sw[i] = torch.conv2(Wd:select(2,i),flip(Wd),'F') 
    end
    print('Computing L...') 
    for i = 1,niter do 
        progress(i,niter)
        for i = 1,outplane do
            output[i] = torch.conv2(input:select(2,i),Sw,'F'):narrow(2,(k2-1)/2,k2):narrow(3,(k2-1)/2,k2) 
        end
        input_prev:copy(input) 
        input:copy(norm_filters(output))
    end
    local L = output:norm() 
    print('Found L='..L) 
    return L 
end

ConvFISTA = function(decoder,data,niter,l1w,L)
    --infer sparse code of dataset [ds] in dictionary [decoder] 
    local L = L or 1.05*findMaxEigenValue(decoder) 
    local n = 256 
    local inplane = decoder:get(2).weight:size(1)
    local outplane = decoder:get(2).weight:size(2) 
    local codes = torch.Tensor(data:size(1),outplane,X:size(3),X:size(4))
    local k = decoder:get(2).kW
    local padding = (k-1)/2
    local ConvDec = decoder:get(2) 
    local encoder = nn.Sequential()
    local ConvEnc = nn.SpatialConvolutionFFT(inplane,outplane,k,k) 
    ConvEnc.weight:copy(flip(ConvDec.weight)) 
    ConvEnc.bias:fill(0)
    encoder:add(nn.SpatialPadding(padding,padding,padding,padding,3,4))
    encoder:add(ConvEnc) 
    encoder:cuda() 
    --Thresholding-operator 
    local threshold = nn.Threshold(0,0):cuda()  
    local X = torch.CudaTensor(n,X:size(2),X:size(3),X:size(4))
    local Zprev = torch.zeros(n,outplane,X:size(3),X:size(4)):cuda()
    local Z = Zprev:clone() 
    local Y = Zprev:clone() 
    local Ynext = Zprev:clone() 
    local infer = function(X) 
        --FISTA inference iterates
        local t = 1
        Zprev:fill(0)
        Y:fill(0)
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
        return Y 
    end
    --infer codes for entire dataset 
    print('FISTA interence...\n') 
    for i = 0,data:size(1)-n,n do 
        progress(i,data:size(1))
        X:copy(data:narrow(1,i+1,n))
        Y = infer(X) 
        codes:narrow(1,i+1,n):copy(Y) 
    end
    --process the remaining data 
    local a = math.floor(data:size(1)/n)*n+1
    local b = math.fmod(data:size(1),n)
    if a < data:size(1) then 
        X = torch.CudaTensor(b,X:size(2),X:size(3),X:size(4))
        Zprev = torch.zeros(b,outplane,X:size(3),X:size(4)):cuda()
        Z = Zprev:clone() 
        Y = Zprev:clone() 
        Ynext = Zprev:clone() 
        X:copy(data:narrow(1,a,b))
        Y = infer(X)
        codes:narrow(1,a,b):copy(Y)  
    end 
    --clean up  
    X = nil 
    Zprev = nil 
    Z = nil 
    Ynext = nil 
    Y = nil 
    collectgarbage() 
    return codes
end

--(1) s.g.d. minimization of L = ||x-Df(x)|| + l1w*|f(x)| w.r.t. f()  
train_encoder_lasso = function(encoder,decoder,ds,l1w,learn_rate,epochs)
    
    print('Training encoder..') 
    local net = nn.Sequential() 
    net:add(encoder) 
    net:add(nn.L1Penalty(true,l1w,true)) 
    net:add(decoder) 
    net:cuda() 
    local criterion = nn.MSECriterion():cuda() 
    
    for iter = 1,epochs do 
        local epoch_loss = 0 
        local epoch_sparsity = 0 
        local epoch_rec_error = 0 
        sys.tic() 
        for i = 1, ds:size() do 
           progress(i,ds:size()) 
           local X = ds:next() 
           local Xr = net:forward(X)
           local rec_error = criterion:forward(Xr,X)   
           local Y = net:get(1).output 
           net:zeroGradParameters()
           local rec_grad = criterion:backward(Xr,X):mul(0.5) --MSE criterion multiplies by 2 
           net:backward(X,rec_grad)
           --fixed decoder 
           net:get(3):get(2).gradWeight:fill(0) 
           net:get(3):get(2).gradBias:fill(0) 
           net:updateParameters(learn_rate) 
           --track loss 
           local sample_rec_error = Xr:clone():add(-1,X):pow(2):mean()/X:clone():pow(2):mean()
           local sample_sparsity = 1-(Y:float():gt(0):sum()/Y:nElement())
           local sample_loss = 0.5*rec_error + l1w*Y:norm(1)/(bsz*outplane*32*32)  
           epoch_rec_error = epoch_rec_error + sample_rec_error
           epoch_sparsity = epoch_sparsity + sample_sparsity 
           epoch_loss = epoch_loss + sample_loss 
        end
        local epoch_rec_error = epoch_rec_error/ds:size() 
        local average_loss = epoch_loss/ds:size()  
        local average_sparsity = epoch_sparsity/ds:size() 
        print(tostring(iter)..' Time: '..sys.toc()..' %Rec.Error '..epoch_rec_error..' Sparsity:'..average_sparsity..' Loss: '..average_loss) 
        --Irec = image.toDisplayTensor({input=Xr,nrow=8,padding=1}) 
        --image.save(save_dir..'Irec.png', Irec)
        --Ienc = image.toDisplayTensor({input=encoder.weight:float(),nrow=8,padding=1}) 
        --image.save(save_dir..'enc.png', Ienc)
    end 

end




