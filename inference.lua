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
    local codes = torch.Tensor(data:size(1),outplane,data:size(3),data:size(4))
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
    local X = torch.CudaTensor(n,data:size(2),data:size(3),data:size(4))
    local Zprev = torch.zeros(n,outplane,data:size(3),data:size(4)):cuda()
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

construct_deep_net = function(nlayers,inplane,outplane,k,tied_weights)
--deep ReLU network with [optionally] shared weights (inialized identically to LISTA) 
    print('Initilizing deep ReLU from LISTA init') 
    local We = nn.SpatialConvolutionFFT(inplane,outplane,k,k) 
    local LISTA = construct_LISTA(We,1,config.l1w,config.L,config.untied_weights)
    local pad1 = LISTA.pad1
    local pad2 = LISTA.pad2
    local conv1 = LISTA.encoder 
    local conv = LISTA.S 
    local net = nn.Sequential() 
    net:add(pad1:clone())
    net:add(conv1) 
    net:add(nn.Threshold(0,0))
    for i=2,nlayers do 
        local conv_clone = conv:clone()         
        if tied_weights==true then  
            conv_clone:share(conv, 'weight') 
            conv_clone:share(conv, 'bias') 
        end 
        net:add(pad2:clone()) 
        net:add(conv_clone) 
        net:add(nn.Threshold(0,0))
    end
    return net 
end

construct_LISTA = function(encoder,nloops,alpha,L,untied_weights)
--[We] = encoder (linear operator) 
--[S] = 'explaining-away' (square linear operator)
--[n] = number of LISTA loops 
--WARNING: Assumes CudaTensor  
    encoder = encoder:clone() 
    local alpha = alpha or 0.5 
    --initialize S 
    local S = nn.Sequential() 
    if torch.typename(encoder) == 'nn.Linear' then 
        local We = encoder.weight:float()
        local D = We:size(1) 
        local Sw = torch.mm(We,We:t()) 
        local _,L,_ = math.sqrt(torch.svd(We)[1]) 
        Sw = torch.eye(D,D) - torch.mm(We,We:t()):div(L)   
        S:add(nn.Linear(D,D))
        S:get(1).weight:copy(Sw) 
        S:get(1).bias:fill(-alpha/L)
        S:cuda()
        encoder:cuda()
    elseif string.find(torch.typename(encoder),'nn.SpatialConvolution') then 
        print('Initializing convolutional LISTA...') 
        -- flip because conv2 flips whereas nn.SpatialConvolution will not 
        local We = flip(encoder.weight) 
        --dimensions (assume square, odd-sized kernels and stride = 1) 
        local inplane = We:size(1) 
        local outplane = We:size(2)
        local k = We:size(3)
        local padding = (k-1)/2
        local pad = nn.SpatialPadding(padding,padding,padding,padding,3,4) 
        local pad2 = nn.SpatialPadding(2*padding,2*padding,2*padding,2*padding,3,4) 
        local Sw = torch.Tensor(outplane,outplane,2*k-1,2*k-1) 
        for i = 1,outplane do
            Sw[i] = torch.conv2(We:select(2,i),flip(We),'F') 
        end
        --find L using power method
        if L == nil then  
            local k2 = 2*k-1
            local input = norm_filters(torch.rand(outplane,outplane,k2,k2)) 
            local input_prev = input:clone() 
            local output = input:clone():zero()
            for i = 1,100 do 
                progress(i,100)
                for i = 1,outplane do
                    output[i] = torch.conv2(input:select(2,i),Sw,'F'):narrow(2,(k2-1)/2,k2):narrow(3,(k2-1)/2,k2) 
                end
                input_prev:copy(input) 
                input:copy(norm_filters(output))
            end
            L = output:norm() 
        end
        Sw:div(L) 
        local I = torch.zeros(outplane,outplane,2*k-1,2*k-1) 
        for i = 1,outplane do 
            I[i][i][math.ceil(k-0.5)][math.ceil(k-0.5)] = 1 
        end
        Sw = I - Sw
        S:add(pad2:clone()) 
        S:add(nn.SpatialConvolutionFFT(outplane,outplane,2*k-1,2*k-1))
        S:get(2).weight:copy(Sw) 
        S:get(2).bias:fill(0)
        S:cuda() 
        encoder.weight:div(L) 
        encoder.bias:fill(-alpha/L)
        local encoder_same = nn.Sequential() 
        encoder_same:add(pad:clone()) 
        encoder_same:add(encoder) 
        encoder = encoder_same:cuda() 
    else 
        error('Unsupported LISTA encoder') 
    end
    local internal_LISTA_loop = function(S) 
        local net = nn.Sequential() 
        local branch1 = nn.ParallelTable() 
        local split = nn.ConcatTable() 
        split:add(nn.Identity())
        split:add(nn.Identity())
        local ff = nn.Sequential() 
        ff:add(nn.Threshold(0,0)) 
        if untied_weights == true then 
            ff:add(S:clone()) 
        else
            ff:add(S) 
        end
        branch1:add(split)
        branch1:add(ff) 
        net:add(branch1) 
        local forward = function(x) return {x[1][1], {x[1][2], x[2]}} end
        local backward = function(x) return {{x[1], x[2][1]}, x[2][2]} end  
        net:add(nn.ReshapeTable(forward,backward))  
        local branch2 = nn.ParallelTable() 
        branch2:add(nn.Identity()) 
        branch2:add(nn.CAddTable())
        net:add(branch2) 
        return net 
    end
    --first stage 
    local net = nn.Sequential() 
    net:add(encoder)
    if nloops == 0 then 
        net:add(nn.Threshold(0,0)) 
        net.encoder = encoder:get(2)  
        net:cuda()
        return net 
    end 
    local split = nn.ConcatTable() 
    split:add(nn.Identity()) 
    split:add(nn.Identity())
    net:add(split) 
    --internal stages
    local nloops = nloops or 1
    for i = 1, nloops-1 do 
        net:add(internal_LISTA_loop(S)) 
    end
    --last stage 
    local last_stage = nn.Sequential()
    local branch = nn.ParallelTable() 
    branch:add(nn.Identity()) 
    local ff = nn.Sequential() 
    ff:add(nn.Threshold(0,0)) 
    if untied_weights == true then 
        ff:add(S:clone()) 
    else
        ff:add(S) 
    end
    branch:add(ff) 
    last_stage:add(branch) 
    last_stage:add(nn.CAddTable()) 
    last_stage:add(nn.Threshold(0,0)) 
    net:add(last_stage) 
    net:cuda() 
    net.S = S:get(2) 
    net.encoder = encoder:get(2) 
    net.pad1 = net:get(1):get(1) 
    net.pad2 = S:get(1) 
    return net
end 

--(1) s.g.d. minimization of L = ||x-Df(x)|| + l1w*|f(x)| w.r.t. f() (and D)  
minimize_lasso_sgd = function(encoder,decoder,fix_decoder,ds,l1w,learn_rate,epochs,save_dir)
    if save_dir ~= nil then  
        local record_file = io.open(save_dir..'output.txt', 'a') 
        record_file:write('Training Output:\n') 
        record_file:close()
    end
    local inplane = decoder:get(2).weight:size(1)
    local outplane = decoder:get(2).weight:size(2) 
    
    local net = nn.Sequential() 
    net:add(encoder) 
    net:add(nn.ModuleL1Penalty(true,l1w,true)) 
    net:add(decoder) 
    net:cuda() 
    local criterion = nn.MSECriterion():cuda() 
    local loss_plot = torch.zeros(epochs)
    
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
           if fix_decoder == true then 
            net:get(3):get(2).gradWeight:fill(0) 
            net:get(3):get(2).gradBias:fill(0)
           end
           --training a decoder 
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
        loss_plot[iter] = average_loss
        local average_sparsity = epoch_sparsity/ds:size() 
        local output = tostring(iter)..' Time: '..sys.toc()..' %Rec.Error '..epoch_rec_error..' Sparsity:'..average_sparsity..' Loss: '..average_loss 
        print(output) 
        if save_dir ~= nil then  
            local record_file = io.open(save_dir..'output.txt', 'a') 
            record_file:write(output..'\n') 
            record_file:close()
        end
    end 
    return encoder,loss_plot 
end




