--==========================Visualization============================
--===================================================================
network_plots = function(net, ds, save_path, image_weights, iter)  
        local ps = ''
        if math.mod(iter, 10) == 0 or iter == 1 then
            ps = '_ep' .. tostring(iter)
        end
        --Plot a snapshot of the histogram of the [weights] and [gradWeights] 
        local We = net.encoder.weight:clone() 
        local Wd = net.decoder.weight:clone() 
        gnuplot.hist(We,100) 
        gnuplot.figprint(save_path..(init or '')..'We'..ps..'.eps')  
        gnuplot.hist(Wd,100) 
        gnuplot.figprint(save_path..(init or '')..'Wd'..ps..'.eps')  
        local dWe = net.encoder.gradWeight:clone() 
        local dWd = net.decoder.gradWeight:clone() 
        gnuplot.hist(dWe,100) 
        gnuplot.figprint(save_path..(init or '')..'dWe'..ps..'.eps')  
        gnuplot.hist(dWd,100) 
        gnuplot.figprint(save_path..(init or '')..'dWd'..ps..'.eps')  
        
        --plot %activations   
        local z = transform_data(ds.data:select(2,1), net.encoder):gt(0):float():sum(1):squeeze() 
        local zave = z:div(ds.data:size(1))
        
        gnuplot.plot(zave)
        gnuplot.figprint(save_path..'average_activations'..ps..'.eps')     
        
        ----Image the weights 
        if image_weights == true then 
            local enc_image = image_linear(net.encoder.weight:float(),16,1)  
            local dec_image = image_linear(net.decoder.weight:float():t(),16,1)  
            image.save(save_path..(init or '')..'decoder'..ps..'.png',dec_image) 
            image.save(save_path..(init or '')..'encoder'..ps..'.png',enc_image) 
        end 
end

function serializeTable(val, name, skipnewlines, depth)
    skipnewlines = skipnewlines or false
    depth = depth or 0
    local tmp = string.rep(" ", depth)
    if name then tmp = tmp .. name .. " = " end
    if type(val) == "table" then
        tmp = tmp .. "{" .. (not skipnewlines and "\n" or "")
        for k, v in pairs(val) do
            tmp =  tmp .. serializeTable(v, k, skipnewlines, depth + 1) .. "," .. (not skipnewlines and "\n" or "")
        end
        tmp = tmp .. string.rep(" ", depth) .. "}"
    elseif type(val) == "number" then
        tmp = tmp .. tostring(val)
    elseif type(val) == "string" then
        tmp = tmp .. string.format("%q", val)
    elseif type(val) == "boolean" then
        tmp = tmp .. (val and "true" or "false")
    else
        tmp = tmp .. "\"[inserializeable datatype:" .. type(val) .. "]\""
    end
    return tmp
end

--measure the error on the test set 
test_error = function(net, ds_test) 

    print('Computing test error...') 
    
    --add up all the internal costs 
    local L1_error = 0 
    local SF_error = 0 
    local MSE_error = 0 

    for i = 1, ds_test:size() do 
        
         progress(i,ds_test:size()) 

         local sample = ds_test:next() 
         net:forward(sample)
     
         if net.cost_table.L1 then 
             for i = 1,#net.cost_table.L1 do 
                 L1_error = L1_error + net.cost_table.L1[i] 
             end 
         end 

         if net.cost_table.SF then 
             for i = 1,#net.cost_table.SF do 
                 SF_error = SF_error + net.cost_table.SF[i] 
             end 
         end
         
         if net.cost_table.MSE then 
             for i = 1,#net.cost_table.MSE do 
                 MSE_error = MSE_error + net.cost_table.MSE[i] 
             end
         end

    end
    
    L1_error = L1_error/ds_test:size() 
    SF_error = SF_error/ds_test:size()
    MSE_error = MSE_error/ds_test:size()

    return {L1_error, SF_error, MSE_error} 

end 

temporal_precision_recall = function(Z,K,test_scenes,NNtime)
    
    if NNtime == nil then 
    
        local chunk = 256 
        local qidx = (test_scenes:select(2,1)+test_scenes:select(2,2)):mul(0.5):floor()
        NNtime = temporal_neighbors(Z,qidx,K,chunk) 

    end
    
    --(1) determine if NN is in the same scene or not 
    local hits = NNtime:clone():zero()
    
    for i = 1,hits:size(1) do 
        for j = 1,hits:size(2) do 
            if NNtime[i][j]>=test_scenes[i][1] and NNtime[i][j]<=test_scenes[i][2] then 
                hits[i][j] = 1 
            else 
                hits[i][j] = 0 
            end 
        end
    end
    
    hits = hits:narrow(2,2,K) 
    
    --(2) compute recall = hits/#frames in scene 
    local RE = hits:cumsum(2) 
    
    --new normalization 
    --local NNtrue = (test_scenes:select(2,2)-test_scenes:select(2,1)):add(-1)   
    --local REnorm = NNtime:narrow(2,2,K):clone():fill(0) 
    --for i = 1,REnorm:size(1) do 
    --    for j = 1,REnorm:size(2) do 
    --        if j < NNtrue[i] then 
    --            REnorm[i][j] = j 
    --        else 
    --            REnorm[i][j] = NNtrue[i]
    --        end
    --    end
    --end
    --RE = RE:cdiv(REnorm):mean(1):squeeze()  

    --old normalization 
    local NNtrue = (test_scenes:select(2,2)-test_scenes:select(2,1)):add(-1) 
    NNtrue = NNtrue:resize(NNtrue:size(1),1):repeatTensor(1,K) 
    RE = RE:cdiv(NNtrue):mean(1):squeeze()   
    
    --(3) compute precision
    local PR = hits:cumsum(2) 
    local PRnorm = torch.range(1,K):resize(1,K):repeatTensor(PR:size(1),1)
    PR = PR:cdiv(PRnorm):mean(1):squeeze()
   
    --(4)Area under the curve  
    local AUC = 0 
    for i = 1,PR:size(1)-1 do 
        AUC = AUC + 0.5*(PR[i]+PR[i+1])*(RE[i+1]-RE[i])
    end

    return PR, RE, AUC, hits 
end 

class_precision_recall = function(train_data,test_data,train_labels,test_labels,K)
    
    local NNclass = class_neighbors(train_data, test_data, train_labels, test_labels, K) 
    
    --(1) determine if NN is in the same scene or not 
    local hits = NNclass:clone():zero()
    
    for i = 1,hits:size(1) do 
        for j = 1,hits:size(2) do 
            if NNclass[i][j]== NNclass[i][1] then 
                hits[i][j] = 1 
            else 
                hits[i][j] = 0 
            end 
        end
    end
    
    hits = hits:narrow(2,2,K) 
    
    --(2) compute recall = hits/#frames in scene 
    local RE = torch.range(1,K)--hits:cumsum(2) 
    
    --old normalization
    --local NNtrue = torch.zeros(NNclass:size(1)):fill(K) 
    --NNtrue = NNtrue:resize(NNtrue:size(1),1):repeatTensor(1,K) 
    --RE = RE:cdiv(NNtrue):mean(1):squeeze()   
    
    --(3) compute precision
    local PR = hits:cumsum(2) 
    local PRnorm = torch.range(1,K):resize(1,K):repeatTensor(PR:size(1),1)
    PR = PR:cdiv(PRnorm):mean(1):squeeze()
   
    --(4)Area under the curve  
    local AUC = 0 
    for i = 1,PR:size(1)-1 do 
        AUC = AUC + 0.5*(PR[i]+PR[i+1])*(RE[i+1]-RE[i])
    end

    return PR, RE, AUC, hits 
end 

--measure temporal coherence 
function temporal_neighbors(Z,qidx,K,chunk)

    local chunk = chunk or 1024 
    local nq = qidx:nElement() 
    local NNtime = torch.Tensor(nq,K+1) 
    
    do --for garbage collection   
        
        --cosine distance 
        print('computing distance matrix...') 
        if Z:dim() == 4 then 
            Z = Z:resize(Z:size(1),Z:size(2)*Z:size(3)*Z:size(4)):contiguous()
        end
        local Znorm = Z:norm(2,2):resize(Z:size(1),1):expand(Z:size(1),Z:size(2)):contiguous()  
        Z:cdiv(Znorm) 
        Znorm = nil
        collectgarbage() 
        
        local Zq = torch.CudaTensor(nq,Z:size(2))
        for i = 1,nq do 
            Zq[i]:copy(Z[qidx[i]])
        end
        Zq = Zq:t():contiguous() 

        local dist = torch.Tensor(Z:size(1),Zq:size(2)):zero()
        local dist_chunk = torch.CudaTensor(chunk,Zq:size(2)):fill(0)  
        local Z_chunk = torch.CudaTensor(chunk,Z:size(2)) 
        
        for i = 0,Z:size(1)-chunk,chunk do 
        
            progress(i,Z:size(1)) 
            dist_chunk:zero()
            Z_chunk:copy(Z:narrow(1,i+1,chunk))
            dist_chunk:addmm(Z_chunk,Zq)
            dist:narrow(1,i+1,chunk):copy(dist_chunk)
            collectgarbage() 
        
        end
        
        local a = math.floor(Z:size(1)/chunk)*chunk+1
        local b = math.fmod(Z:size(1),chunk)
        dist_chunk:zero()
        dist_chunk:narrow(1,1,b):addmm(Z_chunk:narrow(1,1,b):copy(Z:narrow(1,a,b)),Zq) 
        dist:narrow(1,a,b):copy(dist_chunk:narrow(1,1,b))
        
        --find a way around?  
        print('transposing...') 
        dist = dist:t():contiguous() 
        collectgarbage() 
        
        print('finding NN...') 

        for n = 1,nq do 
            progress(n,nq) 
            NNtime[n][1] = qidx[n] 
            dist[n][qidx[n]] = -math.huge 
        
            for i = 2,K+1 do 
                local val,idx = torch.max(dist[n],1) 
                NNtime[n][i] = idx[1]
                dist[n][idx[1]] = -math.huge 
            end
        end 
    
    end
    collectgarbage() 

    return NNtime 
end

function class_neighbors(Ztrain,Ztest,train_labels,test_labels,K) 
    
    local chunk = chunk or 1024 
    local ntest = Ztest:size(1) 
    local NNclass = torch.Tensor(ntest,K+1) 
    
    do --for garbage collection   
        
        --cosine distance 
        print('computing distance matrix...') 
       
       if Ztrain:dim() == 4 then 
        Ztrain = Ztrain:resize(Ztrain:size(1),Ztrain:size(2)*Ztrain:size(3)*Ztrain:size(4)):contiguous()
       end 

        local Ztrain_norm = Ztrain:norm(2,2):resize(Ztrain:size(1),1):expand(Ztrain:size(1),Ztrain:size(2)):contiguous()  
        Ztrain:cdiv(Ztrain_norm) 
        
       if Ztest:dim() == 4 then  
        Ztest = Ztest:resize(Ztest:size(1),Ztest:size(2)*Ztest:size(3)*Ztest:size(4)):contiguous()
       end     

        local Ztest_norm = Ztest:norm(2,2):resize(Ztest:size(1),1):expand(Ztest:size(1),Ztest:size(2)):contiguous()  
        Ztest:cdiv(Ztest_norm) 
        
        Znorm = nil
        collectgarbage() 
        
        Zq = Ztest:t():cuda():contiguous() 

        local dist = torch.Tensor(Ztrain:size(1),Zq:size(2)):zero()
        local dist_chunk = torch.CudaTensor(chunk,Zq:size(2)):fill(0)  
        local Ztrain_chunk = torch.CudaTensor(chunk,Ztrain:size(2)) 
        
        for i = 0,Ztest:size(1)-chunk,chunk do 
        
            progress(i,Ztrain:size(1)) 
            dist_chunk:zero()
            Ztrain_chunk:copy(Ztrain:narrow(1,i+1,chunk))
            dist_chunk:addmm(Ztrain_chunk,Zq)
            dist:narrow(1,i+1,chunk):copy(dist_chunk)
            collectgarbage() 
        
        end
        
        --remaining data
        local a = math.floor(Ztrain:size(1)/chunk)*chunk+1
        local b = math.fmod(Ztrain:size(1),chunk)
        dist_chunk:zero()
        dist_chunk:narrow(1,1,b):addmm(Ztrain_chunk:narrow(1,1,b):copy(Ztrain:narrow(1,a,b)),Zq) 
        dist:narrow(1,a,b):copy(dist_chunk:narrow(1,1,b))
        
        --find a way around?  
        print('transposing...') 
        dist = dist:t():contiguous() 
        collectgarbage() 
        
        print('finding NN...') 

        for n = 1,ntest do 
            progress(n,ntest) 
            NNclass[n][1] = test_labels[n] 
        
            for i = 2,K+1 do 
                local val,idx = torch.max(dist[n],1) 
                NNclass[n][i] = train_labels[idx[1]]
                dist[n][idx[1]] = -math.huge 
            end
        end 
    
    end
    
    collectgarbage() 

    return NNclass 
end 


function kNN(X,K,global_idx) 
    
    local chunk = chunk or 1024 
    local Ztrain = X 
    local Ztest = X 
    local ntest = Ztest:size(1) 
    local NN = torch.zeros(ntest,K+1) 
    
    do --for garbage collection   
        
        --cosine distance 
        print('computing distance matrix...') 
       
       if Ztrain:dim() == 4 then 
        Ztrain = Ztrain:resize(Ztrain:size(1),Ztrain:size(2)*Ztrain:size(3)*Ztrain:size(4)):contiguous()
       end 

        local Ztrain_norm = Ztrain:norm(2,2):resize(Ztrain:size(1),1):expand(Ztrain:size(1),Ztrain:size(2)):contiguous()  
        Ztrain:cdiv(Ztrain_norm) 
        
       if Ztest:dim() == 4 then  
        Ztest = Ztest:resize(Ztest:size(1),Ztest:size(2)*Ztest:size(3)*Ztest:size(4)):contiguous()
       end     

        local Ztest_norm = Ztest:norm(2,2):resize(Ztest:size(1),1):expand(Ztest:size(1),Ztest:size(2)):contiguous()  
        Ztest:cdiv(Ztest_norm) 
        
        Znorm = nil
        collectgarbage() 
        
        Zq = Ztest:t():cuda():contiguous() 

        local dist = torch.Tensor(Ztrain:size(1),Zq:size(2)):zero()
        local dist_chunk = torch.CudaTensor(chunk,Zq:size(2)):fill(0)  
        local Ztrain_chunk = torch.CudaTensor(chunk,Ztrain:size(2)) 
        
        for i = 0,Ztest:size(1)-chunk,chunk do 
        
            progress(i,Ztrain:size(1)) 
            dist_chunk:zero()
            Ztrain_chunk:copy(Ztrain:narrow(1,i+1,chunk))
            dist_chunk:addmm(Ztrain_chunk,Zq)
            dist:narrow(1,i+1,chunk):copy(dist_chunk)
            collectgarbage() 
        
        end
        
        --remaining data
        local a = math.floor(Ztrain:size(1)/chunk)*chunk+1
        local b = math.fmod(Ztrain:size(1),chunk)
        dist_chunk:zero()
        dist_chunk:narrow(1,1,b):addmm(Ztrain_chunk:narrow(1,1,b):copy(Ztrain:narrow(1,a,b)),Zq) 
        dist:narrow(1,a,b):copy(dist_chunk:narrow(1,1,b))
        
        --find a way around?  
        print('transposing...') 
        dist = dist:t():contiguous() 
        collectgarbage() 
        
        print('finding NN...') 

        for n = 1,ntest do 
            progress(n,ntest) 
        
            for i = 2,K+1 do 
                local val,idx = torch.max(dist[n],1) 
                if global_idx ~= nil then 
                    NN[n][i] = global_idx[idx[1]]
                else 
                    NN[n][i] = idx[1]
                end
                dist[n][idx[1]] = -math.huge 
            end
        end 
    
    end
    
    NN = NN:narrow(2,2,K) 
    collectgarbage() 

    return NN 
end 


--transforms the data [[nsamples]x[dim]] using the :forward 
--function of the auto-encoder network [net] 
--The data is then renormalized in this new space

transform_data = function(data, net, n) 
   
    local zsz = net:forward(data:narrow(1,1,16):cuda()):size()  
    zsz[1] = data:size(1) 

    local n = n or 128 
    local z = torch.Tensor():resize(zsz)  

    for i = 0,data:size(1)-n,n do 

        progress(i,data:size(1))  

        z:narrow(1,i+1,n):copy(net:forward(data:narrow(1,i+1,n):cuda()))  
          
        collectgarbage()  
    
    end 

    --fprop the remaining data 
    local a = math.floor(data:size(1)/n)*n+1
    local b = math.fmod(data:size(1),n)
    
    if a < z:size(1) then 
        z:narrow(1,a,b):copy(net:forward(data:narrow(1,a,b):cuda()))  
    end 

    collectgarbage() 

    return z 
    
end

function shuffle(x) 

    local rand_idx = torch.randperm(x:size(1))
    local x_shuffled = torch.Tensor(x:size(1))
    
    for i = 1,x:size(1) do 
        x_shuffled[i] = x[rand_idx[i]]
    end

    return x_shuffled
end 


--normalize convolution kernels 
function norm_filters(w) 
 
    local wsz = w:size()
    w = w:clone():transpose(1,2):resize(wsz[2],wsz[1]*wsz[3]*wsz[4]) 
    local norm = w:norm(2,2):expandAs(w):contiguous() 
    w:cdiv(norm)

    w = w:resize(wsz[2],wsz[1],wsz[3],wsz[4]):transpose(1,2) 
    return w 

end

zca_whiten = function(data,cutoff)

    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions 
    if data:dim() == 4 then 
        n_dimensions = dims[2] * dims[3] * dims[4]
    elseif data:dim() == 2 then 
        n_dimensions = dims[2] 
    else 
        error('zca_whiten: data must be 4D or 2D tensor') 
    end 

    local mdata = data:reshape(nsamples, n_dimensions)
    mdata:add(torch.ger(torch.ones(nsamples), torch.mean(mdata, 1):squeeze()):mul(-1))
    
    local ce, cv = unsup.pcacov(data:reshape(nsamples, n_dimensions))
    
    --assume noise level is smallest 5% of the energy 
    local cutoff = cutoff or 0.05 
    local csum = torch.cumsum(ce,1)
    local thresh = csum[csum:size(1)]*cutoff
    local tmp = csum:clone():add(-thresh):abs()
    local _,idx = torch.min(tmp,1) 
    local epsilon = ce[idx[1]] 
    
    local invval = ce:clone():add(epsilon):sqrt()
    local val = invval:clone():pow(-1)
    local diag = torch.diag(val)
    local P = torch.mm(cv, diag)
    P = torch.mm(P, cv:t())

    local invdiag = torch.diag(invval)
    local invP = torch.mm(cv, invdiag)
    invP = torch.mm(invP, cv:t())
    
    local wdata = torch.mm(mdata, P):resize(dims):contiguous()  
    
    return wdata, P, invP, ce, cv
end


saveTensorToFile = function(filename, tensor)
  local out = torch.DiskFile(filename, 'w')
  out:binary()
  if (torch.typename(tensor) == 'torch.FloatTensor') then
    out:writeFloat(tensor:storage())
  elseif (torch.typename(tensor) == 'torch.DoubleTensor') then
    out:writeDouble(tensor:storage())
  elseif (torch.typename(tensor) == 'torch.CudaTensor') then
    out:writeFloat(tensor:float():storage())
  else
    error('Tensor not supported by this function (but could be easily added)')
  end
  out:close()
end

safe_read = function(filename) 
    os.execute('lockfile '..filename..'.lock') 
    local file = torch.load(filename) 
    os.execute('rm -f '..filename..'.lock') 
    return file 
end

safe_write = function(filename,file) 
    os.execute('lockfile '..filename..'.lock') 
    local file = torch.save(filename,file) 
    os.execute('rm -f '..filename..'.lock') 
end

--image linear layer weights 
image_linear = function(W,n,poolsz)  
    
    local I 
    local sz = W:size()
    if poolsz then 
        n = n/math.sqrt(poolsz)
    end 
    
    poolsz = poolsz or 1 
    --no pooling 
    if poolsz == 1 then 

        W = W:reshape(sz[1],math.sqrt(sz[2]),math.sqrt(sz[2])) 
        I = image.toDisplayTensor{nrow=n,scaleeach=true,input=W,padding=1,symmetric=true} 
    
    else 
    --image the basis arranged by non-overlapping pooled groups 
        local sz = W:size()
        W = W:reshape(sz[1],math.sqrt(sz[2]),math.sqrt(sz[2])) 
        local tmp = W:unfold(1,poolsz,poolsz):transpose(2,4):transpose(3,4) 
        I = image.toDisplayTensor{nrow=math.sqrt(poolsz),scaleeach=true,input=tmp[1],padding=1,symmetric=true}  
        I = I:reshape(1,I:size(1),I:size(2))  
        
        for i = 2,tmp:size(1) do 
            
            local Itmp = image.toDisplayTensor{nrow=math.sqrt(poolsz),scaleeach=true,input=tmp[i],padding=1,symmetric=true}  
            Itmp = Itmp:reshape(1,Itmp:size(1),Itmp:size(2))  
            I = torch.cat(I,Itmp,1)  
        
        end
        
        I = image.toDisplayTensor{nrow=n,input=I,padding=2} 
   
    end

    return I 

end 

--visualize the decoder weights  
image_decoder = function(decoder, n) 

    --image the basis in no particular order 
    local Wd = decoder:get(1).weight:float():t()  
    local I = image_linear(Wd,n) 

    return I 

end

image_hyper_spectral = function(W,nrow) 

    local Iw 
    local wsz = W:size() 
    nrow = nrow or 8 

    if W:dim()~=4 then 
        error('Error imaging: '..tostring(W:dim())..' dimensional tensor') 
    end

    if wsz[2] == 1 or wsz[2] == 3 then 
       Iw = image.toDisplayTensor({input=W,nrow=nrow,padding=1})
    else  
       W = W:resize(wsz[1]*wsz[2],1,wsz[3],wsz[4]):contiguous()   
       Iw = image.toDisplayTensor({input=W,nrow=wsz[2],padding=1}) 
    end 

    return Iw 
end 

--patchifies (non-overlapping) 
--the images in X and then  
--computes [nk]-means centers 
kmeans_patches = function(arg) 
    
    local X = arg.X 
    local nsamples = arg.nsamples or 100 
    local psz = arg.psz 
    local nk = arg.nk or 100 
    local niter = arg.niter or 10 
    local bsz = arg.bsz or 16 

    local idx = torch.randperm(X:size(1)):narrow(1,1,nsamples)  
    
    for i = 1,nsamples do  
    
        local Xp = patchify(X[idx[i]],psz)  
        
        if patches == nil then 
            patches = Xp 
        else 
            patches = torch.cat(patches,Xp,1)
        end
    
    end
    
    print('Computing kmeans..') 
    local k,c = unsup.kmeans(patches,nk,niter,bsz,false,true) 

    return k 

end

--converts gray-scale image [I] 
--into square patches of size [psz]
patchify = function(I, psz) 

    if I:dim() == 2 then 

        return patchify_gray(I, psz) 
    
    elseif I:dim() == 3 then 
 
        local p1 = patchify(I[1],psz)
        local p2 = patchify(I[2],psz)
        local p3 = patchify(I[3],psz)

        p1 = p1:resize(p1:size(1),1,p1:size(2),p1:size(3))
        p2 = p2:resize(p2:size(1),1,p2:size(2),p2:size(3))
        p3 = p3:resize(p3:size(1),1,p3:size(2),p3:size(3))

        local X = torch.cat(p1,torch.cat(p2,p3,2),2)             
        --X = X:transpose(1,2) 
        --X = X:reshape(X:size(1),X:size(2)*X:size(3))
        return X 
        
    else 

        error('Patchify works only for 1 or 3 color images!') 
    
    end 

end

patchify_gray = function(I, psz) 
    
    local sz = I:size()
    local sub = psz*math.floor(sz[1]/psz)
    local Isub = I:narrow(1,1,sub):narrow(2,1,sub):clone()  
    sz = Isub:size() 
    local X = Isub:unfold(1,psz,psz):unfold(2,psz,psz):reshape(sz[1]*sz[2]/(psz^2), psz,psz) 
    Isub = nil 
    collectgarbage() 
    return X

end

--linear index to subscript (3D, should be made recursive for ND)
ind2sub = function(idx3,sz) 

    local sub1 = math.ceil(idx3/(sz[2]*sz[3]))
    local idx2 = idx3-((sub1-1)*sz[2]*sz[3])
    local sub2 = math.ceil(idx2/sz[3])
    local sub3 = idx2-((sub2-1)*sz[3])

    return torch.Tensor({sub1,sub2,sub3})  
end

--project points to factored space
project_points = function(X,factors,encoder,selector) 

    local n = factors:size(1) 
    local Xproj = torch.Tensor(n,2) 
    local net = encoder:clone():add(selector) 

    for i = 1,n do 
       
       Xproj[i]:copy(net:forward(X[factors[i][1]]:cuda())) 
    
    end

    return Xproj 

end 

--Center 2D Points at the Origin 
center = function(X) 
    
    X:select(2,1):add(-X:select(2,1):mean())
    X:select(2,2):add(-X:select(2,2):mean())

    X:select(2,1):div(-X:select(2,1):max())
    X:select(2,2):div(-X:select(2,2):max())

    return(X) 
end 

--returns number of elements in any tensor 
function numel(x) 
    local dims = x:size():totable() 
    local sz = 1
    for i = 1,#dims do 
        sz = sz * dims[i]
    end
    return sz
end

--transpose operator for convolutional filter bank 
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

--TODO: connect similar points with line segments 
--Plots points in 2D with color 
color_scatter = function(x, color) 

    --Convert Tensor to String 
    --the format of [s] is x,y,color  
    
    local s = ''
    
        for i = 1, x:size(1) do 
    
            s = s..tostring(x[i][1])..' '..tostring(x[i][2])..' '..tostring(color[i])..'\n'  
            
        end 
    
    local file = io.open("./Results/temp.txt", "w") 
    file:write(s) 
    file:close() 
    
    gnuplot.raw('set palette rgb 3,11,16; plot "./Results/temp.txt" using 1:2:3 with points palette unset colorbox')

end


--======================Performance Evaluation=======================
--===================================================================
--Trains a simple classifier to 
--a predict factor value from features
--Tests factor seperability
classify = function(Xfeat, labels, ntest, classifier, epochs, learn_rate) 
    
    local criterion = nn.ClassNLLCriterion()
    local shuffle = torch.randperm(Xfeat:size(1))
    --Train/Test Split
    local train = shuffle:narrow(1,1,shuffle:size(1)-ntest)
    local test  = shuffle:narrow(1,shuffle:size(1)-ntest+1, ntest)
    print(train:size(1)) 
    print(test:size(1)) 

    for i = 1, epochs do 
        
        local trainErrors = 0 
        local testErrors = 0 
       
        for j = 1, train:size(1) do 
        
            classifier:zeroGradParameters()
            local sample = Xfeat[train[j]]
            local target = labels[train[j]]
            local pred = classifier:forward(sample)
            local error = criterion:forward(pred, target) 
            local derror = criterion:backward(pred, target) 
            classifier:backward(sample, derror) 
            classifier:updateParameters(learn_rate)
            _,pred = torch.max(pred,1)
            
            if pred[1] ~= target then 
                trainErrors = trainErrors + 1 
            end 
    
        end 
       
        for j = 1, test:size(1) do 
        
            local sample = Xfeat[test[j]] 
            local target = labels[test[j]]
            local pred = classifier:forward(sample)
            _,pred = torch.max(pred,1)
            
            if pred[1] ~= target then 
                testErrors = testErrors + 1 
            end 
             
        end 
    
        print(tostring(i)..' Train Error: '..tostring(trainErrors/train:size(1))..' Test Error: '..tostring(testErrors/test:size(1)))
    
    end 

    return classifier 
end

function string:parse_numbers( inSplitPattern, outResults )

   if not outResults then
      outResults = { } 
   end 
   local theStart = 1 
   local theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
   while theSplitStart do
      table.insert( outResults, tonumber(string.sub( self, theStart, theSplitStart-1)) )
      theStart = theSplitEnd + 1 
      theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
   end 
   table.insert(outResults, tonumber(string.sub( self, theStart)) )
   return outResults
end

ls_in_dir = function(data_dir, ls_command) 
    -- paths.dir DOES NOT always return a storted list
    -- local list = paths.dir(data_dir)     
    
    -- THIS METHOD RETURNS A SORTED LIST (ls in posix operating systems
    -- is spec'ed to do so).
    
    local list = {}
    for f in io.popen(ls_command .. data_dir):lines() do
      table.insert(list, f)
    end
    
    if recursive == false then 
        for i,p in ipairs(list) do 
            list[i] = data_dir..'/'..p
        end
    end

    return list 

end

--Trains a simple regressor to 
--a predict factor value from features
--Tests factor seperability
regress = function(Xfeat, targets, ntest, regressor, epochs, learn_rate) 

    local criterion = nn.MSECriterion():float() 
    
    for i = 1,epochs do 
    
      local trainError = 0 
      local testError = 0 
      local shuffle = torch.randperm(Xfeat:size(1))
       --Train/Test Split
       local ntest = 100
       local train = shuffle:narrow(1,1,shuffle:size(1)-ntest)
       local test  = shuffle:narrow(1,shuffle:size(1)-ntest+1, ntest)
    
       for j = 1, train:size(1) do 
       
           regressor:zeroGradParameters()
    
            local sample = Xfeat[train[j]]
            local target = torch.Tensor({targets[train[j]]}):float()
            local pred = regressor:forward(sample)
            local cerror = criterion:forward(pred, target) 
            local derror = criterion:backward(pred, target) 
           regressor:backward(sample, derror) 
           regressor:updateParameters(learn_rate)
           trainError = trainError + cerror 
       end

       for j = 1, test:size(1) do 

            local sample = Xfeat[test[j]] 
            local target = torch.Tensor({targets[train[j]]}):float()
            local pred = regressor:forward(sample)
            local cerror = criterion:forward(pred, target) 
           testError = testError + cerror  
            
       end 
    
        print(tostring(i)..' Train Error: '..tostring(trainError/train:size(1))..' Test Error: '..tostring(testError/test:size(1)))
    
    end 

end


