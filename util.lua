function serializeTable(val, name, skipnewlines, depth)
    --this function is used for printing config tables 
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

flip = function(W) 
    local reverse = function(x) 
        local n = x:size(1) 
        for i = 1,math.floor(n/2) do 
            local tmp = x[i]
            x[i] = x[n-i+1]
            x[n-i+1] = tmp
        end 
        return x 
    end
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

transform_data = function(data, net, n) 
--transforms the data [[nsamples]x[dim]] using the :forward 
--function of the auto-encoder network [net] 
--The data is then renormalized in this new space
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

find_learn_rate = function(encoder,decoder,fix_decoder,ds_small,l1w,epochs,niter,max,min,res)
    --finds the optimal learning rate for [net] optimized by calling [foptim]
    --the trained network is evaluated by calling [feval]. 
    --This function uses a coarse-to-fine searchof depth [niter] to find
    --the optimal learning rate.  
    print('Finding optimal learning rate...') 
    local min = min or -6 
    local max = max or -1
    local res = res or 3 
    local grid = torch.logspace(min,max,res)
    local best_learn_rate = 0 
    niter = niter or 2 
    epochs = epochs or 3 
    for iter = 1,niter do
        print('Level '..iter..' of '..niter) 
        print(grid) 
        local loss_grid = grid:clone():zero()
        for i = 1,grid:size(1) do
            local learn_rate = grid[i] 
            local init_encoder = encoder:clone()
            local init_decoder = decoder:clone() 
            local trained_encoder = minimize_lasso_sgd(init_encoder,init_decoder,fix_decoder,ds_small,l1w,learn_rate,epochs)
            local Z = transform_data(ds_small.data[1],trained_encoder)
            local eval = eval_sparse_code(ds_small.data[1],Z,decoder,l1w) 
            loss_grid[i] = eval.average_loss
        end         
        local vv,ii = torch.min(loss_grid,1); ii = ii[1] 
        best_learn_rate = grid[ii] 
        if ii == 1 or ii == res then 
           return best_learn_rate 
        end
        local new_grid = torch.logspace(math.log10(grid[ii-1]),math.log10(grid[ii+1]),res) 
        if grid:add(-1,new_grid):norm(1)==0 then 
            return best_learn_rate 
        end
        grid = new_grid 
    end 
    print('found learning rate = '..best_learn_rate) 
    return best_learn_rate 
end

eval_sparse_code = function(X,Z,decoder,l1w) 
    local X = X:float() 
    local Z = Z:float() 
    local Xr = transform_data(Z,decoder) 
    local Xsqerr = Xr:add(-1,X):pow(2)
    local rec_mse = Xsqerr:mean() 
    local rel_rec_mse = rec_mse/X:pow(2):mean()
    local loss = 0.5*rec_mse + l1w*(Z:abs():mean())
    local sparsity = 1-(Z:gt(0):sum()/Z:nElement())
    X = nil 
    Z = nil
    collectgarbage() 
    return {average_loss=loss, average_relative_rec_error=rel_rec_mse, average_sparsity=sparsity}
end

 
