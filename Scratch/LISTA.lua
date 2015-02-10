construct_LISTA = function(encoder, nloops, alpha, L)
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
        local We = flip(encoder.weight:resize(encoder.nOutputPlane,encoder.nInputPlane,encoder.kH,encoder.kW)) 
        --dimensions (assume square, odd-sized kernels and stride = 1) 
        local inplane = We:size(1) 
        local outplane = We:size(2)
        local k = We:size(3)
        local padding = (k-1)/2
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
        local S = nn.SpatialConvolutionMM(outplane,outplane,2*k-1,2*k-1,1,1,2*padding)
        S.weight:copy(Sw:resize(Sw:size(1),Sw:size(2)*Sw:size(3)*Sw:size(4))) 
        S.bias:fill(0)
        S:cuda() 
        
        encoder.weight:div(L) 
        encoder.bias:fill(-alpha/L)
        local encoder_same = nn.Sequential() 
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
        ff:add(S) 
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
    ff:add(S) 
    branch:add(ff) 
    last_stage:add(branch) 
    last_stage:add(nn.CAddTable()) 
    last_stage:add(nn.Threshold(0,0)) 
    net:add(last_stage) 
    net:cuda() 
    net.S = S:get(2) 
    net.encoder = encoder:get(2) 
    
    return net

end 



