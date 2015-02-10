--In this first experiment, we will compare learned versus 
--iterative (FISTA) inference in a *fixed* dictionary. The
--learned encoders will be trained minimizing the lasso w.r.t.
--the parameters of the encoder, i.e.
--s.g.d. minimization of L = ||x-Df(x)|| + l1w*|f(x)| w.r.t. f()  
--The list of encoders/codes obtained via inference to compare 
--can be found in EXP1/config.lua 
--======================================
--Evaluations criteria 
--1-Lasso loss 
--2-Sparisity, reconstruction error
--=+=+=+=+=++=+=+=+=+=+=+=+=+=+=+=+=+=+=
dofile('init.lua') 
dofile('EXP1/config.lua') 

os.execute('mkdir -p '..save_dir)  
os.execute('cp ./'..exp_name..'/config.lua '..save_dir)
record_file = io.open(save_dir..'output.txt', 'w') 
record_file:close()

--load the data 
if train_data == nil then 
    train_data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
    train_data = train_data.datacn:resize(50000,3,32,32) 
end
if test_data == nil then 
    test_data = torch.load('./Data/CIFAR/CIFAR_CN_test.t7')
    test_data = test_data.datacn:resize(10000,3,32,32) 
end

--data sources 
ds_small = DataSource({dataset = train_data:narrow(1,1,1000), batchSize = bsz})
ds_train = DataSource({dataset = train_data, batchSize = bsz})
ds_test = DataSource({dataset = test_data, batchSize = bsz})

--load a pre-trained decoder (trained on CIFAR-training set, l1w =?, using FISTA) 
decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7') 
--find max eigen value 
L = decoder.L or 1.05*findMaxEigenValue(decoder)

get_codes = function(config,decoder,ds_train,ds_test) 
    local Ztrain, Ztest 
    if config.name == 'FISTA' then 
        print('FISTA interence...')
        Ztrain = ConvFISTA(decoder,ds_train.data[1],config.niter,config.l1w,config.L) 
        Ztest = ConvFISTA(decoder,ds_test.data[1],config.niter,config.l1w,config.L) 
    elseif config.name == 'LISTA' then 
        local inplane = decoder:get(2).weight:size(1)
        local outplane = decoder:get(2).weight:size(2) 
        local k = decoder:get(2).kW
        local We = nn.SpatialConvolutionFFT(inplane,outplane,k,k,stride,stride) 
        We.weight:copy(flip(decoder:get(2).weight)) 
        We.bias:fill(0)
        --local 
        encoder = construct_LISTA(We,config.nloops,config.l1w,config.L,config.untied_weights)
        --training
        print('Training LISTA via lasso..')
        encoder = train_encoder_lasso(encoder,decoder,ds_train,config.l1w,config.learn_rate,config.epochs,config.save_dir)
        Ztrain = transform_data(ds_train.data[1],encoder)
        Ztest = transform_data(ds_test.data[1],encoder)
    else 
        error('Unsuported encoding config') 
    end 
    return Ztrain, Ztest 
end

all_results = {} 

for i=1,#configs do 
    configs[i].L = L
    output = serializeTable(configs[i])
    local record_file = io.open(save_dir..'output.txt', 'a') 
    record_file:write(output..'\n') 
    record_file:close()
    Ztrain,Ztest = get_codes(configs[i],decoder,ds_train,ds_test)
    eval_test = eval_sparse_code(ds_test.data[1],Ztest,decoder,_l1w)
    eval_train = eval_sparse_code(ds_train.data[1],Ztrain,decoder,_l1w)
    output = output..'\nEval Train\n'..serializeTable(eval_train)..'\n' 
    output = output..'Eval Test\n'..serializeTable(eval_test) 
    local record_file = io.open(save_dir..'output.txt', 'a') 
    record_file:write(output..'\n') 
    record_file:close()
    results[i] = {config=configs[i],train=eval_train,test=eval_test} 
    torch.save(save_dir,'results.t7') 
end

