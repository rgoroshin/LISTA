--In this first experiment, we will compare learned versus 
--iterative (FISTA) inference in a *fixed* dictionary. The
--learned encoders will be trained minimizing the lasso w.r.t.
--the parameters of the encoder, i.e.
--s.g.d. minimization of L = ||x-Df(x)|| + l1w*|f(x)| w.r.t. f() (and D) 
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
    local Ztrain,Ztest,loss_plot 
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
        local encoder = construct_LISTA(We,config.nloops,config.l1w,config.L,config.untied_weights)
        if config.learn_rate == nil then 
            config.learn_rate = find_learn_rate(encoder,decoder,config.fix_decoder,ds_small,config.l1w)
            local record_file = io.open(save_dir..'output.txt', 'a') 
            record_file:write('found learn_rate = '..config.learn_rate..'\n') 
            record_file:close()
        end
        --training
        print('Training LISTA via lasso..')
        encoder,loss_plot = minimize_lasso_sgd(encoder,decoder,config.fix_decoder,ds_train,config.l1w,config.learn_rate,config.epochs,config.save_dir)
        Ztrain = transform_data(ds_train.data[1],encoder)
        Ztest = transform_data(ds_test.data[1],encoder)
    elseif config.name == 'ReLU' then 
        local inplane = decoder:get(2).weight:size(1)
        local outplane = decoder:get(2).weight:size(2) 
        local k = decoder:get(2).kW
        local encoder = construct_deep_net(config.nlayers,inplane,outplane,k,config.untied_weights,config)
        if config.learn_rate == nil then 
            config.learn_rate = find_learn_rate(encoder,decoder,config.fix_decoder,ds_small,config.l1w)
            local record_file = io.open(save_dir..'output.txt', 'a') 
            record_file:write('found learn_rate = '..config.learn_rate..'\n') 
            record_file:close()
        end
        --training
        print('Training ReLU network via lasso..')
        encoder,loss_plot = minimize_lasso_sgd(encoder,decoder,config.fix_decoder,ds_train,config.l1w,config.learn_rate,config.epochs,config.save_dir)
        Ztrain = transform_data(ds_train.data[1],encoder)
        Ztest = transform_data(ds_test.data[1],encoder)
    else 
        error('Unsuported encoding config') 
    end 
    return Ztrain,Ztest,loss_plot  
end

results = {} 
plots = {} 
for i=1,#configs do 
    configs[i].L = L
    rpt = torch.Tensor(configs[i].repeat_exp) 
    results[i]={config=configs[i].name, 
                train={loss=rpt:clone(),rec=rpt:clone(),sparsity=rpt:clone()},
                test ={loss=rpt:clone(),rec=rpt:clone(),sparsity=rpt:clone()}}
    plots[i]={} 
    for j=1,configs[i].repeat_exp do 
        local record_file = io.open(save_dir..'output.txt', 'a') 
        record_file:write('========Config '..i..' Repeat '..j..' ========\n') 
        record_file:write(serializeTable(configs[i])..'\n') 
        record_file:close()
        Ztrain,Ztest,loss_plot = get_codes(configs[i],decoder,ds_train,ds_test)
        eval_test = eval_sparse_code(ds_test.data[1],Ztest,decoder,configs[i].l1w)
        eval_train = eval_sparse_code(ds_train.data[1],Ztrain,decoder,configs[i].l1w)
        output = '\nEval Train\n'..serializeTable(eval_train)..'\n' 
        output = output..'Eval Test\n'..serializeTable(eval_test)..'\n' 
        local record_file = io.open(save_dir..'output.txt', 'a') 
        record_file:write(output..'\n') 
        record_file:close()
        results[i].train.loss[j] = eval_train.average_loss 
        results[i].train.rec[j] = eval_train.average_relative_rec_error  
        results[i].train.sparsity[j] = eval_train.average_sparsity 
        results[i].test.loss[j] = eval_test.average_loss 
        results[i].test.rec[j] = eval_test.average_relative_rec_error  
        results[i].test.sparsity[j] = eval_test.average_sparsity 
        if type(configs[i].epochs) == 'number' then 
            plots[i][j] = {i..'-'..configs[i].name,torch.range(1,configs[i].epochs),loss_plot,'+'}
        end
        torch.save(save_dir..'loss_plot.t7',plots) 
        torch.save(save_dir..'results.t7',results) 
    end 
    plot_results(results,configs,save_dir) 
end



