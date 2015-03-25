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
torch.setnumthreads(4)
math.randomseed(123)
cutorch.manualSeed(123) 
torch.manualSeed(123)
--==****interactive mode****==
interactive=false 
gpu_id = 1 
--============================
if interactive == false then 
        disp_results = 10
        cutorch.setDevice(tonumber(arg[1]))
        config_list_dir = './Experiments/config/'
        done_dir = './Experiments/done/'
    elseif interactive == true then  
        cutorch.setDevice(gpu_id)
        config_list_dir = './Experiments/interactive/'
        done_dir = './Experiments/interactive/' 
    else error('Unspecified run mode')
end 

while #ls_in_dir(config_list_dir, 'ls ') > 0 do 
    
    config_files = ls_in_dir(config_list_dir, 'ls ')     
    exp_config = nil 
    
    while exp_config == nil and #config_files > 0 do  
        --to minimize the chance of race condition 
        --with another worker, choose a random experiment
        config_file = config_list_dir..config_files[math.random(#config_files)] 
        if config_file ~= nil then
            os.execute('mv '..config_file..' '..done_dir)   
            exp_config = loadfile(done_dir..paths.basename(config_file)) 
        end 
        config_files = ls_in_dir(config_list_dir, 'ls ')    
    end
    
    --load the config 
    print(config_file) 
    exp_config() 
    save_dir = './Results/Experiments/'..exp_name..'/'
    os.execute('mkdir -p '..save_dir)  
    os.execute('mkdir -p '..save_dir..'Code/') 
    os.execute('cp *.lua '..save_dir..'Code/')
    os.execute('cp '..done_dir..paths.basename(config_file)..' '..save_dir) 
    record_file = io.open(save_dir..'output.txt', 'w') 
    record_file:close()

    --load the data 
    if dataset == 'CIFAR_CN' then 
        if train_data == nil then 
            train_data = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
            train_data = train_data.datacn:resize(50000,3,32,32) 
        end
        if test_data == nil then 
            test_data = torch.load('./Data/CIFAR/CIFAR_CN_test.t7')
            test_data = test_data.datacn:resize(10000,3,32,32) 
        end
    else 
        error('unknown dataset!') 
    end 
    
    --data sources 
    ds_small = DataSource({dataset = train_data:narrow(1,1,1000), batchSize = bsz})
    ds_train = DataSource({dataset = train_data, batchSize = bsz})
    ds_test = DataSource({dataset = test_data, batchSize = bsz})
   
    --small experiment for debugging 
    if small_exp==true then 
        ds_train = ds_small 
    end

    --load a pre-trained decoder (trained on CIFAR-training set, l1w =?, using FISTA) 
    decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7')
    decoder:cuda()

    --find max eigen value 
    L = 600 or 1.05*findMaxEigenValue(decoder)
    
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
            local We = nn.SpatialConvolution(inplane,outplane,k,k,stride,stride) 
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
    rpt = torch.Tensor(repeat_exp) 
    results={config=arch, 
             train={loss=rpt:clone(),rec=rpt:clone(),sparsity=rpt:clone()},
             test ={loss=rpt:clone(),rec=rpt:clone(),sparsity=rpt:clone()}}
    --pack arguments 
    local config = {} 
    config.l1w=l1w 
    config.L=L 
    config.epochs=epochs 
    config.bsz=bsz 
    config.repeat_exp=repeat_exp
    config.untied_weights=arch.untied_weights
    config.nlayers=arch.nlayers 
    config.fix_decoder=arch.fix_decoder 
    config.name=arch.name
    config.niter=arch.niter 
    ---
    repeat_loss_plot={}  
    for j=1,repeat_exp do 
        local record_file = io.open(save_dir..'output.txt', 'a') 
        record_file:write('======== Repeat '..j..' ========\n') 
        record_file:close()
        Ztrain,Ztest,loss_plot = get_codes(config,decoder,ds_train,ds_test)
        eval_test = eval_sparse_code(ds_test.data[1],Ztest,decoder,l1w)
        eval_train = eval_sparse_code(ds_train.data[1],Ztrain,decoder,l1w)
        output = '\nEval Train\n'..serializeTable(eval_train)..'\n' 
        output = output..'Eval Test\n'..serializeTable(eval_test)..'\n' 
        local record_file = io.open(save_dir..'output.txt', 'a') 
        record_file:write(output..'\n') 
        record_file:close()
        results.train.loss[j] = eval_train.average_loss 
        results.train.rec[j] = eval_train.average_relative_rec_error  
        results.train.sparsity[j] = eval_train.average_sparsity 
        results.test.loss[j] = eval_test.average_loss 
        results.test.rec[j] = eval_test.average_relative_rec_error  
        results.test.sparsity[j] = eval_test.average_sparsity 
        repeat_loss_plot[j] = {arch.name,torch.range(1,epochs),loss_plot,'+'}
    end 
    --save in shared file  
    if type(arch.name) ~= 'FISTA' then 
        plot_path = paths.dirname(save_dir)..'/loss_plot.t7'
        if paths.filep(plot_path) == true then
            exp_plots = torch.load(plot_path)
        else 
            exp_plots = {} 
        end 
        n = #exp_plots+1
        exp_plots[n]=repeat_loss_plot
        torch.save(plot_path,exp_plots) 
    end
    results_path = paths.dirname(save_dir)..'/results.t7'
    if paths.filep(results_path) == true then
        exp_results = torch.load(results_path) 
    else 
        exp_results={} 
    end 
    n = #exp_results+1
    exp_results[n]=results
    torch.save(results_path,exp_results) 
    --pdf plots  
    plot_results(exp_results,paths.dirname(save_dir)..'/') 
end
--all configs done! 


