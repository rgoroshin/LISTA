x = torch.load('results2.t7') 
train_loss = torch.Tensor(#x)
test_loss = torch.Tensor(#x)
scatter_train = torch.Tensor(#x,2) 
scatter_test = torch.Tensor(#x,2) 
exp_labels = {} 

for i = 1,#x do 
    local tied_weights
    local nlayers 
    if x[i].config.name ~= 'FISTA' then 
        if x[i].config.tied_weights ~= nil then 
            print(i..' 1') 
            tied_weights = x[i].config.tied_weights 
            nlayers = x[i].config.nlayers
        elseif x[i].config.untied_weights ~= nil then 
            print(i..' 2') 
            tied_weights = not x[i].config.untied_weights 
            nlayers = x[i].config.nloops
        else 
            error('????') 
        end 
        exp_labels[i] = x[i].config.name..nlayers..'_'..tostring(tied_weights) 
    else 
        print(i..' 3') 
        exp_labels[i] = x[i].config.name..x[i].config.niter
    end
    train_loss[i] = x[i].train.average_loss 
    test_loss[i] = x[i].test.average_loss
    scatter_train[i][1] = x[i].train.average_relative_rec_error
    scatter_train[i][2] = x[i].train.average_sparsity 
    scatter_test[i][1] = x[i].test.average_relative_rec_error
    scatter_test[i][2] = x[i].test.average_sparsity 
end 
