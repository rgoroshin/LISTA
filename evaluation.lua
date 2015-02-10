eval_sparse_code = function(X,Z,decoder,_l1w) 
    local X = X:float() 
    local Z = Z:float() 
    local Xr = transform_data(Z,decoder) 
    local Xsqerr = Xr:add(-1,X):pow(2)
    local rec_mse = Xsqerr:mean() 
    local rel_rec_mse = rec_mse/X:pow(2):mean()
    local loss = 0.5*rec_mse + _l1w*(Z:abs():mean())
    local sparsity = 1-(Z:gt(0):sum()/Z:nElement())
    X = nil 
    Z = nil
    collectgarbage() 
    return {average_loss=loss, average_relative_rec_error=rel_rec_mse, average_sparsity=sparsity}
end

