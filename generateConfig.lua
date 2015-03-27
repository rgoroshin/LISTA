generateConfig = function(arg,start_idx) 
    
    print('placing config files in '..arg.config_dir) 
    local exp_idx = start_idx or 1
    --misc 
    for _,exp_name in ipairs(arg.exp_name) do 
    for _,small_exp in ipairs(arg.small_exp) do 
    --data
    for _,dataset in ipairs(arg.dataset) do
    --loss 
    for _,l1w in ipairs(arg.l1w) do  
    --hyper-parameter optimization 
    for _,resolution in ipairs(arg.resolution) do 
    for _,depth in ipairs(arg.depth) do 
    for _,rand_grid in ipairs(arg.rand_grid) do 
    --learning 
    for _,epochs in ipairs(arg.epochs) do 
    for _,bsz in ipairs(arg.bsz) do 
    for _,repeat_exp in ipairs(arg.repeat_exp) do
    --experiment specific options: architecture, etc. 
    for _,config in ipairs(arg.configs) do
    output = '--==misc==\n'
    output = output..'exp_name='..'\''..tostring(exp_name)..'/config'..tostring(exp_idx)..'\'\n'   
    output = output..'small_exp='..tostring(small_exp)..'\n'   
    output = output..'--==dataset==\n'  
    output = output..'dataset='..'\''..dataset..'\'\n'
    output = output..'--==loss==\n'  
    output = output..'l1w='..l1w..'\n'  
    output = output..'--==learning==\n'  
    output = output..'epochs='..tostring(epochs)..'\n'  
    output = output..'bsz='..tostring(bsz)..'\n'  
    output = output..'repeat_exp='..tostring(repeat_exp)..'\n'  
    output = output..'--==architectures==\n'  
    output = output..'arch='..serializeTable(config)..'\n'  
    
    config_file = io.open(arg.config_dir..'config'..tostring(exp_idx)..'.lua', 'w') 
    config_file:write(output..'\n') 
    config_file:close()
    exp_idx = exp_idx + 1 

    end end end end end end end end end end end
    
    return exp_idx-1

end
