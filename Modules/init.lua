if nn.ReshapeTable == nil then 
    dofile('./Modules/ReshapeTable.lua') 
end 
if nn.ModuleL1Penalty == nil then 
    dofile('./Modules/ModuleL1Penalty.lua') 
end
if nn.NormLinear == nil then 
    dofile('./Modules/NormLinear.lua') 
end 
if nn.NormSpatialConvolutionMM == nil then 
    dofile('./Modules/NormSpatialConvolutionMM.lua') 
end 
