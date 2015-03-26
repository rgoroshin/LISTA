if nn.ReshapeTable == nil then 
    dofile('./Modules/ReshapeTable.lua') 
end 
if nn.ModuleL1Penalty == nil then 
    dofile('./Modules/ModuleL1Penalty.lua') 
end
if nn.NormLinear == nil then 
    dofile('./Modules/NormLinear.lua') 
end 
if nn.NormSpatialConvolution == nil then 
    dofile('./Modules/NormSpatialConvolution.lua') 
end 
