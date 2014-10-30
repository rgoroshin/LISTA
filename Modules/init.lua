if nn.ReshapeTable == nil then 
    dofile('./Modules/ReshapeTable.lua') 
end 
if nn.L1Penalty == nil then 
    dofile('./Modules/L1Penalty.lua') 
end
if nn.NormLinear == nil then 
    dofile('./Modules/NormLinear.lua') 
end 
if nn.NormSpatialConvolutionFFT == nil then 
    dofile('./Modules/NormSpatialConvolutionFFT.lua') 
end 
