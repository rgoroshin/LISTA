dofile('init.lua') 
cutorch.setDevice(1)

decoder = torch.load('./Results/Trained_Networks/FISTA_decoder.t7') 

We = decoder:get(2).weight:float()
inplane = We:size(1) 
outplane = We:size(2)
k = We:size(3) 
Sw = torch.Tensor(outplane,outplane,2*k-1,2*k-1) 
for i = 1,outplane do
    Sw[i] = torch.conv2(We:select(2,i),flip(We),'F') 
end

--power method
k2 = 2*k-1
input = norm_filters(torch.rand(32,32,17,17)) 
input_prev = input:clone() 
output = input:clone():zero()
--Sw = flip(Sw) 
n = 100

for i = 1,n do 
    progress(i,n)
    for i = 1,outplane do
        output[i] = torch.conv2(input:select(2,i),Sw,'F'):narrow(2,(k2-1)/2,k2):narrow(3,(k2-1)/2,k2) 
    end
    input_prev:copy(input) 
    input:copy(norm_filters(output))
end

L = output:norm()


