require 'FFTconv'
require 'cunn'

if nn.NormSpatialConvolutionFFT ~= nil then
   error("NormSpatialConvolutionFFT already exists. Update/clean cache your torch installation")
end

local NormSpatialConvolutionFFT, parentCSFFT = torch.class('nn.NormSpatialConvolutionFFT', 'nn.Module')

function NormSpatialConvolutionFFT:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parentCSFFT.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
   self.tmp = torch.Tensor()

   self:reset()
end

function NormSpatialConvolutionFFT:norm_filters() 
    local w = self.weight 
    local wsz = w:size()
    w = w:transpose(1,2):contiguous():resize(wsz[2],wsz[1]*wsz[3]*wsz[4]) 
    local norm = w:norm(2,2):expandAs(w):contiguous() 
    w:cdiv(norm)
    self.weight:copy(w:resize(wsz[2],wsz[1],wsz[3],wsz[4]):transpose(1,2):contiguous())   
    collectgarbage() 
end

function NormSpatialConvolutionFFT:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end) 
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function NormSpatialConvolutionFFT:updateOutput(input)
   self:norm_filters()  
   g_fft_tmp = g_fft_tmp or torch.CudaTensor()  -- This is ugly, make tmp global
   libFFTconv.cu_SpatialConvolutionFFT_updateOutput(self, input, g_fft_tmp)
   return self.output
end

function NormSpatialConvolutionFFT:updateGradInput(input, gradOutput)
   g_fft_tmp = g_fft_tmp or torch.CudaTensor()  -- This is ugly, make tmp global
   if gradOutput:dim() == 3 then
      if (gradOutput:size(2) ~= input:size(2) - self.kH + 1) or
         (gradOutput:size(3) ~= input:size(3) - self.kW + 1) then
	    error("gradOutput has wrong size")
      end
   elseif gradOutput:dim() == 4 then
      if (gradOutput:size(1) ~= input:size(1)) or
	 (gradOutput:size(3) ~= input:size(3) - self.kH + 1) or
         (gradOutput:size(4) ~= input:size(4) - self.kW + 1) then
	    error("gradOutput has wrong size")
      end
   else
      error("gradOutput must be 3D or 4D")
   end
   libFFTconv.cu_SpatialConvolutionFFT_updateGradInput(self, gradOutput, 
     g_fft_tmp)
   return self.gradInput
end

function NormSpatialConvolutionFFT:accGradParameters(input, gradOutput, scale)
   g_fft_tmp = g_fft_tmp or torch.CudaTensor()  -- This is ugly, make tmp global
   scale = scale or 1
   libFFTconv.cu_SpatialConvolutionFFT_accGradParameters(self, input,
							 gradOutput, g_fft_tmp, scale)
end
