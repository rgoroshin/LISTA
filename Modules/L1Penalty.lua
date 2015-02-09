local ModuleL1Penalty, parent = torch.class('nn.ModuleL1Penalty','nn.Module')

--This module acts as an L1 latent state regularizer, adding the 
--[gradOutput] to the gradient of the L1 loss. The [input] is copied to 
--the [output]. 

function ModuleL1Penalty:__init(provideOutput,weight,sizeAverage)
    parent.__init(self)
    self.provideOutput = provideOutput or false
    self.l1weight = weight or 0.5  
    self.sizeAverage = sizeAverage or false  
    self.L1Cost = torch.Tensor(1) 
end

function ModuleL1Penalty:updateOutput(input)
    local m = self.l1weight 
    if self.sizeAverage == true then 
      m = m/input:nElement()
    end
    local loss = m*input:norm(1) 
    self.L1Cost:fill(loss) 
    self.output = input 
    if self.provideOutput == true then 
        return self.output 
    end
end

function ModuleL1Penalty:updateGradInput(input, gradOutput)
    local m = self.l1weight 
    if self.sizeAverage == true then 
      m = m/input:nElement() 
    end
    
    self.gradInput:resizeAs(input):copy(input):sign():mul(m)
    
    if self.provideOutput == true then 
        self.gradInput:add(gradOutput)  
    end 

    return self.gradInput 
end

