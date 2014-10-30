local ReshapeTable, parent = torch.class('nn.ReshapeTable', 'nn.Module')

function ReshapeTable:__init(fw, bw)
   parent.__init(self)

 if type(fw) == 'function' and type(bw) == 'function' then  
    self.fw = fw
    self.bw = bw 
 else 
    error('fw and/or bw methods undefined!') 
 end 

end

function ReshapeTable:updateOutput(input)
   self.output = self.fw(input) 
   return self.output
end

function ReshapeTable:updateGradInput(input, gradOutput)
    self.gradInput = self.bw(gradOutput) 
    return self.gradInput
end
