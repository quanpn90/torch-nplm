require 'nn'

do

	local Linear, parent = torch.class('nn.IdentityLinear', 'nn.Linear')


	-- override the constructor to have the additional range of initialization
    function Linear:__init(inputSize, outputSize)
        parent.__init(self,inputSize,outputSize)
                
        self:reset(inputSize, outputSize)
    end
    
    -- override the :reset method to use custom weight initialization.        
    function Linear:reset(inputSize, outputSize)
        
        if inputSize and outputSize then
            self.weight = torch.eye(inputSize, outputSize)
        else
            self.weight = torch.eye(300, 300)
        end
        self.bias:zero()
    end

    function Linear:init_params(inputSize, outputSize)

        if inputSize and outputSize then
            self.weight = torch.eye(inputSize, outputSize)
        else
            self.weight = torch.eye(300, 300)
        end
        self.bias:zero()

    end


end