--=====================Data Container Class======================= 
--================================================================
--Functions for loading various datasets: NORB, MNIST, CIFAR, etc
--++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
--[dtype] = type of data to load 
--[dir] = table of directories which contain image sequences 
--[psz] = patch size 
--++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
local DataSource = torch.class('DataSource')

function DataSource:__init(arg)  
   
    local dataset = arg.dataset 
    local targets = arg.targets 
    local group_idx = arg.group_idx 
    local dtype = arg.dtype 
    local dir = arg.dir 
    local psz = arg.psz
    local color = arg.color
    local batchSize = arg.batchSize 

    self.group_idx = group_idx 
    self.batchSize = batchSize or 1 
   
    if type(dataset) == 'table' then 
        self.nframes = #dataset 
        self.nsamples = dataset[1]:size(1) 
        self.data = dataset
        self.targets = targets
    else 
        if group_idx ~= nil then 
            self.nsamples = group_idx:size(1) 
            self.nframes = group_idx:size(2) 
        else 
            self.nsamples = dataset:size(1) 
            self.nframes = 1 
        end 
        self.data = {dataset}
        if targets ~= nil then 
            self.targets = {targets}
        end 
    end 
    
    if arg.dataset == nil then 
        if dtype == nil then 
            self.dataType = 'none' 
            self.nsamples = 0
        elseif dtype == 'video patches' then  
            psz = psz or 20
            self:load(dir, psz, color) 
        else 
            print('ERROR: no data loaded') 
        end 
    
    else 

       self.idx = 1 
       self:shuffle() 

    end 

end

function DataSource:size() 

    return self.nsamples 

end

function DataSource:load(dir, psz, color)

    self.dataType = 'video patches'
    self.data = load_movie_patches(dir, psz, color):cuda() 
    self.nsamples = self.data:size(1) 
    self.idx = 1
    self:shuffle() 
    
end

function DataSource:shuffle()
    local n 
    if self.group_idx then 
       n = math.floor(self.group_idx:size(1)/self.batchSize)  
    else 
       n = math.floor(self.data[1]:size(1)/self.batchSize)  
    end 
    self.order = torch.randperm(n*self.batchSize):unfold(1, self.batchSize, self.batchSize)  
    self.nsamples = self.order:size(1) 
end

function DataSource:next() 
   
    if self.order == nil and self.data then 
        self:shuffle(self.batchSize) 
    elseif self.data == nil then 
        error('ERROR: data source contains no data') 
    end

    if (self.currentBatch == nil) or (self.currentBatch[1]:size(1) ~= self.batchSize) then 
       local dsz = self.data[1]:size()
       dsz[1] = self.batchSize 
       self.currentBatch = {} 
       for i = 1,self.nframes do 
        self.currentBatch[i] = torch.CudaTensor(dsz)
       end
       
       if self.targets then 
          local tsz = self.targets[1]:size()
          tsz[1] = self.batchSize 
          self.currentBatchTargets = {} 
          for i = 1,self.nframes do 
            self.currentBatchTargets[i] = torch.CudaTensor(tsz)  
          end 
       end
    end

    if self.idx < self.nsamples then    
        
        if self.group_idx~= nil then 
            --frame groups indexed by group_idx 
            for j = 1, self.nframes do 
                for i = 1, self.batchSize do 
                        local frame_idx = self.group_idx[self.order[self.idx][i]][j]  
                        self.currentBatch[j][i]:copy(self.data[1][frame_idx]) 
                    if self.targets then 
                        local target_idx = self.group_idx[self.order[self.idx][i]][j]  
                        self.currentBatchTargets[j][i] = self.targets[1][target_idx]  
                    end
                end  
            end
        else 
            --frame groups stored in a table 
            for j = 1, self.nframes do 
                for i = 1, self.batchSize do 
                        self.currentBatch[j][i]:copy(self.data[j][self.order[self.idx][i]]) 
                    if self.targets then 
                        self.currentBatchTargets[j][i] = self.targets[j][self.order[self.idx][i]]  
                    end
                end  
            end
        end
        
        self.idx = self.idx + 1 
    else
        --print('shuffling') 
        self:shuffle() 
        self.idx = 1 
        self:next() 
    end 
    
    if self.nframes == 1 and type(self.currentBatch) == 'table' then 
        self.currentBatch = self.currentBatch[1] 
        if self.targets and type(self.currentBatchTargets) == 'table' then 
            self.currentBatchTargets = self.currentBatchTargets[1]
        end 
    end 

    return self.currentBatch, self.currentBatchTargets 

end 
