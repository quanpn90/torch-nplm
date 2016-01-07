-- Trains a neural network n-gram language model
-- The model is based on Yoshua Bengio's neural probabilistic language model (2003)
require 'torch'
require 'nn'
require 'nngraph'
require 'fbcunn'
require 'fbnn'
require 'xlua'
require 'lfs'
local c = require 'trepl.colorize'
require 'optim'
require 'util.Squeeze'

LookupTable = nn.LookupTableGPU

local model_utils = require 'util.model_utils'
require 'util.misc'
local BatchLoader = require 'util.BatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a word-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain the file train.txt, valid.txt, test.txt')
-- model params
cmd:option('-hsm',-1,'number of clusters to use for hsm. 0 = normal softmax, -1 = use sqrt(|V|)')
cmd:option('-order', 6, 'ngram order')
cmd:option('-projection_size', 256, 'size of the projected word vectors')
cmd:option('-hidden_size', {1024, 256}, 'size of LSTM internal state')
-- optimization
cmd:option('-opt_method', 'sgd', 'optimization method to be used. Support sgd and rmsprop') --rmsprop reduces training speed by about 10%
cmd:option('-learning_rate',5e-1,'learning rate') -- needs to be very small for rmsprop
cmd:option('-learning_rate_decay',2,'learning rate decay')
cmd:option('-momentum', 0.9, 'momentum sgd parameter')
cmd:option('-weight_decay', 3e-5, 'weight decay parameter')
cmd:option('-learning_rate_decay_after',15,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.99,'decay rate for rmsprop')
cmd:option('-dropout',0.5,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',128,'number of sequences to train on in parallel')
cmd:option('-max_epochs',30,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-param_init', 0.05, 'initialized value for parameter matrices')
cmd:option('-load_checkpoint', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-coefL1', 0, 'L1 regularization')
cmd:option('-coefL2', 1e-3, 'L2 regularization')
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',1000,'how many steps/minibatches between printing out the loss')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','rnn','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',1,'which gpu to use. -1 = use CPU')
cmd:text()
-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

assert(opt.opt_method == 'sgd' or opt.opt_method == 'rmsprop', 'Only sgd and rmsprop is supported as the optimization method')
if (opt.opt_method == 'rmsprop' and opt.learning_rate > 1e-2 ) then
    print("Warning. learning rate too high for rmsprop optimization")
end

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0  then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name .. '...')
        cutorch.setDevice(opt.gpuid) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

local prev_layer_size
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.order)

vocab_size = loader.vocab_size
train_data = loader.all_batches[1]
valid_data = loader.all_batches[2]
test_data  = loader.all_batches[3]

local context_size = opt.order-1
-- Now we define the model

-- if number of clusters is not explicitly provided
if opt.hsm == -1 then
    opt.hsm = torch.round(torch.sqrt(#loader.idx2word))
end

-- Initialize the hierarchical softmax

if opt.hsm > 0 then
    -- partition into opt.hsm clusters
    -- we want roughly equal number of words in each cluster
    HSMClass = require 'util.HSMClass'
    require 'util.HLogSoftMax'
    mapping = torch.LongTensor(#loader.idx2word, 2):zero()
    local n_in_each_cluster = #loader.idx2word / opt.hsm -- number of entries in each cluster can be different ???
    local _, idx = torch.sort(torch.randn(#loader.idx2word), 1, true)   
    -- sorting words based on a normal distribution

    local n_in_cluster = {} --number of tokens in each cluster
    local c = 1
    for i = 1, idx:size(1) do
        local word_idx = idx[i] 
        if n_in_cluster[c] == nil then
            n_in_cluster[c] = 1
        else
            n_in_cluster[c] = n_in_cluster[c] + 1
        end
        mapping[word_idx][1] = c
        mapping[word_idx][2] = n_in_cluster[c]        
        if n_in_cluster[c] >= n_in_each_cluster then
            c = c+1
        end
        if c > opt.hsm then --take care of some corner cases
            c = opt.hsm
        end
    end
    print(string.format('using hierarchical softmax with %d classes', opt.hsm))
end


model = nn.Sequential()

projection_matrix = LookupTable(vocab_size, opt.projection_size)
model:add(projection_matrix)
model:add(nn.Reshape(opt.projection_size * context_size, true))

if opt.dropout > 0 then
	model:add(nn.Dropout(opt.dropout))
end

prev_layer_size = opt.projection_size * context_size

for L, hidden_size in pairs(opt.hidden_size) do
	model:add(nn.Linear(prev_layer_size, hidden_size))
	
	prev_layer_size = hidden_size
	
	model:add(nn.ReLU())
	
	if opt.dropout > 0 then
		model:add(nn.Dropout(opt.dropout))
	end

end
-- Projecting to the vocab distribution

if opt.hsm > 0 then 
  criterion = nn.HLogSoftMax(mapping, prev_layer_size)
else
  model:add(nn.Linear(prev_layer_size, vocab_size))
  model:add(nn.LogSoftMax())
  criterion =  nn.ClassNLLCriterion()
end

if opt.gpuid >= 0 then
	model:cuda()
	criterion:cuda()
end

function prepro(x,y)
    -- x = x:contiguous() -- swap the axes for faster indexing
    -- y = y:contiguous()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    
    return x,y
end

-- Test the forward
-- print(model:forward(torch.Tensor(5, opt.order-1):random(1, 100):cuda()))
-- Looks like its good now, hehe


params, grad_params = model:getParameters()
params = params:cuda()
grad_params = grad_params:cuda()
do_random_init = true
if do_random_init then
	params:uniform(-opt.param_init, opt.param_init) 
end

if opt.hsm > 0 then
    hsm_params, hsm_grad_params = criterion:getParameters()
    hsm_params:uniform(-opt.param_init, opt.param_init)
    print('number of parameters in the model: ' .. params:nElement() + hsm_params:nElement())
else
    print('number of parameters in the model: ' .. params:nElement())
end

if opt.opt_method == 'sgd' then
  optimState = {
    learning_rate = opt.learning_rate,
    weight_decay = opt.weight_decay,
    momentum = opt.momentum,
    learning_rate_decay = opt.learning_rate_decay,
  }
elseif opt.opt_method == 'rmsprop' then
  optimState = {
    learning_rate = opt.learning_rate,
    alpha = opt.decay_rate    
  }
end

iteration = 0
beginning_time = torch.tic()

function train()
  local total_samples = 0
  local start_time = torch.tic()
  model:training() -- Enter training mode
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  -- if epoch >= opt.learning_rate_decay_after  then 
  --   opt.learning_rate = opt.learning_rate/opt.learning_rate_decay 
  --   print(string.format("Decay learning rate to %f", opt.learning_rate))
  -- end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batch size = ' .. opt.batch_size .. ']')

  -- local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(train_data[1]:size(1)):long():split(opt.batch_size)
  -- print(indices)
  -- remove last element so that all the batches have equal size
  -- indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    iteration = iteration + 1
    total_samples = total_samples + opt.batch_size
    -- xlua.progress(t, #indices)

    -- t = batch index
    -- v = ?
    -- print(torch.type(provider.trainData.data))
    local inputs = train_data[1]:index(1,v)
    local cputargets = train_data[2]:index(1,v)
    local targets = torch.CudaTensor(cputargets:size(1))
    targets:copy(cputargets)
    inputs = inputs:cuda()

    local feval = function(x)
      
      if x ~= params then params:copy(x) end
      grad_params:zero()
      if opt.hsm > 0 then
          hsm_grad_params:zero()
      end

      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      
      model:backward(inputs, df_do)
      local norm, sign =torch.norm, torch.sign
      
      local loss = f
      -- local loss = f + opt.coefL2 * torch.norm(params,2)^2/2 + opt.coefL1 * torch.norm(params, 1) 
      
      -- grad_params:add(sign(params):mul(opt.coefL1) + params:clone():mul(opt.coefL2) )
      
      params:add(grad_params:mul(-opt.learning_rate))

      if opt.hsm > 0 then
            hsm_params:add(hsm_grad_params:mul(-opt.learning_rate))
      end

      -- local loss = f / opt.batch_size
      return loss, grad_params
    end
    
    loss, _ = feval(params)
    -- if opt.opt_method == 'sgd' then
    --   _, f = optim.sgd(feval, params, optimState)
    -- elseif opt.opt_method == 'rmsprop' then
    --   _, f = optim.rmsprop(feval, params, optimState)
    -- end

    -- local train_perp = torch.exp(f[1])
    -- train_perp = torch.exp(loss)

    if t % opt.print_every == 0 then
      wps = torch.floor(total_samples / torch.toc(start_time))
      local since_beginning = torch.round(torch.toc(beginning_time) / 60)
      -- print(wps)
      print(string.format("%d/%d (epoch %.3f), train loss= %6.8f, speed = %.4f words/s, trained for %d minutes", t, #indices, epoch + t / #indices, loss, wps, since_beginning))
      -- print(string.format("%d/%d (epoch %i), train_perp = %6.8f", t, #indices, epoch, train_perp))

    end

    if t % 10 == 0 then
      collectgarbage()
    end

  end

  -- confusion:updateValids()
  -- print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
  --       confusion.totalValid * 100, torch.toc(tic)))

  -- train_acc = confusion.totalValid * 100

  -- confusion:zero()
  epoch = epoch + 1
end

function eval(split_index)
  if opt.hsm > 0 then
        criterion:change_bias()
  end
  model:evaluate()
  data = loader.all_batches[split_index]
  local test_indices = torch.Tensor(data[1]:size(1))
  i = 0
  test_indices:apply(function() i = i + 1 return i end)
  test_indices = test_indices:long():split(opt.batch_size)

  total_loss = 0
  for t,v in ipairs(test_indices) do
    xlua.progress(t, #test_indices)
    local inputs =data[1]:index(1,v)
    local targets = data[2]:index(1,v)
    inputs, targets = prepro(inputs, targets)
    local outputs = model:forward(inputs)

    total_loss = total_loss + criterion:forward(outputs, targets)
  end

  total_loss = total_loss / #test_indices
  perplexity = torch.exp(total_loss)

  print(string.format("Perplexity for eval split %i is %6.8f", split_index, perplexity))
    -- confusion:batchAdd(outputs, targets)
  

  return perplexity



end

eval(2)
for i=1,opt.max_epochs do
  train()
  current_perp = eval(2)

  if prev_perp == nil then
    prev_perp = current_perp
  end

  if current_perp > prev_perp then
    opt.learning_rate = opt.learning_rate / opt.learning_rate_decay
    print(string.format("Decay learning rate to %f", opt.learning_rate))
  end
  
  prev_perp = current_perp
end

eval(3)







