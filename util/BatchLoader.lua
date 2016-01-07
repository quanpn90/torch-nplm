-- Modified from https://github.com/karpathy/char-rnn
-- This version is for cases where one has already segmented train/val/test splits

local BatchLoader = {}
local stringx = require('pl.stringx')
local file = require('pl.file')
BatchLoader.__index = BatchLoader
utf8 = require 'lua-utf8'

function BatchLoader.create(data_dir, batch_size, ngram_order)
	local self = {}
	setmetatable(self, BatchLoader)

	local tail = tostring(ngram_order) .. "grams"
	local train_file = path.join(data_dir, 'train.' .. tail)
	local valid_file = path.join(data_dir, 'valid.' .. tail)
	local test_file = path.join(data_dir, 'test.' .. tail)
	local input_files = {train_file, valid_file, test_file}
	local vocab_file = path.join(data_dir, 'vocab.t7')
	local tensor_file = path.join(data_dir, 'tensor.t7')

	if not (path.exists(train_file) or path.exists(valid_file) or path.exists(test_file)) then
		print ("You need to convert the text file into ngram files first.")
		assert(1 == 2, "Program terminated.")
	end

	if not (path.exists(vocab_file) or path.exists(tensor_file)) then
		print("Setup vocabulary and preprocessing train/valid/test tensors in dir:" .. data_dir)
		BatchLoader.text2Tensor(input_files, vocab_file, tensor_file, ngram_order)
	end

	print("Loading data files...")
	local all_data = torch.load(tensor_file) --train, valid, test tensors
	local vocal_mapping = torch.load(vocab_file)
	self.idx2word, self.word2idx, self.word_freq = table.unpack(vocal_mapping)
	self.vocab_size = #self.idx2word
	print(string.format("Vocab size: %d", #self.idx2word))

	self.split_sizes = {}
	self.batch_size = batch_size
	self.all_batches = {}



	
	print ('Reshaping tensors...')
	local x_batches, y_batches, nbatches

	for split, data in ipairs(all_data) do
		local len = data:size(1)

		-- convert the data array (1D) into n_samples * ngram tensor
		x_batches = data:view(len/ngram_order, -1)
		-- take the last column as the label
		y_batches = x_batches:select(2, ngram_order):clone()
		-- take the remaining columns as inputs
		x_batches = x_batches:sub(1, x_batches:size(1), 1, ngram_order-1)
		assert(x_batches:size(1) == y_batches:size(1), "Number of inputs and labels must be equal")
		nbatches = x_batches:size(1)
		self.split_sizes[split] = nbatches
		self.all_batches[split] = {x_batches, y_batches}



	-- 	if split < 2 then
	-- 		x_batches = data:view(batch_size, -1):split(seq_length, 2)
	-- 		y_batches = ydata:view(batch_size, -1):split(seq_length, 2)
	-- 		nbatches = #x_batches
	-- 		self.split_sizes[split] = nbatches
	-- 		assert(#x_batches == #y_batches)

	-- 	else -- for valid and test we repeat dimensions to batch size (easy but inefficient)
	-- 		x_batches = {data:resize(1, data:size(1)):expand(batch_size, data:size(2))}
	-- 		y_batches = {ydata:resize(1, ydata:size(1)):expand(batch_size, ydata:size(2))}
	-- 		self.split_sizes[split] = 1 -- only one batch

	-- 	end

	-- 	self.all_batches[split] = {x_batches, y_batches}
	end 

	
	self.ntrain = self.split_sizes[1]
	print(string.format('data load done. Number of samples in train: %d, val: %d, test: %d', 
          self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()

	return self
end


-- function BatchLoader:reset_batch_pointer(split_idx, batch_idx)
--     batch_idx = batch_idx or 0
--     self.batch_idx[split_idx] = batch_idx
-- end

-- function BatchLoader:next_batch(split_idx)
--     -- split_idx is integer: 1 = train, 2 = val, 3 = test
--     self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
--     if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
--         self.batch_idx[split_idx] = 1 -- cycle around to beginning
--     end
--     -- pull out the correct next batch
--     local idx = self.batch_idx[split_idx]
--     return self.all_batches[split_idx][1][idx], self.all_batches[split_idx][2][idx]
-- end



function BatchLoader.text2Tensor(input_files, out_vocabfile, out_tensorfile, order)

	local f
	local output_tensors = {} 
	local idx2word = {}
    local word2idx = {}

    local wid
    for	split = 1,3 do
    	print(input_files[split])
		local data = file.read(input_files[split])
		data = stringx.replace(data, '\n', ' ')
		data = stringx.split(data)

		print(string.format("Loading %s, size of data = %d", input_files[split], #data))
		assert(#data % order == 0, "Datasize invalid")
		output_tensors[split] = torch.LongTensor(#data)
		for i = 1, #data do
			-- put a word into dictionary
			if word2idx[data[i]] == nil then
				idx2word[#idx2word + 1] = data[i]
				word2idx[data[i]] = #idx2word
			end

			-- convert word to tensor
			wid = word2idx[data[i]]
			output_tensors[split][i] = wid

		end

    end

    print "done"
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, {idx2word, word2idx})
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, output_tensors)

end

return BatchLoader