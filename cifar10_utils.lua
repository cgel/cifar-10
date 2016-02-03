require "optim"
require "image"
utils = {}

utils.opt = { batch_size = 64,
cuda = true,
--opt.train_size = 60000,
  test_size = 0,
  epochs = 2,
  eval_batch_size = 256,
  eval_iters = 5000,
  eval_train_iters = 1000,
  noise = 0.25,
  optimMethod = optim.sgd,
  optimState = {
    learningRate = 1,
    weightDecay = 1e-3,
    momentum = 0.9,
    learningRateDecay = 0
  }
}

utils.err = {any = false}

utils.feval_counter = 0

-- different data augmentation methods
utils.flip_augment = function(self, minibatch)
  --augmented = torch.FloatTensor(minibatch:size())
  for i = 1, minibatch:size(1) do
    if torch.rand(1)[1] > 0.5 then image.hflip(augmented[i], minibatch[i]) end
  end
  return augmented
end
utils.crop5_flip_augment = function(self, minibatch)
  augmented = torch.FloatTensor(minibatch:size())
  for i = 1, minibatch:size(1) do
    local crop_size = 5
    local x = torch.random(1, crop_size)
    local y = torch.random(1, crop_size)
    local x_crop = {x , 32 - crop_size + x}
    local y_crop = {y, 32 - crop_size + y}
    augmented[i] = image.scale(minibatch[i][{{}, x_crop, y_crop}],32,32)
    if torch.rand(1)[1] > 0.5 then image.hflip(augmented[i], augmented[i]) end
  end
  return augmented
end
utils.edge_crop_flip_augment = function(self, minibatch)
  augmented = torch.FloatTensor(minibatch:size())
  for i = 1, minibatch:size(1) do
    local crop_size = 26
    local crops = { {{1, crop_size},{1, crop_size}},
                  {{1, crop_size}, {33 - crop_size, 32}},
                  {{33 - crop_size, 32}, {1, crop_size, 32}},
                  {{33 - crop_size, 32}, {33 - crop_size, 32}},
                  {{(33-crop_size)/2, 33 - (33-crop_size)/2}, {(33-crop_size)/2, 33 - (33-crop_size)/2}}
                }
    local n = torch.random(1,5)
    augmented[i] = image.scale(minibatch[i][{{},crops[n][1],crops[n][2]}],32,32)
    if torch.rand(1)[1] > 0.5 then image.hflip(augmented[i], augmented[i]) end
  end
  return augmented
end
utils.rand_crop_flip_augment = function(self, minibatch)
  augmented = torch.FloatTensor(minibatch:size())
  for i = 1, minibatch:size(1) do
    local crop_size = torch.random(1,5)
    local x = torch.random(1, crop_size)
    local y = torch.random(1, crop_size)
    local x_crop = {x , 32 - crop_size + x}
    local y_crop = {y, 32 - crop_size + y}
    augmented[i] = image.scale(minibatch[i][{{}, x_crop, y_crop}],32,32)
    if torch.rand(1)[1] > 0.5 then image.hflip(augmented[i], augmented[i]) end
  end
  return augmented
end
utils.no_augment = function(self, minibatch)
  return minibatch
end
utils.augment = function(self)
  print("error: no augment method defined")
end


utils.train_epoch = function(self)
  utils.net:training()
  local last_iter = false
  local minibatch = {}
  minibatch.data = torch.CudaTensor(self.opt.batch_size, 3, 32, 32)
  minibatch.labels= torch.CudaTensor(self.opt.batch_size)
  local counter = 0
  local epoch_loss = 0
  while last_iter == false do
    local start_index = math.min(counter * utils.opt.batch_size + 1, trainset.data:size(1))
    local end_index = math.min((counter + 1)* utils.opt.batch_size, trainset.data:size(1))
    if end_index >= trainset:size() then
      last_iter = true
    end
    counter = counter + 1
    augmented = self:augment(trainset.data[{{start_index, end_index}}])
    minibatch.data[{{1, end_index - start_index + 1}}]:copy(augmented) -- now in gpu mem
    minibatch.labels:copy(trainset.label[{{start_index, end_index}}])

    local feval = function (x)
      if utils.parameters ~= x then
        print("x and parameters are different tensors")
        utils.parameters:copy(x)
      end
      utils.net:zeroGradParameters()
      minibatch.outputs = utils.net:forward(minibatch.data)
      minibatch.loss = utils.criterion:forward(minibatch.outputs, minibatch.labels)
      minibatch.dloss_doutput = utils.criterion:backward(minibatch.outputs, minibatch.labels)
      utils.net:backward(minibatch.data,  minibatch.dloss_doutput)
      if minibatch.loss == math.huge or minibatch.loss == -math.huge then
        print("err: loss is " .. minibatch.loss)
        last_iter = true
      end
      epoch_loss = epoch_loss + minibatch.loss
      return bm_loss, utils.gradParameters
    end
    utils.opt.optimMethod(feval, utils.parameters, utils.opt.optimState)
    -- DataParallelTable return true
    if(self.net.name ~= nil and self.net:name() == "DataParallelTable") then
      self.net:syncParameters()
    end
  end
  return epoch_loss/counter
end

utils.evaluate = function (self, set)
    self.net:evaluate()
    local size = set:size()
    local minib_counter = 0
    local loss_acc = 0
    local last_minibatch = false
    local prediction_list = torch.Tensor(size):cuda()
    local minibatch = {}
    local correct_count = 0
    while(last_minibatch ~= true) do
      local start_index = minib_counter * self.opt.eval_batch_size +1
      local end_index = math.min((minib_counter + 1)* self.opt.eval_batch_size, size)
       if end_index == size then
        last_minibatch = true
        else
        minib_counter = minib_counter +1
      end
      minibatch.data = set.data[{{start_index,end_index}}]:cuda()
      minibatch.labels = set.label[{{start_index,end_index}}]:cuda()
      minibatch.outputs = self.net:forward(minibatch.data)
      minibatch.loss = self.criterion:forward(minibatch.outputs, minibatch.labels)
      for j = 1, minibatch.data:size(1) do
        local groundtruth = minibatch.labels[j]
        local prediction = minibatch.outputs[j]
        local confidences, indices = torch.sort(prediction, true)
        if groundtruth == indices[1] then
          correct_count = correct_count +1
        end
      end
      loss_acc = loss_acc + minibatch.loss
      local _maxs, inds = minibatch.outputs:max(2)
      prediction_list[{{start_index, end_index}}] = inds[{{},1}]
    end
    loss_acc = loss_acc/(minib_counter+1)
    local hist = torch.Tensor(10):zero()
    prediction_list:apply(function(x)
            hist[x] = hist[x] +1
        end)
    local error_rate = 1 - correct_count/size
    return loss_acc, hist, error_rate
end

utils.visualize_example = function (self, set, i)
  print(classes[set.label[i]])
  itorch.image(image.scale(set.data[i],90,90))
  local predicted = utils.net:forward(set.data[i]:cuda())
  predicted:exp()
  local confidences, indices = torch.sort(predicted, true)
  for j = 1, predicted:size(1) do
      print(classes[indices[j]]..string.format("    %.3f%%", confidences[j]*100))
  end
end

utils.visualize = function (self, n)
  utils.net:evaluate()
  for i = 1, n do
    print("=======")
    utils:visualize_example(testset, math.ceil(torch.uniform(1,testset:size())))
  end
end


utils.class_error = function(set)
  local class_perf = {0,0,0,0,0,0,0,0,0,0}
  local t_size = set:size()
  for i = 1, t_size do
    local groundtruth = set.label[i]
    local prediction = utils.net:forward(set.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        class_perf[groundtruth] = class_perf[groundtruth] +1
    end
  end
  sorted_perf = torch.Tensor(class_perf):sort(true)
  for i = 1, 10 do
    print(classes[i], 1000*sorted_perf[i]/t_size..' %')
  end
end

utils.ping = function(self)
  os.execute("mpg123 ~/workspace/SVHN_torch/sounds/ping.mp3 &")
end


return utils
