require "optim"
require "image"
utils = {}

utils.opt = { batch_size = 64,
  cuda = true,
--opt.train_size = 60000,
  test_size = 0,
  epochs = 2,
  eval_batch_size = 256,
  eval_batch_size = 32,
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
-- Requires model that takes a 32x32 input
utils.feval = function (x)
  if utils.parameters ~= x then
    print("x and parameters are different tensors")
    utils.parameters:copy(x)
  end
  local noise = utils.opt.noise
  local start_index = utils.feval_counter * utils.opt.batch_size +1
  if start_index >= trainset:size() then
      utils.feval_counter = 0
      start_index = 1
  end
  local end_index = math.min((utils.feval_counter + 1)* utils.opt.batch_size, trainset.data:size(1))
  if end_index == trainset.data:size(1) then
    utils.feval_counter = 0;
  else
    utils.feval_counter = utils.feval_counter +1
  end
  local minibatch = {}
  minibatch.data = trainset.data[{{start_index,end_index}}]
  minibatch.labels = trainset.label[{{start_index,end_index}}]
  utils.net:zeroGradParameters()
  utils.net:training()
  minibatch.outputs = utils.net:forward(minibatch.data)
  minibatch.loss = utils.criterion:forward(minibatch.outputs, minibatch.labels)
  minibatch.dloss_doutput = utils.criterion:backward(minibatch.outputs,minibatch.labels)
  utils.net:backward(minibatch.data,  minibatch.dloss_doutput)
  if minibatch.loss ~= minibatch.loss or minibatch.loss == math.huge or minibatch.loss == -math.huge then
    print("err: loss is " .. minibatch.loss)
    utils.err.any = true
  end
  return minibatch.loss, utils.gradParameters
end

-- Requires model that takes a 24x24 input
utils.feval_24 = function (x)
  if utils.parameters ~= x then
    print("x and parameters are different tensors")
    utils.parameters:copy(x)
  end
  local noise = utils.opt.noise
  local start_index = utils.feval_counter * utils.opt.batch_size +1
  if start_index >= trainset:size() then
      utils.feval_counter = 0
      start_index = 1
  end
  local end_index = math.min((utils.feval_counter + 1)* utils.opt.batch_size, trainset.data:size(1))
  if end_index == trainset.data:size(1) then
    utils.feval_counter = 0;
  else
    utils.feval_counter = utils.feval_counter +1
  end
  local minibatch = {}
  local raw_data = trainset.data[{{start_index,end_index}}]
  minibatch.labels = trainset.label[{{start_index,end_index}}]
  minibatch.data = torch.CudaTensor(utils.opt.batch_size, 3,24,24)
  for i = 1, raw_data:size(1) do
    local x1,x2,y1,y2
    x1 = torch.random(1,8)
    y1 = torch.random(1,8)
    x2 = x1 + 23
    y2 = y1 + 23
    minibatch.data[i]:copy(raw_data[i][{{},{x1,x2},{y1,y2}}])
  end
  utils.net:zeroGradParameters()
  utils.net:training()
  minibatch.outputs = utils.net:forward(minibatch.data)
  minibatch.loss = utils.criterion:forward(minibatch.outputs, minibatch.labels)
  minibatch.dloss_doutput = utils.criterion:backward(minibatch.outputs,minibatch.labels)
  utils.net:backward(minibatch.data,  minibatch.dloss_doutput)
  if minibatch.loss ~= minibatch.loss or minibatch.loss == math.huge or minibatch.loss == -math.huge then
    print("err: loss is " .. minibatch.loss)
    utils.err.any = true
  end
  return minibatch.loss, utils.gradParameters
end

utils.evaluate = function (set)
    utils.net:evaluate()
    local size = utils.opt.eval_iters
    if size == 0 or size > set:size() then size = set:size() end
    local minib_counter = 0
    local loss_acc = 0
    last_minibatch = false
    if utils.opt.cuda then
      prediction_list = torch.Tensor(size):cuda()
    else
      prediction_list = torch.Tensor(size)
    end
    local minibatch = {}
    while(last_minibatch ~= true) do
      local start_index = minib_counter * utils.opt.eval_batch_size +1
      local end_index = math.min((minib_counter + 1)* utils.opt.eval_batch_size, size)
       if end_index == size then
        last_minibatch = true
        else
        minib_counter = minib_counter +1
      end
      minibatch.data = set.data[{{start_index,end_index}}]
      minibatch.labels = set.label[{{start_index,end_index}}]
      utils.net:zeroGradParameters()
      minibatch.outputs = utils.net:forward(minibatch.data)
      minibatch.loss = utils.criterion:forward(minibatch.outputs, minibatch.labels)
      loss_acc = loss_acc + minibatch.loss
      _maxs, inds = minibatch.outputs:max(2)
      prediction_list[{{start_index, end_index}}] = inds[{{},1}]
    end
    loss_acc = loss_acc/(minib_counter+1)
    local hist = torch.Tensor(10):zero()
    prediction_list:apply(function(x)
            hist[x] = hist[x] +1
        end)
    return loss_acc, hist
end

utils.update  = function ()
  return utils.opt.optimMethod(utils.feval,utils.parameters, utils.opt.optimState)
end

utils.visualize_example = function (set, i)
  print(classes[set.label[i]])
  itorch.image(set.data[i])
  local predicted = utils.net:forward(set.data[i])
  predicted:exp()
  local confidences, indices = torch.sort(predicted, true)
  for j = 1, predicted:size(1) do
      print(classes[indices[j]]..string.format("    %.3f%%", confidences[j]*100))
  end
end

utils.visualize = function (n)
  utils.net:evaluate()
  for i = 1, n do
    print("=======")
    utils.visualize_example(testset, math.ceil(torch.uniform(1,testset:size())))
  end
end

utils.error_rate = function(set)
  utils.net:evaluate()
  local correct = 0
  local batch_size = 256
  local size = testset:size()
  local count = 0
  for i = 0, math.ceil(size/batch_size) - 1 do
    local start_index = count * batch_size + 1
    if start_index >= size then
      count = 0
      start_index = 1
    end
    local end_index = math.min( start_index + batch_size - 1, size)
    count = count + 1
    local minibatch = {}
    minibatch.data = set.data[{{start_index,end_index}}]
    minibatch.labels = set.label[{{start_index,end_index}}]
    minibatch.outputs = utils.net:forward(minibatch.data)
    for j = 1, minibatch.data:size(1) do
      local groundtruth = minibatch.labels[j]
      local prediction = minibatch.outputs[j]
      local confidences, indices = torch.sort(prediction, true)
      if groundtruth == indices[1] then
        correct = correct +1
      end
    end
  end
  percent = 100 * correct/size
  print("testset ", percent,'%')
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

return utils
