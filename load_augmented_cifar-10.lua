local image = require "image"
require "Set"
local cifar_10 = require("load_cifar-10")
-- out of every example of the trainset will make 5 extra: center, right-top, left-top, right-bottom, left-bottom
local function generate_augmented_trainset()
  trainset = cifar_10.trainset
  trainset:float()
  local augmented_data = torch.Tensor(trainset:size()*6, 3, 32, 32)
  local augmented_label = torch.Tensor(trainset:size()*6)
  local shuffle = torch.randperm(trainset:size()*6)
  for i=0,trainset:size()-1 do
    for j=1,5 do
      augmented_data[shuffle[i*6+j]] = image.scale(trainset.data[i+1][{{},{4,28},{4,28}}],32,32)
      augmented_label[shuffle[i*6+j]] = trainset.label[i+1]
    end
    augmented_data[shuffle[i*6+6]] = trainset.data[i+1]
    augmented_label[shuffle[i*6+6]] = trainset.label[i+1]
  end
  torch.save("data/augmented_cifar_10_trainset", {augmented_data=augmented_data, augmented_label=augmented_label})
  return augmented_data, augmented_label
end

local augmented_data, augmented_label
if paths.filep("data/augmented_cifar_10_trainset") then
  local data_label = torch.load("data/augmented_cifar_10_trainset")
  augmented_data = data_label.augmented_data
  augmented_label = data_label.augmented_label
else
  augmented_data, augmented_label = generate_augmented_trainset()
end

augmented_trainset = Set.new{data=augmented_data, label=augmented_label}
augmented_cifar_10 = {}
augmented_cifar_10.trainset = augmented_trainset
augmented_cifar_10.testset = cifar_10.testset
augmented_cifar_10.classes = cifar_10.classes
collectgarbage()
return augmented_cifar_10
