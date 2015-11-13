require "cunn"
DATA = torch.load("data/train_x.bin"):cuda()
Y_one_hot = torch.load("data/train_y.bin")
local _, indices = torch.sort(Y_one_hot, true)
LABELS = indices[{{},1}]:clone():cuda()

local CUT = 40000
local NUM = 50000

local trainset = {}
trainset.data = DATA[{{1, CUT}}]
trainset.label = LABELS[{{1, CUT}}]

local testset = {}
testset.data = DATA[{{CUT + 1, NUM}}]
testset.label = LABELS[{{CUT + 1, NUM}}]

set_metatable = { __index = function(t,i) return {t.data[i], t.label[i]} end }
setmetatable(trainset, set_metatable)
function trainset:size() return self.data:size(1) end
setmetatable(testset, set_metatable)
function testset:size() return self.data:size(1) end

local classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
data = {trainset, testset, classes}
data.trainset = trainset
data.testset = testset
data.classes = classes
return data
