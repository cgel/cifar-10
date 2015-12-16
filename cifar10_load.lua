require "cunn"
require "Set"

DATA = torch.load("data/train_x.bin")
Y_one_hot = torch.load("data/train_y.bin")
local _, indices = torch.sort(Y_one_hot, true)
LABELS = indices[{{},1}]:clone()

local NUM = 50000 -- the size of the whole set
local CUT = 40000 -- the number of examples used for training

local trainset = Set.new{data = DATA[{{1, CUT}}], label = LABELS[{{1, CUT}}]}

local testset = Set.new{ data = DATA[{{CUT + 1, NUM}}], label = LABELS[{{CUT + 1, NUM}}]}

local classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
data = {trainset, testset, classes}
data.trainset = trainset
data.testset = testset
data.classes = classes
return data
