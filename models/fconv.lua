local net = nn.Sequential()
  --local convwidths = {3, 64, 128, 256, 256, 256, 256, 256, 256, 256, 256}
  local convwidths = {3, 64, 128, 256, 256, 256, 256, 256, 256, 256, 10}
  local fullwidths = {512, 256}

local function xavier_init(fan_in, fan_out)
    stddev = math.sqrt(2/(fan_in + fan_out))
    return stddev
end

local conv_count = 1
local function add_conv()
    local i = conv_count
    conv = cudnn.SpatialConvolution(convwidths[i], convwidths[i+1], 3, 3, 1, 1, 1, 1)
    conv:reset(xavier_init(conv.nInputPlane*conv.kH*conv.kW, conv.nOutputPlane*conv.kH*conv.kW))
    net:add(conv)
    net:add(cudnn.ReLU())
    conv_count = conv_count+1
end

local maxPool_count = 0
local function add_maxPool()
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))
  maxPool_count = maxPool_count + 1
end

local function add_full(in_width, out_with)
    net:add(nn.Linear(in_width, out_with))
    net:add(cudnn.ReLU())
end

local function view_size()
  return convwidths[#convwidths] * 2^(2*(5-maxPool_count))
end

local dropRate = 0.3

add_conv()
add_conv()
net:add(nn.Dropout(dropRate))
add_conv()
add_conv()
add_maxPool()
net:add(nn.Dropout(dropRate))

add_conv()
add_conv()
net:add(nn.Dropout(dropRate))
add_conv()
add_conv()
add_maxPool()
net:add(nn.Dropout(dropRate))

add_conv()
add_conv()
net:add(nn.Dropout(dropRate))

net:add(nn.Sum(3)) -- Sum over height
net:add(nn.Sum(3)) -- Sum over width
net:add(cudnn.LogSoftMax())

return net
