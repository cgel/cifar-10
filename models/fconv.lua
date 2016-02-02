local net = nn.Sequential()
  local convwidths = {3, 96, 96, 192, 192, 192, 192, 10}

local function xavier_init(fan_in, fan_out)
    stddev = math.sqrt(2/(fan_in + fan_out))
    return stddev
end

local conv_count = 1
local function add_conv(k)
    local i = conv_count
    conv = cudnn.SpatialConvolution(convwidths[i], convwidths[i+1], k, k, 1, 1, 1, 1)
    conv:reset(xavier_init(conv.nInputPlane*conv.kH*conv.kW, conv.nOutputPlane*conv.kH*conv.kW))
  --  net:add(nn.BatchNormalization())
    net:add(conv)
    net:add(cudnn.ReLU())
    conv_count = conv_count+1
end

local maxPool_count = 0
local function add_maxPool()
  net:add(cudnn.SpatialMaxPooling(3,3,2,2,1,1))
  maxPool_count = maxPool_count + 1
end

local dropRate = 0.5

net:add(nn.Dropout(0.2))
add_conv(3)
add_conv(3)
add_maxPool()
net:add(nn.Dropout(dropRate))
add_conv(3)
add_conv(3)
add_maxPool()
net:add(nn.Dropout(dropRate))
add_conv(3)
add_conv(1)
add_conv(1)
net:add(nn.Sum(3)) -- Sum over height
net:add(nn.Sum(2,2)) -- Sum over width
net:add(cudnn.LogSoftMax())
  
return net
