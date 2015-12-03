local net = nn.Sequential()
  local convwidths = {3, 64, 128, 256, 256, 256, 256, 256,256}
  local fullwidths = {512, 256, 128}

local function xavier_init(fan_in, fan_out)
    stddev = math.sqrt(2/(fan_in + fan_out))
    return stddev
end

local function add_conv(i)
    conv = nn.SpatialConvolution(convwidths[i], convwidths[i+1], 3, 3, 1, 1, 1, 1)
    conv:reset(xavier_init(conv.nInputPlane*conv.kH*conv.kW, conv.nOutputPlane*conv.kH*conv.kW))
    net:add(conv)
    net:add(nn.ReLU())
end

local function add_maxPool()
  net:add(nn.SpatialMaxPooling(2,2,2,2))
end

local function add_maxPool3()
  net:add(nn.SpatialMaxPooling(3,3,3,3))
end

local function add_full(in_width, out_with)
    net:add(nn.Linear(in_width, out_with))
    net:add(nn.ReLU())
end

local dropRate = 0.6

add_conv(1)
net:add(nn.Dropout(0.3))
add_conv(2)
add_maxPool()
net:add(nn.Dropout(dropRate))

add_conv(3)
add_conv(4)
add_maxPool()
net:add(nn.Dropout(dropRate))

add_conv(5)
add_conv(6)
add_maxPool()
net:add(nn.Dropout(dropRate))

add_conv(7)
add_conv(8)
add_maxPool3()
net:add(nn.Dropout(dropRate))

net:add(nn.View(convwidths[#convwidths]))
net:add(nn.View(convwidths[#convwidths]))

net:add(nn.Linear(convwidths[#convwidths], fullwidths[1]))
net:add(nn.ReLU())
net:add(nn.Dropout(dropRate))

net:add(nn.Linear(fullwidths[1], fullwidths[2]))
net:add(nn.ReLU())
net:add(nn.Dropout(dropRate))

net:add(nn.Linear(fullwidths[2], fullwidths[3]))
net:add(nn.ReLU())
net:add(nn.Dropout(dropRate))

net:add(nn.Linear(fullwidths[3], 10))
net:add(nn.LogSoftMax())

return net
