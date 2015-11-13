local net = nn.Sequential()
  local convwidths = {3, 64,128,256,256,256,256,256,256}
  local fullwidths = {1024, 256, 128}

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

local function add_full(in_width, out_with)
    net:add(nn.Linear(in_width, out_with))
    net:add(nn.ReLU())
end

local dropRate = 0.5

local view_size = convwidths[#convwidths] * 8 * 8 -- two pooling layers
--view_size = convwidths[3] * 16 * 16 -- one pooling layer

add_conv(1)
add_conv(2)
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Dropout(dropRate))

add_conv(3)
add_conv(4)
--net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Dropout(dropRate))

add_conv(5)
add_conv(6)
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Dropout(dropRate))

add_conv(7)
add_conv(8)
net:add(nn.Dropout(dropRate))

net:add(nn.View(view_size))

net:add(nn.Linear(view_size, fullwidths[1]))
net:add(nn.ReLU())
net:add(nn.Dropout(dropRate))

net:add(nn.Linear(fullwidths[1], fullwidths[2]))
net:add(nn.ReLU())
net:add(nn.Dropout(dropRate))

net:add(nn.Linear(fullwidths[2], 10))
net:add(nn.LogSoftMax())

return net
