net = nn.Sequential()
  local convwidths = {3, 64,128,256,256,256}
  local fullwidths = {128, 128}

local function add_conv(i, out_with)
    net:add(nn.SpatialConvolution(convwidths[i], convwidths[i+1], 5, 5, 1, 1, 2, 2))
    net:add(nn.ReLU())
end

local function add_full(in_width, out_with)
    net:add(nn.Linear(in_width, out_with))
    net:add(nn.ReLU())
end

local dropRate = 0.5

view_size = convwidths[#convwidths] * 8 * 8 -- two pooling layers
--view_size = convwidths[3] * 16 * 16 -- one pooling layer

add_conv(1)
add_conv(2)
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Dropout(dropRate))

add_conv(3)
add_conv(4)
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Dropout(dropRate))

net:add(nn.View(view_size))
net:add(nn.Linear(view_size, 512))
net:add(nn.ReLU())
net:add(nn.Dropout(dropRate))

net:add(nn.Linear(512, 512))
net:add(nn.ReLU())
net:add(nn.Dropout(dropRate))

net:add(nn.Linear(512, 10))
net:add(nn.LogSoftMax())

return net
