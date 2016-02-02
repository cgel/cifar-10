Set = {}

Set.proto = {float= function(self)
   self.data = self.data:float()
   self.label = self.label:float()
   return self
 end,
 cuda = function(self)
   self.data = self.data:cuda()
   self.label = self.label:cuda()
   return self
 end,
  size = function(self)
  return self.data:size(1)
 end,
 normalize = function(self)
  for ch = 1, 3 do
    mean = self.data[{{ch}}]:mean()
    std = self.data[{{ch}}]:std()
    self.data[{{ch}}]:add(-mean)
    self.data[{{ch}}]:div(std)
  end
  return self
end
}

Set.mt = {__index = function(tab,key) return Set.proto[key] end}

function Set.new(o)
  setmetatable(o, Set.mt)
  return o
end
