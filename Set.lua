Set = {}
Set.proto = {float= function(self)
   self.data = self.data:float()
   self.label = self.label:float()
 end,
 cuda = function(self)
   self.data = self.data:cuda()
   self.label = self.label:cuda()
 end,
  size = function(self)
  return self.data:size(1)
end}
Set.mt = {__index = function(tab,key) return Set.proto[key] end}
function Set.new(o)
  setmetatable(o, Set.mt)
  return o
end
