local Matrix = require "matrix"

local Layer = {
  activation = {}
}

-- Create a layer
---@param neurons number Neurons count
---@param input_size number Size of the input for each neuron
---@param activation fun(x: number): number Activation function
function Layer:new(neurons, input_size, activation)
  local layer = setmetatable({
    weights = Matrix:new(input_size, neurons),
    bias = Matrix:new(1, neurons),
    activation = activation,
    z = Matrix:new(1, neurons)
  }, self)
  self.__index = self

  -- randomize
  local scale = input_size and (1 / math.sqrt(input_size)) or 1
  local mx_w, mx_b = layer.weights.data, layer.bias.data

  for i = 0, input_size * neurons - 1 do
    mx_w[i] = (math.random() * 2 - 1) * scale
  end
  for i = 0, neurons - 1 do
    mx_b[i] = (math.random() * 2 - 1) * scale
  end

  return layer
end

-- Computes the layer’s forward pass output
---@param x Matrix Inputs
---@return number
function Layer:forward(x)
  x:__mul(self.weights, self.z)
  self.z:add(self.bias)

  return self.z:map(self.activation, true)
end

-- Sigmoid activation function (phi) for a neuron
---@param z number Weighted sum of inputs (plus bias)
---@return number
function Layer.activation.sigmoid(z)
  return 1 / (1 + math.exp(-1 * z))
end

-- ReLU activation function (phi) for a neuron
---@param z number Weighted sum of inputs (plus bias)
---@return number
function Layer.activation.relu(z)
  return math.max(0, z)
end

return Layer
