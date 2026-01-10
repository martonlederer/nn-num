local Matrix = require "matrix"

---@alias ActivationFunction fun(z: number): number
---@alias Activation { fn: ActivationFunction, d: ActivationFunction }

---@class Layer
---@field weights Matrix
---@field bias Matrix
---@field activation Activation
---@field z Matrix
---@field last_x Matrix|nil
local Layer = {
  activation = {
    derivative = {}
  }
}

-- Create a layer
---@param neurons number Neurons count
---@param input_size number Size of the input for each neuron
---@param activation Activation Activation function
---@return Layer
function Layer:new(neurons, input_size, activation)
  local layer = setmetatable({
    weights = Matrix:new(input_size, neurons),
    bias = Matrix:new(1, neurons),
    activation = activation,
    z = Matrix:new(1, neurons),
    last_x = nil
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
---@param training? boolean Is the model trading
---@return Matrix
function Layer:forward(x, training)
  if training then self.last_x = x:clone() end

  x:__mul(self.weights, self.z)
  self.z:add(self.bias)

  return self.z:map(self.activation.fn, true)
end

-- Adjusts the layer's weights according to the received gradient and returns
-- the gradient to pass backward
---@param gradient Matrix Gradient matrix
---@param learning_rate number The learning rate for modifying weights
---@return Matrix
function Layer:backward(gradient, learning_rate)
  assert(self.last_x, "No saved last input for training")

  local delta = gradient:map(function(val, _, col)
    return val * self.activation.d(self.z.data[col])
  end)
  local next_gradient = delta * self.weights:transpose()

  delta:mul(learning_rate)
  self.weights:sub(self.last_x:transpose() * delta)
  self.bias:sub(delta)

  return next_gradient
end

-- Sigmoid activation function (phi) for a neuron
Layer.activation.sigmoid = {
  fn = function (z)
    return 1 / (1 + math.exp(-1 * z))
  end,
  d = function (a)
    return a * (1 - a)
  end
}

return Layer
