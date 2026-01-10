local Matrix = require "matrix"

---@class NeuralNetwork
---@field layers Layer[]
local NeuralNetwork = {}

-- Create an empty Neural Network
---@return NeuralNetwork
function NeuralNetwork:new()
	local network = setmetatable({
	  layers = {}
	}, self)
	self.__index = self

	return network
end

-- Adds a layer to the network
---@param layer Layer Layer to add
---@return NeuralNetwork
function NeuralNetwork:addLayer(layer)
  table.insert(self.layers, layer)

  return self
end

-- Runs the model and gives a prediction
---@param input Matrix The input to predict for
---@param training? boolean Is the model training
---@return Matrix
function NeuralNetwork:predict(input, training)
  local predictions = input

  -- forward pass
  for _, layer in ipairs(self.layers) do
    predictions = layer:forward(predictions, training)
  end

  return predictions
end

-- Adjusts the model based on the prediction output and the expected prediction, then returns the loss
---@param y Matrix Output from model prediction
---@param d Matrix Expected output
---@param learning_rate number Learning rate for the training
---@return number
function NeuralNetwork:adjust(y, d, learning_rate)
  assert(
    y.rows == 1 and d.rows == 1 and y.cols == d.cols,
    "y and d matrices have to be the same size"
  )

  ---@type Matrix
  local gradient = y - d
  local loss = 0

  for i = 0, gradient.cols - 1 do
    local diff = gradient.data[i]
    loss = loss + (diff * diff)
  end
  loss = loss * 0.5

  for i = 0, #self.layers - 1 do
    gradient = self.layers[#self.layers - i]:backward(
      gradient,
      learning_rate
    )
  end

  return loss
end

-- Train the model with a dataset
---@param dataset Sample[] The dataset to train on
---@param epochs number Amount of iterations over the dataset
---@param learning_rate number Learning rate for the training
---@param log? fun(epoch: number, total: number, loss: number) Optional logging function
function NeuralNetwork:fit(dataset, epochs, learning_rate, log)
  for i = 1, epochs do
    local total_loss = 0

    for _, sample in ipairs(dataset) do
      -- forward pass
      local y = self:predict(sample.image, true)

      -- expected result
      local d = Matrix:new(y.rows, y.cols):fill(0)
      d.data[sample.label] = 1.0

      local loss = self:adjust(y, d, learning_rate)
      total_loss = total_loss + loss
    end

    if log then log(i, epochs, total_loss) end
  end
end

-- Saves weights and biases for each layer
---@param location string Location to save to
---@return boolean
function NeuralNetwork:save(location)
  local file = io.open(location, "w")
  if not file then return false end

  for _, layer in ipairs(self.layers) do
    local mx_w = layer.weights.data
    local wcount = layer.weights.cols * layer.weights.rows

    for i = 0, wcount - 1 do
      file:write(tostring(mx_w[i]), i < wcount - 1 and "," or "")
    end

    file:write("\t")

    local bias_w = layer.bias.data

    for i = 0, layer.bias.cols - 1 do
      file:write(tostring(bias_w[i]), i < layer.bias.cols - 1 and "," or "")
    end

    file:write("\n")
  end

  file:close()
  return true
end

-- Loads weights and biases
---@param location string
---@return boolean
function NeuralNetwork:load(location)
  local file = io.open(location, "r")
  if not file then return false end

  local layer_idx = 1
  for line in file:lines() do
    local current_layer = self.layers[layer_idx]
    if not current_layer then break end

    local weights_str, bias_str = line:match("([^\t]+)\t([^\t]+)")

    if weights_str and bias_str then
      local w_idx = 0
      for val in weights_str:gmatch("[^,]+") do
        current_layer.weights.data[w_idx] = tonumber(val)
        w_idx = w_idx + 1
      end

      local b_idx = 0
      for val in bias_str:gmatch("[^,]+") do
        current_layer.bias.data[b_idx] = tonumber(val)
        b_idx = b_idx + 1
      end
    end

    layer_idx = layer_idx + 1
  end

  file:close()
  return true
end

return NeuralNetwork
