local Layer = require "layer"
local NeuralNetwork = require "neuralnetwork"
local mnist = require "mnist"

local model = NeuralNetwork:new()
model:addLayer(Layer:new(128, 784, Layer.activation.sigmoid))
model:addLayer(Layer:new(10, 128, Layer.activation.sigmoid))

local dataset = mnist.load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 10)

model:fit(dataset, 20, 0.1, function (epoch, total, loss)
  print("Epoch " .. epoch .. "/" .. total .. " (loss: " .. loss .. ")")
end)

local function get_predicted_num(predictions)
  local predictions_raw = predictions.data
  local output, p = 0, predictions_raw[0]

  for i = 1, predictions.rows * predictions.cols - 1 do
    if predictions_raw[i] > p then
      output = i
    end
  end

  return output, p
end

for i = 1, math.min(20, #dataset) do
  local sample = dataset[i]
  local prediction, confidence = get_predicted_num(model:predict(sample.image))

  print("Tip for " .. sample.label .. " is " .. prediction .. " (confidence: " .. confidence .. ")")
end
