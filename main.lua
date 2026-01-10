local Layer = require "layer"
local NeuralNetwork = require "neuralnetwork"
local mnist = require "mnist"

local model = NeuralNetwork:new()
model:addLayer(Layer:new(128, 784, Layer.activation.sigmoid))
model:addLayer(Layer:new(10, 128, Layer.activation.sigmoid))

if not model:load("./model") then
  local dataset = mnist.load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 10000)

  model:fit(dataset, 20, 0.1, function (epoch, total, loss)
    print("Epoch " .. epoch .. "/" .. total .. " (loss: " .. loss .. ")")
  end)
else
  print("Loaded existing model")
end

local function get_predicted_num(predictions)
  local predictions_raw = predictions.data
  local output, p = 0, predictions_raw[0]

  for i = 1, predictions.rows * predictions.cols - 1 do
    if predictions_raw[i] > p then
      output = i
      p = predictions_raw[i]
    end
  end

  return output, p
end

local test_amount = 30
local validation = mnist.load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", test_amount)

print("Testing")
local correct = 0

for i = 1, test_amount do
  local sample = validation[i]
  local prediction, confidence = get_predicted_num(model:predict(sample.image))
  local is_correct = sample.label == prediction

  if is_correct then correct = correct + 1 end
  print(i .. "/" .. test_amount .. " " .. (is_correct and "[CORRECT]" or "[INCORRECT]") .. " Tip for " .. sample.label .. " is " .. prediction .. " (confidence: " .. confidence .. ")")
end

print("Accuracy " .. (correct / test_amount))

model:save("./model")
