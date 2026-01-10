local Matrix = require "matrix"

local mnist = {}

---@alias Sample { image: Matrix, label: number }

-- Reads a 32 bit integer from the header of a file
---@param file file* The file to read the integer from
---@return number
function mnist.read_int32(file)
  local bytes = file:read(4)
  local b1, b2, b3, b4 = bytes:byte(1, 4)

  return b1 * 256 ^ 3 + b2 * 256 ^ 2 + b3 * 256 + b4
end

-- Loads images and labels and creates a list of image matrices and labels
---@param image_path string Path for images file
---@param label_path string Path for labels file
---@param num_to_load number Amount of samples to load
---@return Sample[]
function mnist.load(image_path, label_path, num_to_load)
  local img_file = assert(io.open(image_path, "rb"), "Could not find image")
  local lbl_file = assert(io.open(label_path, "rb"), "Could not find label")

  -- headers
  local magic_img = mnist.read_int32(img_file)
  local n_imgs = mnist.read_int32(img_file)
  local n_rows = mnist.read_int32(img_file)
  local n_cols = mnist.read_int32(img_file)

  local magic_lbl = mnist.read_int32(lbl_file)
  local n_lbls = mnist.read_int32(lbl_file)

  num_to_load = math.min(num_to_load, n_imgs)

  local dataset = {}

  print(string.format("Loading: %d image (%dx%d)...", num_to_load, n_rows, n_cols))

  for i = 1, num_to_load do
    -- read image file and normalize
    local pixels = img_file:read(n_rows * n_cols)
    local matrix = Matrix:new(1, n_rows * n_cols)

    for j = 0, (n_rows * n_cols) - 1 do
      matrix.data[j] = pixels:byte(j + 1) / 255.0
    end

    -- read label
    local label = lbl_file:read(1):byte()
    table.insert(dataset, { image = matrix, label = label })

    if i % 10000 == 0 then print(i .. " ready...") end
  end

  img_file:close()
  lbl_file:close()

  return dataset
end

return mnist
