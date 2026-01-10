local ffi = require "ffi"

---@class Matrix
---@field data table
---@field rows number
---@field cols number
local Matrix = {}

-- Create a matrix with m rows and n cols
---@param m number Matrix rows
---@param n number Matrix columns
---@return Matrix
function Matrix:new(m, n)
  local mx = setmetatable({
    data = ffi.new("double[?]", n * m),
    rows = m,
    cols = n
  }, self)

  return mx
end

-- Creates a clone instance of the matrix
---@return Matrix
function Matrix:clone()
  local clone = Matrix:new(self.rows, self.cols)
  local mx0, mx1 = self.data, clone.data

  for i = 0, self.rows * self.cols - 1 do
    mx1[i] = mx0[i]
  end

  return clone
end

-- Overload indexing for matrix row/col support
---@param idx any Matrix row
function Matrix:__index(idx)
  if type(idx) == "number" and idx >= 0 and idx < self.rows then
    return self.data + idx * self.cols
  end

  return Matrix[idx]
end

-- Addition support for matrices
---@param rhs Matrix Other matrix to add
---@return Matrix
function Matrix:__add(rhs)
  assert(
    self.rows == rhs.rows and self.cols == rhs.cols,
    "Only same-size matrices can be added together"
  )

  local res = Matrix:new(self.rows, self.cols)
  local mx0, mx1, mx_res = self.data, rhs.data, res.data

  for i = 0, self.rows * self.cols - 1 do
    mx_res[i] = mx0[i] + mx1[i]
  end

  return res
end

-- In-place addition support for matrices
---@param rhs Matrix Other matrix to add
---@return Matrix
function Matrix:add(rhs)
  assert(
    self.rows == rhs.rows and self.cols == rhs.cols,
    "Only same-size matrices can be added together"
  )

  local mx0 = self.data
  local mx1 = rhs.data

  for i = 0, self.rows * self.cols - 1 do
    mx0[i] = mx0[i] + mx1[i]
  end

  return self
end

-- Subtraction support for matrices
---@param rhs Matrix Other matrix to subtract
---@return Matrix
function Matrix:__sub(rhs)
  assert(
    self.rows == rhs.rows and self.cols == rhs.cols,
    "Only same-size matrices can be subtracted"
  )

  local res = Matrix:new(self.rows, self.cols)
  local mx0, mx1, mx_res = self.data, rhs.data, res.data

  for i = 0, self.rows * self.cols - 1 do
    mx_res[i] = mx0[i] - mx1[i]
  end

  return res
end

-- In-place subtraction support for matrices
---@param rhs Matrix Other matrix to subtract
---@return Matrix
function Matrix:sub(rhs)
  assert(
    self.rows == rhs.rows and self.cols == rhs.cols,
    "Only same-size matrices can be subtracted"
  )

  local mx0 = self.data
  local mx1 = rhs.data

  for i = 0, self.rows * self.cols - 1 do
    mx0[i] = mx0[i] - mx1[i]
  end

  return self
end

-- Multiplication support for matrices
---@param rhs Matrix|number Matrix to multiply with
---@param target Matrix|nil Matrix to write the result to
---@return Matrix
function Matrix:__mul(rhs, target)
  if target then target:clear() end

  -- multiply by a number
  if type(rhs) == "number" then
    assert(
      not target or target.rows == self.rows and target.cols == self.cols,
      "Invalid target matrix, has to be " .. tostring(self.rows) .. "x" .. tostring(self.cols)
    )

    local res = target or Matrix:new(self.rows, self.cols)
    local mx0, mx_res = self.data, res.data

    for i = 0, self.rows * self.cols - 1 do
      mx_res[i] = mx0[i] * rhs
    end

    return res
  end

  assert(
    self.cols == rhs.rows,
    "Cannot multiply matrices " .. self.rows .. "x" .. self.cols .. " * " .. rhs.rows .. "x" .. rhs.cols
  )
  assert(
    not target or target.rows == self.rows and target.cols == rhs.cols,
    "Invalid target matrix, has to be " .. tostring(self.rows) .. "x" .. tostring(rhs.cols)
  )

  local res = target or Matrix:new(self.rows, rhs.cols)
  local mx0, mx1, mx_res = self.data, rhs.data, res.data

  for i = 0, self.rows - 1 do
    local mx0_col_offset = i * self.cols
    local mx_res_col_offset = i * rhs.cols

    for k = 0, self.cols - 1 do
      local mx0_val = mx0[mx0_col_offset + k]
      local mx1_row_offset = k * rhs.cols

      if mx0_val ~= 0 then
        for j = 0, rhs.cols - 1 do
          local mx_res_idx = mx_res_col_offset + j
          local mx1_idx = mx1_row_offset + j

          mx_res[mx_res_idx] = mx_res[mx_res_idx] + mx0_val * mx1[mx1_idx]
        end
      end
    end
  end

  return res
end

-- In-place multiplication support for matrices (only with numbers)
---@param rhs number Value to multiply with
---@return Matrix
function Matrix:mul(rhs)
  assert(
    type(rhs) == "number",
    "In-place multiplication is only supported with numbers"
  )

  local mx = self.data

  for i = 0, self.rows * self.cols - 1 do
    mx[i] = mx[i] * rhs
  end

  return self
end

-- Component-by-component multiplication
---@param lhs Matrix Matrix A
---@param rhs Matrix Matrix B
---@return Matrix
function Matrix.compMul(lhs, rhs)
  assert(lhs.rows == rhs.rows and lhs.cols == rhs.cols, "Only same size matrices can be multiplied component by component")

  local res = Matrix:new(lhs.rows, lhs.cols)
  local mx0, mx1, mx_res = lhs.data, rhs.data, res.data

  for i = 0, lhs.rows * lhs.cols - 1 do
    mx_res[i] = mx0[i] * mx1[i]
  end

  return res
end

-- Transpose a matrix (creates a new instance)
---@return Matrix
function Matrix:transpose()
  local res = Matrix:new(self.cols, self.rows)
  local mx, mx_t = self.data, res.data

  for r = 0, self.rows - 1 do
    local mx_row_offset = r * self.cols

    for c = 0, self.cols - 1 do
      local val = mx[mx_row_offset + c]
      local idx_t = c * self.rows + r

      mx_t[idx_t] = val
    end
  end

  return res
end

-- Map matrix cells to a new matrix
---@param map_fn fun(val: number, row: number, col: number): number Map function
---@param in_place boolean|nil Enable in place mapping
---@return Matrix
function Matrix:map(map_fn, in_place)
  local res = in_place and self or Matrix:new(self.rows, self.cols)
  local mx, mx_res = self.data, res.data
  local num_cols = self.cols

  for i = 0, self.rows * self.cols - 1 do
    local row = math.floor(i / num_cols)
    local col = i % num_cols

    mx_res[i] = map_fn(mx[i], row, col)
  end

  return res
end

-- Equalitiy check
---@param rhs Matrix Matrix to compare to
---@return boolean
function Matrix:__eq(rhs)
  assert(self.rows == rhs.rows and self.cols == rhs.cols, "Only same size matrices can be compared")

  local mx0, mx1 = self.data, rhs.data

  for i = 0, self.rows * self.cols - 1 do
    if mx0[i] ~= mx1[i] then
      return false
    end
  end

  return true
end

-- Fills the matrix with a given value
---@param val number The value to fill all cells with
---@return Matrix
function Matrix:fill(val)
  local mx = self.data

  for i = 0, self.rows * self.cols - 1 do
    mx[i] = val
  end

  return self
end

-- Clears the matrix, sets all cells to 0
---@return Matrix
function Matrix:clear()
  ffi.fill(self.data, ffi.sizeof("double") * self.rows * self.cols, 0)
  return self
end

-- Stringify matrix
---@return string
function Matrix:__tostring()
  local buffer = {}
  local mx = self.data
  local n = self.rows * self.cols

  local max_decimal_places = 0

  for i = 0, n - 1 do
    local s = tostring(mx[i])
    local dot_pos = s:find("%.")

    if dot_pos then
      local decimal_places = #s - dot_pos

      if decimal_places > max_decimal_places then
        max_decimal_places = decimal_places
      end
    end
  end

  local max_total_width = 0
  local precision_fmt = "%." .. max_decimal_places .. "f"

  for i = 0, n - 1 do
    local s = string.format(precision_fmt, mx[i])
    if #s > max_total_width then max_total_width = #s end
  end

  local final_fmt = "%" .. max_total_width .. "." .. max_decimal_places .. "f"
  local rawlen = self.rows * self.cols

  for i = 0, rawlen - 1 do
    local col = math.fmod(i, self.cols)

    if col > 0 then
      table.insert(buffer, " ")
    end

    table.insert(buffer, string.format(final_fmt, mx[i]))

    if col == self.cols - 1 and i < rawlen - 1 then
      table.insert(buffer, "\n")
    end
  end

  return table.concat(buffer)
end

return Matrix
