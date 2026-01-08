local Matrix = require "matrix"

local mx0 = Matrix:new(2, 3)
mx0[0][0] = 1
mx0[0][1] = 2
mx0[0][2] = 3
mx0[1][0] = 4
mx0[1][1] = 5
mx0[1][2] = 6

local mx1 = Matrix:new(3, 2)
mx1[0][0] = 1
mx1[0][1] = 0
mx1[1][0] = 7
mx1[1][1] = 2
mx1[2][0] = 3
mx1[2][1] = 8

print(tostring(mx0))
print("*")
print(tostring(mx1))

local mx_mul = mx0 * mx1

print("=")
print(tostring(mx_mul))

print("\n")
print(tostring(mx_mul:transpose()))

print("\n")
print(tostring(mx_mul * -1))
