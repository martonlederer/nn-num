local Matrix = require "matrix"
local Layer = require "layer"

-- 1. Setup: Hozzunk létre egy réteget (2 bemenet -> 3 neuron)
-- Aktivációnak egyelőre használjunk egy "identitás" függvényt (visszaadja önmagát),
-- hogy könnyebb legyen ellenőrizni a matekot.
local identity = function(x) return x end
local my_layer = Layer:new(2, 3, identity)

-- 2. Manuális súlyok beállítása (opcionális, a teszt kedvéért)
-- Weights: 2x3 mátrix
-- [ 1  2  3 ]
-- [ 4  5  6 ]
my_layer.weights[0][0] = 1; my_layer.weights[0][1] = 2; my_layer.weights[0][2] = 3
my_layer.weights[1][0] = 4; my_layer.weights[1][1] = 5; my_layer.weights[1][2] = 6

-- Bias: 1x3 vektor
-- [ 10 10 10 ]
my_layer.bias:fill(10)

-- 3. Bemenet: 1x2 vektor
-- [ 1  1 ]
local input = Matrix:new(1, 3)
input[0][0] = 1
input[0][1] = 1
input[0][2] = 1

-- 4. Forward Pass futtatása
local result = my_layer:forward(input)

-- 5. Eredmény kiírása
print("Bemenet:\n" .. tostring(input))
print("\nSúlyok:\n" .. tostring(my_layer.weights))
print("\nBias:\n" .. tostring(my_layer.bias))
print("\nEredmény (Várt: [15, 17, 19]):\n" .. tostring(result))
