using LibMetalhead
using Test
using Random

Random.seed!(1234)
x = randn(Float32, (256,256,3,1))

@testset "LibMetalhead.jl" begin
    include("./resnets-tests.jl")
    include("./unets-tests.jl")
end
