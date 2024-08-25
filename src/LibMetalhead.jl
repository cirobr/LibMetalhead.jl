module LibMetalhead



export ResNet50, ResNet34
export UResNet50, UResNet34

import Metalhead, Flux; m=Metalhead
import Flux: Chain, Dense, Conv, AdaptiveMeanPool, BatchNorm, Dropout,
             kaiming_normal,
             σ, sigmoid, softmax, relu, leakyrelu

include("resnets.jl")
include("unets.jl")



end   # module
