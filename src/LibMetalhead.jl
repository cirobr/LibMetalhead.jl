module LibMetalhead


export UResNet34, UResNet50

import Metalhead, Flux
import Metalhead: UNet, ResNet, backbone
import Flux: Chain, σ, sigmoid, softmax

include("unets.jl")


end   # module
