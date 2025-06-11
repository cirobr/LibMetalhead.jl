function ResNet34(classes::Int=2)
    @assert classes ∈ 2:64 || error("classes must be in 2:64")
    kf = 1e-2

    return Chain(m.backbone( m.ResNet(34; pretrain=true) ),
            AdaptiveMeanPool((1,1)),
            Flux.flatten,
            Dense(512, 64; init=kaiming_normal(gain=kf*√512)), BatchNorm(64, leakyrelu),
            Dropout(0.25),
            Dense(64, classes; init=kaiming_normal(gain=kf*√64)), softmax   # columns are independent vectors
    )
end


function ResNet50(classes::Int=2)
    @assert classes ∈ 2:64 || error("classes must be in 2:64")
    kf = 1e-2

    return Chain(m.backbone( m.ResNet(50; pretrain=true) ),
            AdaptiveMeanPool((1,1)),
            Flux.flatten,
            Dense(2048, 512; init=kaiming_normal(gain=kf*√2048)), BatchNorm(512, leakyrelu),
            # Dropout(0.1),     # from htresnet50.jl
            Dense(512, 64; init=kaiming_normal(gain=kf*√512)),    BatchNorm(64, leakyrelu),
            # Dropout(0.0),   # from htresnet50.jl
            Dense(64, classes; init=kaiming_normal(gain=kf*√64)), softmax   # columns are independent vectors
    )
end
