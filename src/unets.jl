# model = UResNet34((512,512), 3, C)
function UResNet34((w,h), ch_in::Int, ch_out::Int)
    # [18, 34, 50, 101, 152]
    backbone = m.backbone( m.ResNet(34; pretrain=true) )
    activation = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    return Chain(m.UNet((w,h), ch_in, ch_out, backbone), activation)
end


# model = UResNet50((512,512), 3, C)
function UResNet50((w,h), ch_in::Int, ch_out::Int)
    # [18, 34, 50, 101, 152]
    backbone = m.backbone( m.ResNet(50; pretrain=true) )
    activation = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    return Chain(m.UNet((w,h), ch_in, ch_out, backbone), activation)
end
