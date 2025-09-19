function UResNet34((w,h)::Tuple{Int,Int}, ch_in::Int, ch_out::Int)
    # [18, 34, 50, 101, 152]
    encoder_backbone = backbone( ResNet(34) )
    activation = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)
    return Chain(UNet((w,h), ch_in, ch_out, encoder_backbone), activation)
end


function UResNet50((w,h)::Tuple{Int,Int}, ch_in::Int, ch_out::Int)
    # [18, 34, 50, 101, 152]
    encoder_backbone = backbone( ResNet(50) )
    activation = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)
    return Chain(UNet((w,h), ch_in, ch_out, encoder_backbone), activation)
end
