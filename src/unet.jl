function metalhead_unet(type::Int, framesize::Tuple{Int, Int}, ch_in::Int, ch_out::Int)
    @assert type ∈ (18, 34, 50, 101, 152) || error("Unsupported ResNet type: $type")
    bb = backbone(ResNet(type; pretrain=false))
    activation = ch_out == 1 ? x -> σ(x) : x -> softmax(x; dims=3)

    return Chain(UNet(framesize, ch_in, ch_out, bb), activation)
end
