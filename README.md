# LibMetalhead.jl

[![Build Status](https://github.com/cirobr/LibMetalhead.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/LibMetalhead.jl/actions/workflows/CI.yml?query=branch%3Amain)

Adapted Metalhead.jl models.

### Networks:
* metalhead_unet(type, framesize, ch_in, ch_out)

where: \
type ∈ (18, 34, 50, 101, 152) \\
framesize = (w,h) \\
ch_in = 3 (for RGB) \\
ch_out = 1 => sigmoid output activation \\
ch_out > 1 => softmax output activation \\