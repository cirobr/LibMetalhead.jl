# LibMetalhead.jl

[![Build Status](https://github.com/cirobr/LibMetalhead.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/LibMetalhead.jl/actions/workflows/CI.yml?query=branch%3Amain)

Adapted Metalhead.jl models for specific purposes.

### Networks:
* ResNet34(classes::Int=2), 2-64 classes, softmax output activation
* ResNet50(classes::Int=2), 2-64 classes, softmax output activation
* UResNet34((w,h), ch_in, ch_out), softmax (ch_out > 1) or sigmoid (ch_out == 1) output activation
* UResNet50((w,h), ch_in, ch_out), softmax (ch_out > 1) or sigmoid (ch_out == 1) output activation


### Versions:

### v0.1.0
* First commit
