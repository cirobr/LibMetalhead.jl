m = ResNet50(3)
@test size(m(x)) == (3,1)

m = ResNet34(3)
@test size(m(x)) == (3,1)

