m = lm.ResNet50(3)
@test size(m(x)) == (3,1)

m = lm.ResNet34(3)
@test size(m(x)) == (3,1)

