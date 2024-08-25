m = UResNet50((256,256), 3, 2)
@test size(m(x)) == (256,256,2,1)

m = UResNet34((256,256), 3, 2)
@test size(m(x)) == (256,256,2,1)
