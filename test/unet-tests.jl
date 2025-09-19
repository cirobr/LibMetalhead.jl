m = metalhead_unet(18, (256,256), 3, 2)
@test size(m(x)) == (256,256,2,1)
