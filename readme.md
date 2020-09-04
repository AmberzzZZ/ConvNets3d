1. c3d
    8 conv layers + 5 pooling layers + 2 fc layers + softmax
    homogeneous：3x3x3 s1 conv throughout
    pool1：1x2x2 kernel size & stride，rest 2x2x2
    fc dims：4096
    C3D video descriptor：fc6 activations + L2-norm
    dropout: 论文里面没提，但是下面那个github的实现里面加了


2. 3d-resnet
    r18和r34，后续文章探讨了更深的resnet，有人说效果奇差：https://github.com/kenshohara/3D-ResNets
      local cfg = {
        [10]  = {{1, 1, 1, 1}, 512, basicblock},
        [18]  = {{2, 2, 2, 2}, 512, basicblock},
        [34]  = {{3, 4, 6, 3}, 512, basicblock},
        [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
        [101] = {{3, 4, 23, 3}, 2048, bottleneck},
        [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }
    identity shortcuts：use zero-padding
    7x7x7 stem + 3x3x3 conv blocks + GAP + 400d-fc + softmax


3. pseudo-resnet
    伪3d：通过1x3x3和3x1x1的S和T来实现
    bottleneck 循环ABC bottleneck blocks
    model size：r50: 92M，p3d-r50: 98M
    input: 16x160x160
    with an extra dropout layer with 0.9 dropout rate



