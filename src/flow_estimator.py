############################################################
# Code modified from https://github.com/sniklaus/pytorch-pwc
############################################################

import math
import torch

from models.pwc_net import PWC_Net


class FlowEstimator(object):

    def __init__(self):
        self.model_pwc = PWC_Net().type(torch.cuda.FloatTensor)
        self.model_pwc.load_state_dict(torch.load('pretrained_models/pwc_net.tar'))


    def estimate_flow_pair(self, tensorInputFirst, tensorInputSecond):
        ### tensor format
        # C x H x W
        # BGR
        # 0-1
        # FloatTensor.cuda
        ###

        moduleNetwork = self.model_pwc
        tensorOutput = torch.FloatTensor().cuda()

        assert(tensorInputFirst.size(1) == tensorInputSecond.size(1))
        assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))

        intWidth = tensorInputFirst.size(2)
        intHeight = tensorInputFirst.size(1)

    #     assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    #     assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

        if True:
            tensorPreprocessedFirst = tensorInputFirst.view(1, 3, intHeight, intWidth)
            tensorPreprocessedSecond = tensorInputSecond.view(1, 3, intHeight, intWidth)

            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

            tensorPreprocessedFirst = torch.nn.functional.upsample(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            tensorPreprocessedSecond = torch.nn.functional.upsample(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

            tensorFlow = 20.0 * torch.nn.functional.upsample(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

            tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            tensorOutput.resize_(2, intHeight, intWidth).copy_(tensorFlow[0, :, :, :])
        # end

        return tensorOutput # C x H x W


    def estimate_flow_batch(self, out_tensor):
        N, C, H, W = out_tensor.size()
        flow_tensor = torch.FloatTensor(N-1, 2, H, W)
        for i in range(N-1):
            first = out_tensor[i]
            second = out_tensor[i+1]
            flow_tensor[i] = estimate_flow_pair(first, second)
        return flow_tensor # N-1 x 2 x H x W
