############################################################
# Code modified from https://github.com/sniklaus/pytorch-pwc
############################################################

import torch
from correlation import correlation


class PWC_Net(torch.nn.Module):
	def __init__(self):
		super(PWC_Net, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tensorInput):
				tensorOne = self.moduleOne(tensorInput)
				tensorTwo = self.moduleTwo(tensorOne)
				tensorThr = self.moduleThr(tensorTwo)
				tensorFou = self.moduleFou(tensorThr)
				tensorFiv = self.moduleFiv(tensorFou)
				tensorSix = self.moduleSix(tensorFiv)

				return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
			# end
		# end

		class Backward(torch.nn.Module):
			def __init__(self):
				super(Backward, self).__init__()
			# end

			def forward(self, tensorInput, tensorFlow):
				if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
					self.tensorPartial = torch.FloatTensor().resize_(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)).fill_(1.0).cuda()
				# end

				if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
					torchHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3))
					torchVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3))

					self.tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda()
				# end

				tensorInput = torch.cat([ tensorInput, self.tensorPartial ], 1)
				tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

				tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

				tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

				return tensorOutput[:, :-1, :, :] * tensorMask
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)

				if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]
				if intLevel < 6: self.moduleBackward = Backward()

				self.moduleCorrelation = correlation.ModuleCorrelation()
				self.moduleCorreleaky = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end

			def forward(self, tensorFirst, tensorSecond, objectPrevious):
				tensorFlow = None
				tensorFeat = None

				if objectPrevious is None:
					tensorFlow = None
					tensorFeat = None

					tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, tensorSecond))

					tensorFeat = torch.cat([ tensorVolume ], 1)

				elif objectPrevious is not None:
					tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
					tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

					tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, self.moduleBackward(tensorSecond, tensorFlow * self.dblBackward)))

					tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat ], 1)

				# end

				tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFiv(tensorFeat), tensorFeat ], 1)

				tensorFlow = self.moduleSix(tensorFeat)

				return {
					'tensorFlow': tensorFlow,
					'tensorFeat': tensorFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.moduleMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1,  dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,  dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,  dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8,  dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16,  dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1,  dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1,  dilation=1)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleMain(tensorInput)
			# end
		# end

		self.moduleExtractor = Extractor()

		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)

		self.moduleRefiner = Refiner()
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst = self.moduleExtractor(tensorFirst)
		tensorSecond = self.moduleExtractor(tensorSecond)

		objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
		objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
		objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
		objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
		objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

		return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])
	# end
# end