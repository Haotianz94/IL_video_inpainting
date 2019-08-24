############################################################
# Code modified from https://github.com/sniklaus/pytorch-pwc
############################################################

import cupy
import torch

kernel_Correlation_rearrange = '''
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	  if (intIndex >= n) {
	    return;
	  }

	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;

	  float dblValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

	  __syncthreads();

	  int intPaddedY = (intIndex / SIZE_3(input)) + 4;
	  int intPaddedX = (intIndex % SIZE_3(input)) + 4;
	  int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;

	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = dblValue;
	}
'''

kernel_Correlation_updateOutput = '''
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
	  extern __shared__ char patch_data_char[];
	  
	  float *patch_data = (float *)patch_data_char;
	  
	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = blockIdx.x + 4;
	  int y1 = blockIdx.y + 4;
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;
	  
	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = ((j * 1) + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }
	  
	  __syncthreads();
	  
	  __shared__ float sum[32];
	  
	  // Compute correlation
	  for(int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;
	  
	    int s2o = (top_channel % 9) - 4;
	    int s2p = (top_channel / 9) - 4;
	    
	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = ((j * 1) + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;
	          
	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;
	          
	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }
	    
	    __syncthreads();
	    
	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  } 
	}
'''

def cupy_kernel(strFunction, objectVariables):
	strKernel = globals()[strFunction]

	for strVariable in objectVariables:
		strKernel = strKernel.replace('SIZE_0(' + strVariable + ')', str(objectVariables[strVariable].size(0)))
		strKernel = strKernel.replace('SIZE_1(' + strVariable + ')', str(objectVariables[strVariable].size(1)))
		strKernel = strKernel.replace('SIZE_2(' + strVariable + ')', str(objectVariables[strVariable].size(2)))
		strKernel = strKernel.replace('SIZE_3(' + strVariable + ')', str(objectVariables[strVariable].size(3)))
	# end

	return strKernel
# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class FunctionCorrelation(torch.autograd.Function):
	@staticmethod
	def forward(ctx, first, second):
		ctx.save_for_backward(first, second)

		assert(first.is_contiguous() == True)
		assert(second.is_contiguous() == True)

		rbot0 = first.new(first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)).zero_()
		rbot1 = first.new(first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)).zero_()

		output = first.new(first.size(0), 81, first.size(2), first.size(3)).zero_()

		if first.is_cuda == True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream
			# end

			n = first.size(2) * first.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'input': first,
				'output': rbot0
			}))(
				grid=tuple([ int((n + 16 - 1) / 16), first.size(1), first.size(0) ]),
				block=tuple([ 16, 1, 1 ]),
				args=[ n, first.data_ptr(), rbot0.data_ptr() ],
				stream=Stream
			)

			n = second.size(2) * second.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'input': second,
				'output': rbot1
			}))(
				grid=tuple([ int((n + 16 - 1) / 16), second.size(1), second.size(0) ]),
				block=tuple([ 16, 1, 1 ]),
				args=[ n, second.data_ptr(), rbot1.data_ptr() ],
				stream=Stream
			)

			n = output.size(1) * output.size(2) * output.size(3)
			cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
				'rbot0': rbot0,
				'rbot1': rbot1,
				'top': output
			}))(
				grid=tuple([ first.size(3), first.size(2), first.size(0) ]),
				block=tuple([ 32, 1, 1 ]),
				shared_mem=first.size(1) * 4,
				args=[ n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr() ],
				stream=Stream
			)

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	@staticmethod
	def backward(ctx, gradOutput):
		first, second = ctx.saved_tensors

		assert(gradOutput.is_contiguous() == True)

		gradFirst = first.new(first.size()).zero_() if ctx.needs_input_grad[0] == True else None
		gradSecond = first.new(first.size()).zero_() if ctx.needs_input_grad[1] == True else None

		if first.is_cuda == True:
			raise NotImplementedError()

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradFirst, gradSecond
	# end
# end

class ModuleCorrelation(torch.nn.Module):
	def __init__(self):
		super(ModuleCorrelation, self).__init__()
	# end

	def forward(self, tensorFirst, tensorSecond):
		correlation = FunctionCorrelation.apply
		return correlation(tensorFirst, tensorSecond)
	# end
# end