import torch_mlir_shape_funcs

# Note, this list was obtained by
# `running python maskrcnn.py`
maskrcnn_ops = ['CachedCompile',
    'CreateLtcTensor',
    'DestroyLtcTensor',
    'DeviceDataCacheMiss',
    'MarkStep',
    'SyncTensorsToData',
    'UncachedCompile',
    'aten::_index_put_impl_',
    'aten::_local_scalar_dense',
    'aten::_unique2',
    'aten::index.Tensor',
    'aten::nonzero',
    'aten::randperm.generator_out',
    'lazy::_copy_from',
    'lazy::_copy_from_and_resize',
    'lazy::_log_softmax',
    'lazy::_log_softmax_backward_data',
    'lazy::_to_copy',
    'lazy::add',
    'lazy::addmm',
    'lazy::any',
    'lazy::arange_out',
    'lazy::bitwise_and',
    'lazy::bitwise_or',
    'lazy::cat',
    'lazy::clamp',
    'lazy::clamp_min',
    'lazy::convolution',
    'lazy::convolution_backward',
    'lazy::div',
    'lazy::eq',
    'lazy::exp',
    'lazy::expand',
    'lazy::fill_',
    'lazy::floor',
    'lazy::ge',
    'lazy::gt',
    'lazy::le',
    'lazy::log',
    'lazy::log2',
    'lazy::lt',
    'lazy::max',
    'lazy::max_pool2d_with_indices',
    'lazy::max_pool2d_with_indices_backward',
    'lazy::maximum',
    'lazy::mean',
    'lazy::minimum',
    'lazy::mm',
    'lazy::mul',
    'lazy::narrow',
    'lazy::neg',
    'lazy::nll_loss_backward',
    'lazy::nll_loss_forward',
    'lazy::permute',
    'lazy::relu',
    'lazy::relu_',
    'lazy::rsqrt',
    'lazy::select',
    'lazy::sigmoid',
    'lazy::slice',
    'lazy::smooth_l1_loss',
    'lazy::smooth_l1_loss_backward',
    'lazy::sort',
    'lazy::sqrt',
    'lazy::stack',
    'lazy::sub',
    'lazy::sum',
    'lazy::t',
    'lazy::threshold_backward',
    'lazy::topk',
    'lazy::unsqueeze',
    'lazy::upsample_bilinear2d',
    'lazy::upsample_nearest2d',
    'lazy::upsample_nearest2d_backward',
    'lazy::view',
    'lazy::zero_',
    'torchvision::_roi_align_backward',
    'torchvision::nms',
    'torchvision::roi_align'
]



def remove_suffix(x):
  return x.split(".")[0]

jit_shape_funcs = set([remove_suffix(x.replace("〇", ".")) for x in  torch_mlir_shape_funcs.get_shape_functions()])

not_implemented = []
for o in maskrcnn_ops:
  co = o.replace("lazy::", "").replace("aten::", "")
  if not co in jit_shape_funcs:
    not_implemented.append(co)

print(not_implemented)

# This is the list of ops in MaskRCNN that do NOT have corresponding JIT SSA functions
"""
_index_put_impl_
_local_scalar_dense
_unique2
index.Tensor
nonzero
randperm.generator_out
_copy_from
_copy_from_and_resize
arange_out
bitwise_or
clamp_min
convolution_backward
fill_
narrow
relu_
smooth_l1_loss
smooth_l1_loss_backward
sort
stack
upsample_bilinear2d
upsample_nearest2d
upsample_nearest2d_backward
zero_
torchvision::_roi_align_backward
torchvision::nms
torchvision::roi_align
"""