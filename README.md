
This repo is used for research into MaskRCNN for the purpose of implementing dynamic shapes.

It includes the following:

* `maskrcnn.json` -- a chrome trace of one full train iteration
* `maskrcnn.py` -- a script to run maskrcnn with real inputs from the COCO dataset.
* `maskrcnn_inputs.pt` -- the tensor inputs used by `maskrcnn.py` and `maskrcnn_profiler.py`
* `maskrcnn_profiler.py` -- an example on running MaskRCNN with PyTorch Profiler
* `missing_jit_shape_funcs.py` -- a script to generate missing JIT SSA shape functions
* `missing_ub_funcs.py` -- a script to generate missing ops in LTC for MaskRCNN
* `torch_mlir_shape_funcs.py` a script to generate the list of JIT shape functions we already implement
