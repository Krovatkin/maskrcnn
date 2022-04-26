import torch
import torchvision
import torch._lazy
import torch._lazy.metrics
import torch._lazy.ts_backend
from torch.utils._pytree import tree_map
import torch.autograd.profiler as profiler

torch._lazy.ts_backend.init()

"""
This script is using pretrained maskrcnn_resnet50_fpn and
10 random images from the COCO dataset
One could use this script to run maskrcnn against the lazy device
and collect statistics on supported and unsupported ops by Lazy
"""

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
device='lazy'
model.to(device)
model.train()

lr = 0.02
momentum = 0.9
weight_decay = 1e-4
# we only need one iteration for profiling
# note this isn't performance profiling, so there's no warmup
niter = 1
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

maskrcnn_inputs = torch.load('maskrcnn_inputs.pt')

with profiler.profile(record_shapes=True) as prof: # , use_cuda=True
    for i, (images, targets) in zip(range(niter), maskrcnn_inputs):
        print(f"running {i}")
        new_targets = tree_map(lambda x: x.to(device), targets)
        new_images = tree_map(lambda x: x.to(device), images)

        with profiler.record_function("forward"):
            loss_dict = model(new_images, new_targets)    
            losses = sum(loss for loss in loss_dict.values())
        with profiler.record_function("zero_grad"):
            optimizer.zero_grad()
        with profiler.record_function("backward"):
            losses.backward()
        with profiler.record_function("optimizer.step"):
            optimizer.step()
        torch._lazy.mark_step()
        print(torch._lazy.metrics.counter_names())
        torch._lazy.wait_device_ops()


# open this in chrome://tracing
prof.export_chrome_trace("maskrcnn.json")
print("done!")