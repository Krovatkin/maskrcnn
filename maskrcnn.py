import torch
import torchvision
import torch._lazy
import torch._lazy.metrics
import torch._lazy.ts_backend
from torch.utils._pytree import tree_map

torch._lazy.ts_backend.init()

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
device='lazy'
model.to(device)
model.train()

lr = 0.02
momentum = 0.9
weight_decay = 1e-4
niter = 10
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

maskrcnn_inputs = torch.load('maskrcnn_inputs.pt')


print(len(maskrcnn_inputs))

for i, (images, targets) in zip(range(niter), maskrcnn_inputs):
    print(f"running {i}")
    new_targets = tree_map(lambda x: x.to(device), targets)
    new_images = tree_map(lambda x: x.to(device), images)

    loss_dict = model(new_images, new_targets)    
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    torch._lazy.mark_step()
    print(torch._lazy.metrics.counter_names())
    torch._lazy.wait_device_ops()