## Introduction examples to transformers module

* GPTs, BERTs, etc. 

## Check if cuda device

import torch

torch.cuda.is_available()

True

torch.cuda.device_count()

1

torch.cuda.current_device()

0

torch.cuda.device(0)

<torch.cuda.device at 0x7efce0b03be0>

torch.cuda.get_device_name(0)

'GeForce GTX 950M'

