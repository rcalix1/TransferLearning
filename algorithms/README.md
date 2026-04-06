## Transfer Learning in PyTorch

* link

## Get string from pred label (e.g. 601)


```

import json
import urllib.request

# load mapping once
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# example
idx = 601
print(classes[idx])



```
