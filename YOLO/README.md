## Object Detection 

* YOLO

## YOLO

```


## !pip install ultralytics

# usage
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # nano model (fast)



img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list


results = model(img)

results[0].show() 


results


```




## Examples

* Watermelon Analyzer Object Detection (https://github.com/rcalix1/WatermelonAnalyzerTK/blob/main/Watermelonizer5000ObjectDetection.ipynb)
