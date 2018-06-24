
Finalized the project goal.

Preparing the input dataset.


```python
from IPython.display import Image
```


```python
import cv2
```


```python
print(cv2.__version__)
```

    3.4.0



```python
import numpy as np
```


```python
cap = cv2.VideoCapture('traffic.mp4')
```


```python
#  VIDEO PROPERTIES
print ("Frame Width : ")
print (cap.get(3))  #Frame Width
print ("Frame Height :")
print (cap.get(4))  #Frame Height
fps = (cap.get(cv2.CAP_PROP_FPS))
print ("FPS :",fps)
```

    Frame Width : 
    426.0
    Frame Height :
    240.0
    FPS : 29.97002997002997



```python
fullmask = cv2.createBackgroundSubtractorMOG2()
while(cap.isOpened):
    ret, frame = cap.read()
    fgmask = fullmask.apply(frame)
    if fgmask is None:
        break
    (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 600:
            continue
        #get bounding box from countour
        (x, y, w, h) = cv2.boundingRect(c)
        #draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(fgmask, (x, y), (x + w, y + h), (0, 0, 0), 2)

    cv2.imshow('foreground and background',fgmask)
    cv2.imshow('rgb',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

```

![title](extracted.png)

I have to feed this as a mask into the Sbnet.
