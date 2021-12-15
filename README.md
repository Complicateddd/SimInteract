# A Simple Interactive VisionProj (Under development...)

## Detection Proj

- #### Hand detection 

  - Dataset Introduction (All taken from public data sets)
    - Ego-hand (first and third perspective, manual annotation):  http://vision.soic.indiana.edu/projects/egohands/
    
    - TI1k-Dataset (first perspective, manual annotation): https://github.com/MahmudulAlam/TI1K-Dataset
    
    - TV-Hand (from the movie screenshot, manual annotation): http://vision.cs.stonybrook.edu/~supreeth/
    
    - Oxford-Hand (video screenshot + VOC part, manual annotation): https://www.robots.ox.ac.uk/~vgg/data/hands/
    
    - COCO-Hand (COCO80, semi-automatic annotation): http://vision.cs.stonybrook.edu/~supreeth/
    
      ![img](https://github.com/Complicateddd/SimInteract/blob/master/demo/YOLOv5/train_batch0.jpg)
    
  - deal with:
    - Clean up incorrect labels, such as negative coordinates
    - Clean up duplicate pictures and annotations
    - Convert to YOLO training format
    
  - Data volume after processing:
    - Ego-hand: Approximately 4782 
    - TI1k-Dataset: Approximately 1000 
    -  TV-Hand: Approximately 4444 
    - Oxford-Hand: Approximately 5655 
    - Note: COCO-hand (S and B) has many wrong labels and missing labels, and the data is dirty, so it is temporarily not used
    
  - The data volume of the completed data set is 15033, the training data is 12026, and the test data is 3007.

- #### Detector Traning result

  - YOLOv5

    - Nano: Fineturn with 200 epochs

      |                                                              | img_size | mAP(0.5) | mAP(0.5:0.95) | Params       | epoch      |
      | :----------------------------------------------------------: | -------- | -------- | ------------- | ------------ | ---------- |
      | YOLOv5-s ([v-1.0]( https://github.com/ultralytics/yolov5/tree/5e970d45c44fff11d1eb29bfc21bed9553abf986 )) | 640      | 91.7     | 58.1          | 27.6M (.pth) | ~300 (15h) |

    - Demo

      - Detect in the wild

        ![img](https://github.com/Complicateddd/SimInteract/blob/master/demo/YOLOv5/yolov5nanoHand.gif)

      - With Sort (not deep)

        ![img](https://github.com/Complicateddd/SimInteract/blob/master/demo/YOLOv5/sort.gif)

        

      









