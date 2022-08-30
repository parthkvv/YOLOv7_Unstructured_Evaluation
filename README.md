YOLOv7 readme file(https://github.com/WongKinYiu/yolov7)

Dataset Preparation
	The base data directory should consist of the following files:
		"images" folder - containing the images for train, test and val
		"labels" folder - containing the labels for train, test and val in .txt 				                          yolo format. Each image has seperate file.  
		
		"class.txt" - Txt file with class names (Input to the conv_xml_to_txt.py file)

		Running conv_xml_to_txt.py at the base data directory:
		"train/txt" - generate txt files containing labels(class,xyxy) from original xml labels(Input)
		"val/txt" - generate txt files containing labels(class,xyxy) from original xml labels(Input)
		"train/txt" - generate txt files containing labels(class,xyxy) from original xml labels(Input)

	Generated custom dataset should be in the format :

	custom_dataset
	├── images
	│   ├── train
	│   │   ├── train0.jpg
	│   │   └── train1.jpg
	│   ├── val
	│   │   ├── val0.jpg
	│   │   └── val1.jpg
	│   └── test
	│       ├── test0.jpg
	│       └── test1.jpg
	└── labels
       	    ├── train
    	    │   ├── train0.txt
    	    │   └── train1.txt
    	    ├── val
    	    │   ├── val0.txt
    	    │   └── val1.txt
    	    └── test
        		├── test0.txt
        		└── test1.txt


TRAIN 

	- python train.py E:\IISc\Object_detection\YOLOv7\yolov7-main\weights\yolov7_training.pt --data E:\IISc\Object_detection\YOLOv7\yolov7-main\data\custom.yaml --workers 4 --batch-size 4 --img 640 640 --cfg E:\IISc\Object_detection\YOLOv7\yolov7-main\cfg\training\yolov7.yaml --name yolov7 --hyp E:\IISc\Object_detection\YOLOv7\yolov7-main\data\hyp.scratch.p5.yaml

	saved weights
	- runs\train\yolov7\weights

(NOTE: DELETE the train.cache, val.cache, test.cache files generated in main_dataset\labels after each run)


Evaluate :
	(For final mAP calculation on custom dataset with xywh labels)

	Set val variable value in data\custom.yaml as- 
	E:\\IISc\\Object_detection\\IDD\\backup\\images\\test	
 
	- python test.py --data E:\IISc\Object_detection\YOLOv7\yolov7-main\data\custom.yaml --img 640 --batch 2 --conf 0.001 --iou 0.65 --device 0 --weights E:\IISc\Object_detection\YOLOv7\yolov7-main\weights\yolov7_training.pt --name yolov7_640_val

(NOTE: DELETE the train.cache, val.cache, test.cache files generated in main_dataset\labels after each run)


EVALUATION :
	(Use xyxy format labels for label and image paths in main() of Evaluation.py) 
	- python Evaluation.py
		- Generate csv files with results in output_files
(Refer Readme_evaluation.txt for details)

(NOTE: DELETE the train.cache, val.cache, test.cache files generated in main_dataset\labels after each run)
