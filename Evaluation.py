from __future__ import absolute_import, division, print_function
from detect_custom import detect
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


INF_MIN = -99999999999999
INF_MAX =  99999999999999
IOU_THR = 0.5

# weights = r'E:\\IISc\\Object_detection\\YOLOv6\\YOLOv6-main\\weights\\best_ckpt.pt'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=filename) 

def get_model_scores_map(pred_boxes):   # Not Required - Kept just for reference

	model_scores_map = {}
	for img_id, val in pred_boxes.items():
		for score in val['scores']:
			if score not in model_scores_map.keys():
				model_scores_map[score] = [img_id]
			else:
				model_scores_map[score].append(img_id)
	return model_scores_map


def average_iou(gt_boxes, pred_boxes, pred_class, iou_thr):
    """Calculate number of True Positive, False Positive, False Negative from single batch of boxes.
    
    Arguments:-	
    gt_boxes : Location of predicted object as [xmin, ymin, xmax, ymax]
    pred_Box   : Dictionary of dictionary of ground truth object as [xmin, ymin, xmax, ymax] and 'scores'
    iou_thr  : Value of IOU to consider as threshold for classify correct prediction and false prediction.	
    
    Returns:
        dict : True Positive , False Positive, False Negative
    """
    # print(pred_class)
    # print("gt_boxes:",gt_boxes)
    # print("pred_boxes:",pred_boxes)

    # print("gt_length", len(gt_boxes))
    # print("pred_lenght", len(pred_boxes))

    # for cl in pred_class:
    # print("class:",pred_class)

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    # print("all_gt_indices",all_gt_indices)
    # print("all_pred_indices", all_pred_indices)

    if len(all_pred_indices) == 0:
        # print("going in pred_indixes==0")
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn ,'avg' : 0, 'total_obj' : len(gt_boxes) }
        
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn, 'avg' : 0, 'total_obj' : len(gt_boxes) }
        #return tp

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    # print("pred_box", pred_boxes)
    # print("gt_box", gt_boxes)
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            # for icl, class_name in enumerate(pred_class):
                # print("gt:", gt_box)
                # print("pred:", pred_box)
                # print("class:",pred_class)

                iou = calc_iou_individual(pred_box, gt_box) # Calculating IOU
                if iou > iou_thr:	
                    gt_idx_thr.append(igb)
                    pred_idx_thr.append(ipb)
                    ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        list_iou = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
                list_iou.append(ious[idx])

        # print("list_iou", list_iou)
        # print("pred_match_idx", pred_match_idx)
        # print("gt_match_idx", gt_match_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    # print("tp",tp)
    # print("fp",fp)
    # print("fn",fn)


    sumx = 0
    avg = 0
    if tp != 0:
        for i in list_iou:
            sumx += i
        avg = sumx/len(gt_boxes)
    else:
        if len(gt_boxes) == 0:
                avg = -1
        else:
                avg = 0
    
    # false_pos.append(fp)
    # true_pos.append(tp)
    # f_pos = np.cumsum(false_pos)
    # t_pos = np.cumsum(true_pos)
    # rec = t_pos/len(gt_boxes)
    # recall.append(rec)
    # prec = t_pos/(t_pos + f_pos)
    # precision.append(prec)
    # print("recall", recall)
    # print("precision", precision)
    # print(average_precision)
    # precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    # average_precision  = compute_ap(recall, precision)
    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn, 'avg' : avg, 'total_obj' : len(gt_boxes) } # Returns cells of Confusion Matrix

# def compute_avgp(recall, precision):
#     mrec = np.concatenate(([0.], recall, [1.]))
#     mpre = np.concatenate(([0.], precision, [0.]))

#     # compute the precision envelope
#     for i in range(mpre.size - 1, 0, -1):
#         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

#     # to calculate area under PR curve, look for points
#     # where X axis (recall) changes value
#     i = np.where(mrec[1:] != mrec[:-1])[0]

#     # and sum (\Delta recall) * prec
#     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap

def calc_iou_individual(pred_Box, gt_Box):
	x1_t, y1_t, x2_t, y2_t = list(map(int, list(map(float, list(gt_Box)))))
    
	x1_p, y1_p, x2_p, y2_p = pred_Box

	if (x1_p > x2_p) or (y1_p > y2_p):
		raise AssertionError(
			"Prediction box is malformed? pred box: {}".format(pred_Box))
	if (x1_t > x2_t) or (y1_t > y2_t):
		raise AssertionError(
			"Ground Truth box is malformed? true box: {}".format(gt_Box))

	if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
		return 0.0
    
	# Calculating Area of Intersection
	far_x = np.min([x2_t, x2_p])
	near_x = np.max([x1_t, x1_p])
	far_y = np.min([y2_t, y2_p])
	near_y = np.max([y1_t, y1_p])

	inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
	tbox_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
	pbox_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
	IOU = inter_area / (tbox_area + pbox_area - inter_area)  # Area of intersection / Area of Union s
	
	return IOU

def convert_polygon_to_rectangle(coordinates):  # Not Required - Kept just for reference 
	
	rect_coordinates = []
	for single in coordinates:
		xmax = INF_MIN
		ymax = INF_MIN
		xmin = INF_MAX
		ymin = INF_MAX
		for coord in single:
			xmin = min(coord[0],xmin)
			ymin = min(coord[1],ymin)
			xmax = max(coord[0],xmax)
			ymax = max(coord[1],ymax)
		rect_coordinates.append([xmin,ymin,xmax,ymax])
		
	return rect_coordinates


def find_average(dict_g, dict_p, img):
	gt_boxes = dict_g
	pred_boxes = dict_p
	g_y = average_iou(gt_boxes[img],pred_boxes[img]['boxes'],pred_boxes[img]['class'],IOU_THR)
	
	return g_y

def coordinate_as_gt(class_type):
	gd = []
	for dat in class_type:
		box = dat
		gd.append(box)
	return gd

# def yolo_detection(image_path):
#     global model
#     results = model(image_path)

#     inference_time = results.t[1]
#     pandas_results = results.pandas().xyxy[0]
#     csvfile = open('output_files\\retina_r.csv', 'w', encoding='utf-8')
#     csvfile_writer = csv.writer(csvfile, lineterminator='\n')
#     csvfile_writer.writerow(["index","box_points", "name", "percentage_probability"])
#     for i in range(len(pandas_results)):
#         box = np.array(pandas_results.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']]).astype(int)
#         name = pandas_results.iloc[i]['name']
#         confidence = pandas_results.iloc[i]['confidence']
#         csv_line = [i, box.astype(int), name, confidence]
#         csvfile_writer.writerow(csv_line)
#     csvfile.close()
#     return inference_time

def retina_thread(image_path):
    weights = 'E:\IISc\Object_detection\YOLOv7\yolov7-main\weights\yolov7_training.pt'
    source = image_path
    # yaml = 'E:\\IISc\\Object_detection\\YOLOv6\\YOLOv6-main\\data\\dataset.yaml'
    yolo_time = detect(source, weights)  # calling run will also generate a csv file -> 'yolo6_y.csv'(for individual images, each time)

    return yolo_time

    # yolo_time = yolo_detection(image_path)
    # return yolo_time

def main():
    # Initializing paths for test data
    lit = os.listdir("E://IISc//Object_detection//IDD//IDD_Detection//test//" + "txt//")   # GT
    
    path_gt = "E://IISc//Object_detection//IDD//IDD_Detection//test//" + "txt//"   # GT - same as lit
    path_image = "E://IISc//Object_detection//IDD//IDD_Detection//test//" + "images//"  # IMAGE 
    
    # -----------------------------------------------------------------------------------------------------
    # Initializing count and iou csv files for each of the 12 classes to store calculated values

    # Bicycle
    csv_bicycle_count = open('output_files\\bicycle_count.csv','a', newline='')                                                                     
    writer_bicycle_count = csv.writer(csv_bicycle_count)
    writer_bicycle_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_bicycle_iou = open('output_files\\bicycle_iou.csv','a', newline='')
    writer_bicycle_iou = csv.writer(csv_bicycle_iou)
    writer_bicycle_iou.writerow(["img_path", "avg_iou"])
    
    # Bus
    csv_bus_count = open('output_files\\bus_count.csv','a', newline='')
    writer_bus_count = csv.writer(csv_bus_count)
    writer_bus_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_bus_iou = open('output_files\\bus_iou.csv','a', newline='')
    writer_bus_iou = csv.writer(csv_bus_iou)
    writer_bus_iou.writerow(["img_path", "avg_iou"])

    # Traffic_sign
    csv_traffic_sign_count = open('output_files\\traffic_sign_count.csv','a', newline='')
    writer_traffic_sign_count = csv.writer(csv_traffic_sign_count)
    writer_traffic_sign_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_traffic_sign_iou = open('output_files\\traffic_sign_iou.csv','a', newline='')
    writer_traffic_sign_iou = csv.writer(csv_traffic_sign_iou)
    writer_traffic_sign_iou.writerow(["img_path", "avg_iou"])

    # Motorcycle
    csv_motorcycle_count = open('output_files\\motorcycle_count.csv','a', newline='')
    writer_motorcycle_count = csv.writer(csv_motorcycle_count)
    writer_motorcycle_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_motorcycle_iou = open('output_files\\motorcycle_iou.csv','a', newline='')
    writer_motorcycle_iou = csv.writer(csv_motorcycle_iou)
    writer_motorcycle_iou.writerow(["img_path", "avg_iou"])

    # Car
    csv_car_count = open('output_files\\car_count.csv','a', newline='')
    writer_car_count = csv.writer(csv_car_count)
    writer_car_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_car_iou = open('output_files\\car_iou.csv','a', newline='')
    writer_car_iou = csv.writer(csv_car_iou)
    writer_car_iou.writerow(["img_path", "avg_iou"])

    # Traffic_light
    csv_traffic_light_count = open('output_files\\traffic_light_count.csv','a', newline='')
    writer_traffic_light_count = csv.writer(csv_traffic_light_count)
    writer_traffic_light_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_traffic_light_iou = open('output_files\\traffic_light_iou.csv','a', newline='')
    writer_traffic_light_iou = csv.writer(csv_traffic_light_iou)
    writer_traffic_light_iou.writerow(["img_path", "avg_iou"])

    # Person
    csv_person_count = open('output_files\\person_count.csv','a', newline='')
    writer_person_count = csv.writer(csv_person_count)
    writer_person_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_person_iou = open('output_files\\person_iou.csv','a', newline='')
    writer_person_iou = csv.writer(csv_person_iou)
    writer_person_iou.writerow(["img_path", "avg_iou"])

    # Vehicle_fallback
    csv_vehicle_fallback_count = open('output_files\\vehicle_fallback_count.csv','a', newline='')
    writer_vehicle_fallback_count = csv.writer(csv_vehicle_fallback_count)
    writer_vehicle_fallback_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_vehicle_fallback_iou = open('output_files\\vehicle_fallback_iou.csv','a', newline='')
    writer_vehicle_fallback_iou = csv.writer(csv_vehicle_fallback_iou)
    writer_vehicle_fallback_iou.writerow(["img_path", "avg_iou"])

    # Truck
    csv_truck_count = open('output_files\\truck_count.csv','a', newline='')
    writer_truck_count = csv.writer(csv_truck_count)
    writer_truck_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_truck_iou = open('output_files\\truck_iou.csv','a', newline='')
    writer_truck_iou = csv.writer(csv_truck_iou)
    writer_truck_iou.writerow(["img_path", "avg_iou"])

    # Autorickshaw
    csv_autorickshaw_count = open('output_files\\autorickshaw_count.csv','a', newline='')
    writer_autorickshaw_count = csv.writer(csv_autorickshaw_count)
    writer_autorickshaw_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_autorickshaw_iou = open('output_files\\autorickshaw_iou.csv','a', newline='')
    writer_autorickshaw_iou = csv.writer(csv_autorickshaw_iou)
    writer_autorickshaw_iou.writerow(["img_path", "avg_iou"])

    # Animal
    csv_animal_count = open('output_files\\animal_count.csv','a', newline='')
    writer_animal_count = csv.writer(csv_animal_count)
    writer_animal_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_animal_iou = open('output_files\\animal_iou.csv','a', newline='')
    writer_animal_iou = csv.writer(csv_animal_iou)
    writer_animal_iou.writerow(["img_path", "avg_iou"])

    # Rider
    csv_rider_count = open('output_files\\rider_count.csv','a', newline='')
    writer_rider_count = csv.writer(csv_rider_count)
    writer_rider_count.writerow(["img_path", "total_obj", "true_pos", "false_pos", "false_neg"])

    csv_rider_iou = open('output_files\\rider_iou.csv','a', newline='')
    writer_rider_iou = csv.writer(csv_rider_iou)
    writer_rider_iou.writerow(["img_path", "avg_iou"])

    # Latency
    lts = open('output_files\\latency.csv','a')
    writer_lts = csv.writer(lts)
    writer_lts.writerow(["img_path", "time"])
    # -------------------------------------------------------------------------------------------------------

    cnt=0
    for f_n in lit:
        cnt += 1
        print(cnt)
        # print(f_n)
        file_gt = open(path_gt + f_n)
        # json_gt = json.load(file_gt)         # old
        txt_file = path_gt + f_n

        ind = f_n.split('.txt')[0] 
        # img = f_n[0:ind] +"_leftImg8bit.png"
        img = ind + ".jpeg"

        path_img = path_image + img
        # path_img = "./Data/IMAGE/train/0/010515_leftImg8bit.png"
        
        # List for all classes objects seperately for this image
        
        bicycle = []
        bus = []
        traffic_sign = []
        motorcycle = []
        car = []
        traffic_light = []
        person = []
        vehicle_fallback = []
        truck = []
        autorickshaw = []
        animal = []
        rider = []
        
        
        # objects= json_gt['objects']      # old
        with open(txt_file) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                # print("line class", line.split(" ")[0])
                if(line.split(" ")[0] == "0"):
                    bicycle.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])         # [xmin,ymin,xmax,ymax] ->  bounding box values
                    # print("bicycle:", bicycle)
                elif(line.split(" ")[0] == '1'):                                                                                         # xmin -> line.split(" ")[1]
                    bus.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])             # ymin -> line.split(" ")[2]
                    # print("bus:", bus)
                elif(line.split(" ")[0] == '2'):                                                                                         # xmax -> line.split(" ")[3]
                    traffic_sign.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])    # xmax -> line.split(" ")[4]
                    # print("traffic_sign:", traffic_sign)
                elif(line.split(" ")[0] == '3'):
                    motorcycle.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("motorcycle:", motorcycle)
                elif(line.split(" ")[0] == '4'):
                    car.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("car:", car)
                elif(line.split(" ")[0] == '5'):
                    traffic_light.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("traffic_light:", traffic_light)
                elif(line.split(" ")[0] == '6'):
                    person.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("person:", person)
                elif(line.split(" ")[0] == '7'):
                    vehicle_fallback.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("vehicle_fallback:", vehicle_fallback)
                elif(line.split(" ")[0] == '8'):
                    truck.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("truck:", truck)
                elif(line.split(" ")[0] == '9'):
                    autorickshaw.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("autorickshaw:", autorickshaw)
                elif(line.split(" ")[0] == '10'):
                    animal.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("animal:", animal)
                elif(line.split(" ")[0] == '11'):
                    rider.append([line.split(" ")[1],line.split(" ")[2],line.split(" ")[3],line.split(" ")[4].split('\n')[0]])
                    # print("rider:", rider)
        
        # for obj in objects:  --- old(removed)

        # animals_rect = convert_polygon_to_rectangle(animals)
        # motorcycle_rect = convert_polygon_to_rectangle(motorcycle)
        # truck_rect = convert_polygon_to_rectangle(truck)
        # caravan_rect = convert_polygon_to_rectangle(caravan)
        # auto_rect = convert_polygon_to_rectangle(auto)
            
        # convert_polygon_to_rectangle function output format -> rect_coordinates.append([xmin,ymin,xmax,ymax])    
        
        fn_retina =  "output_files\\yolo6_y.csv"
        
        retina_time = retina_thread(path_img)
        
        
        writer_lts.writerow([path_img, retina_time])
        
        
        # Making Ground Boxes as [Xmin, Ymin , Xmax , Ymax] from polygons

        # Renaming the original values and declaring new ones(extra) 
        # gd_ar - gd_br   ,dict_ar - dict_br        -----> bicycle
        # gd_mr - gd_br1  ,dict_mr - dict_br1       -----> bus
        # gd_tr - gd_tr   ,dict_tr - dict_tr        -----> traffic_sign
        # gd_cr - gd_mr   ,dict_cr - dict_mr        -----> motorcycle
        # gd_tra - gd_ca  ,dict_tra - dict_ca       -----> car
        # gd_tra          ,dict_tra                 -----> traffic_light
        # gd_pr           ,dict_pr                  -----> person
        # gd_vr           ,dict_vr                  -----> vehicle_fallback
        # gd_tra1         ,dict_tra1                -----> truck
        # gd_ar           ,dict_ar                  -----> autorickshaw
        # gd_anr          ,dict_anr                 -----> animal
        # gd_rr           ,dict_rr                  -----> rider

        gd_br = coordinate_as_gt(bicycle)
        dict_br = {img : gd_br}
        
        gd_br1 = coordinate_as_gt(bus)
        dict_br1 = {img : gd_br1}
        
        gd_tr = coordinate_as_gt(traffic_sign)
        dict_tr = {img : gd_tr}
        
        gd_mr = coordinate_as_gt(motorcycle)
        dict_mr = {img : gd_mr}
        
        gd_ca = coordinate_as_gt(car)
        dict_ca = {img : gd_ca}
        # print("dict_ca",dict_ca)

        gd_tra = coordinate_as_gt(traffic_light)
        dict_tra = {img : gd_tra}

        gd_pr = coordinate_as_gt(person)
        dict_pr = {img : gd_pr}

        gd_vr = coordinate_as_gt(vehicle_fallback)
        dict_vr = {img : gd_vr}

        gd_tra1 = coordinate_as_gt(truck)
        dict_tra1 = {img : gd_tra1}

        gd_ar = coordinate_as_gt(autorickshaw)
        dict_ar = {img : gd_ar}

        gd_anr = coordinate_as_gt(animal)
        dict_anr = {img : gd_anr}

        gd_rr = coordinate_as_gt(rider)
        dict_rr = {img : gd_rr}

        #------------------------------------------------------------------------------
        # Retina Net Model Calculations
        
        retina_box_br = []
        retina_box_br1 = []
        retina_box_tr = []
        retina_box_mr = []
        retina_box_ca = []
        retina_box_tra = []
        retina_box_pr = []
        retina_box_vr = []
        retina_box_tra1 = []
        retina_box_ar = []
        retina_box_anr = []
        retina_box_rr = []
      
        retina_class_br = []
        retina_class_br1 = []
        retina_class_tr = []
        retina_class_mr = []
        retina_class_ca = []
        retina_class_tra = []
        retina_class_pr = []
        retina_class_vr = []
        retina_class_tra1 = []
        retina_class_ar = []
        retina_class_anr = []
        retina_class_rr = []
        
        retina_score = []
        df_r = pd.read_csv(fn_retina)
        for index, row in df_r.iterrows():
            if(row['name'] == 'bicycle'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_br.append(l3)
                retina_class_br.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'bus'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_br1.append(l3)
                retina_class_br1.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'traffic sign'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_tr.append(l3)
                retina_class_tr.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'motorcycle'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_mr.append(l3)
                retina_class_mr.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'car'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                # print("l2",l2)
                l3 = [int(i) for i in l2]
                retina_box_ca.append(l3)
                retina_class_ca.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'traffic light'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_tra.append(l3)
                retina_class_tra.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'person'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_pr.append(l3)
                retina_class_pr.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'vehicle fallback'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_vr.append(l3)
                retina_class_vr.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'truck'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_tra1.append(l3)
                retina_class_tra1.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'autorickshaw'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_ar.append(l3)
                retina_class_ar.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'animal'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_anr.append(l3)
                retina_class_anr.append(row['name'])
                retina_score.append(row['percentage_probability'])
            elif(row['name'] == 'rider'):
                l = row['box_points']
                l1 = l[1:-1]
                l2 = l1.split(',')
                l3 = [int(i) for i in l2]
                retina_box_rr.append(l3)
                retina_class_rr.append(row['name'])
                retina_score.append(row['percentage_probability'])
        
        dr_br = {'boxes': retina_box_br, 'scores': retina_score, 'class': retina_class_br}
        dict_r_br = {img: dr_br}

        dr_br1 = {'boxes': retina_box_br1, 'scores': retina_score, 'class': retina_class_br1}
        dict_r_br1 = {img: dr_br1}

        dr_tr = {'boxes': retina_box_tr, 'scores': retina_score, 'class': retina_class_tr}
        dict_r_tr = {img: dr_tr}

        dr_mr = {'boxes': retina_box_mr, 'scores': retina_score, 'class': retina_class_mr}
        dict_r_mr = {img: dr_mr}

        dr_ca = {'boxes': retina_box_ca, 'scores': retina_score, 'class': retina_class_ca}
        dict_r_ca = {img: dr_ca}

        dr_tra = {'boxes': retina_box_tra, 'scores': retina_score, 'class': retina_class_tra}
        dict_r_tra = {img: dr_tra}

        dr_pr = {'boxes': retina_box_pr, 'scores': retina_score, 'class': retina_class_pr}
        dict_r_pr = {img: dr_pr}

        dr_vr = {'boxes': retina_box_vr, 'scores': retina_score, 'class': retina_class_vr}
        dict_r_vr = {img: dr_vr}

        dr_tra1 = {'boxes': retina_box_tra1, 'scores': retina_score, 'class': retina_class_tra1}
        dict_r_tra1 = {img: dr_tra1}

        dr_ar = {'boxes': retina_box_ar, 'scores': retina_score, 'class': retina_class_ar}
        dict_r_ar = {img: dr_ar}

        dr_anr = {'boxes': retina_box_anr, 'scores': retina_score, 'class': retina_class_anr}
        dict_r_anr = {img: dr_anr}

        dr_rr = {'boxes': retina_box_rr, 'scores': retina_score, 'class': retina_class_rr}
        dict_r_rr = {img: dr_rr}

        #--------------------------------------------------------------------------------
        
        # MRCNN Model Calculations - not needed(removed)
        # Yolo Model Calculation - not needed(removed)
        
        #---------------------------------------------------- Bicycle ---------------------
        
        out_retina = find_average(dict_br, dict_r_br , img)
        
        row = [path_img, out_retina['avg']]
        writer_bicycle_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]  # 'false_pos' & 'false_neg' also added
        writer_bicycle_count.writerow(row)
        csv_bicycle_iou.flush()
        csv_bicycle_count.flush()
        
        #----------------------------------------------- Bus -----------------------
        
        out_retina = find_average(dict_br1, dict_r_br1, img)
        
        row = [path_img, out_retina['avg']]
        writer_bus_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_bus_count.writerow(row)
        csv_bus_iou.flush()
        csv_bus_count.flush()
        
        #-----------------------------------------------  Traffic_sign -------------------------------
        
        out_retina = find_average(dict_tr, dict_r_tr, img)
        
        row = [path_img, out_retina['avg']]
        writer_traffic_sign_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_traffic_sign_count.writerow(row)
        csv_traffic_sign_iou.flush()
        csv_traffic_sign_count.flush()
        #--------------------------------------------- Motorcycle -----------------------------------
        
        out_retina = find_average(dict_mr, dict_r_mr, img)
        
        row = [path_img, out_retina['avg']]
        writer_motorcycle_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_motorcycle_count.writerow(row)
        csv_motorcycle_iou.flush()
        csv_motorcycle_count.flush()
        #-------------------------------------------- Car -------------------------------------
        out_retina = find_average(dict_ca, dict_r_ca, img)
        
        row = [path_img, out_retina['avg']]
        writer_car_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_car_count.writerow(row)
        csv_car_iou.flush()
        csv_car_count.flush()
        #-------------------------------------------- Traffic_light -------------------------------------

        out_retina = find_average(dict_tra, dict_r_tra, img)
        
        row = [path_img, out_retina['avg']]
        writer_traffic_light_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_traffic_light_count.writerow(row)
        csv_traffic_light_iou.flush()
        csv_traffic_light_count.flush()
        #-------------------------------------------- Person -------------------------------------
        
        out_retina = find_average(dict_pr, dict_r_pr, img)
        
        row = [path_img, out_retina['avg']]
        writer_person_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_person_count.writerow(row)
        csv_person_iou.flush()
        csv_person_count.flush()
        #-------------------------------------------- Vehicle_fallback -------------------------------------
        
        out_retina = find_average(dict_vr, dict_r_vr, img)
        
        row = [path_img, out_retina['avg']]
        writer_vehicle_fallback_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_vehicle_fallback_count.writerow(row)
        csv_vehicle_fallback_iou.flush()
        csv_vehicle_fallback_count.flush()
        #-------------------------------------------- Truck -------------------------------------
        
        out_retina = find_average(dict_tra1, dict_r_tra1, img)
        
        row = [path_img, out_retina['avg']]
        writer_truck_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_truck_count.writerow(row)
        csv_truck_iou.flush()
        csv_truck_count.flush()
        #-------------------------------------------- Autorickshaw -------------------------------------
        
        out_retina = find_average(dict_ar, dict_r_ar, img)
        
        row = [path_img, out_retina['avg']]
        writer_autorickshaw_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_autorickshaw_count.writerow(row)
        csv_autorickshaw_iou.flush()
        csv_autorickshaw_count.flush()
        #-------------------------------------------- Animal ---------------------------------------------
        
        out_retina = find_average(dict_anr, dict_r_anr, img)
        
        row = [path_img, out_retina['avg']]
        writer_animal_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_animal_count.writerow(row)
        csv_animal_iou.flush()
        csv_animal_count.flush()
        #-------------------------------------------- Rider -------------------------------------
    
        out_retina = find_average(dict_rr, dict_r_rr, img)
        # print("dict_rr - rider", dict_rr)
        
        row = [path_img, out_retina['avg']]
        writer_rider_iou.writerow(row)
        
        row = [path_img, out_retina['total_obj'], out_retina['true_pos'], out_retina['false_pos'], out_retina['false_neg'] ]
        writer_rider_count.writerow(row)
        csv_rider_iou.flush()
        csv_rider_count.flush()
        
main()




