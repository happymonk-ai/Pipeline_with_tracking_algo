import argparse
import asyncio

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from math import ceil
from PIL import Image

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from matplotlib import gridspec

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

from keras.applications.resnet import ResNet50 
#from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet import preprocess_input 
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import cv2
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from collections import Counter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

 # Load model
device_track=''
devices = select_device(device_track)
model = DetectMultiBackend(WEIGHTS / '27Sep_2022.pt', device=devices, dnn=False, data=None, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size

# # face-id model
# model_1 = ResNet50(weights = 'imagenet', include_top = True)

# NL = 2 # No. of layers to be processed

# print('Total', NL, ' layers used')

# gallery_path = './gallery'
# # count = 0
# classifier_list = []  
# pca_list = [] 
# # Y_list = []
# for i in range(NL): #NL
#     file_id = []
#     filepath = []
#     filename_list = []
#     feature_list = []
#     target = []
    
#     for file in os.listdir(gallery_path):
#         if file.endswith('jpg'):
#             # count+=1
#             filename = file.split('.')[0]
#             # print(filename[0])
#             file = os.path.join(gallery_path, file)
#             file_id.append(filename[0])
#             filepath.append(file)
#             filename_list.append(filename)
#     file_id_unique = list(set(file_id))
#     # layer_name = filename + '_layer_' + str(i)

#     # layer_path = os.path.join(query_path, layer_name)
#     # if not os.path.exists(layer_path):
#     #     os.makedirs(layer_path)
#     print('Layer No.: ', i)
#     model_layer = Model(inputs = model_1.inputs, outputs = model_1.layers[i].output)
#     # Extracting our features
#     for item1 in file_id_unique:
#         # print(item1)
#         while True:
#             # print(file_id, item1)
#             try:
#                 # print(item1)
#                 index = file_id.index(item1)
#                 # if index is not None:
#                 file_id[index] = '-1'
#                 file = filepath[index] 
#                 filename = filename_list[index]
#                 layer_name = filename + '_layer_' + str(i)
#                 layer_path = os.path.join(gallery_path, layer_name)
#                 # if not os.path.exists(layer_path): # Uncomment these two lines for feature visualization 
#                 #     os.makedirs(layer_path)                      
#                 image1 = cv2.imread(file)
#                 image2 = cv2.cvtColor(image1, cv.COLOR_BGR2RGB)
#                 # Resize image to 224x224 size
#                 image3 = cv2.resize(image2, (224, 224)).reshape(-1, 224, 224, 3)
#                 # We need to preprocess imageto fulfill ResNet50 requirements
#                 image = preprocess_input(image3)

#                 features = model_layer.predict(image)
#                 # print(features.shape,"feature shape")
#                 n_features = features.shape[-1]
#                 # print(n_features, 'No. of features')

#                 for item in range(n_features):
#                     # print(item, 'Line 83')
#                     # try:
#                     img = features[0, :, :, item]
#                     mean, std = img.mean(), img.std()
#                     # print(mean, std, 'Mean and Standard deviation')
#                     if std==0.0:
#                         # print('Std Dev 0 was encountered')
#                         continue
#                     img = (img - mean)/std
#                     # clip pixel values to [-1,1]
#                     img = np.clip(img, -1.0, 1.0)
#                     # shift from [-1,1] to [0,1] with 0.5 mean
#                     img = 255*(img + 1.0) / 2.0
#                     # confirm it had the desired effect
#                     mean, std = img.mean(), img.std()
#                     # print(img, "positive global feature image")
#                     layer_feature_file = os.path.join(layer_path, str(item))
#                     # print(layer_feature_file + '.jpg')
#                     # cv2.imwrite(layer_feature_file + '.jpg', img)
#                     (row,col) = img.shape
#                     img = np.reshape(img, row*col)
#                     feature_list.append(img)
#                     # print(filename[0])
#                     # target.append(filename[0])
#                     target.append(item1)

#             except ValueError as e:
#                 # print(e)
#                 break
#     feature_list = np.array(feature_list)
#     target = np.array(target)
#     # target = np.reshape(target,(len(target),1)) 
#     print(np.shape(feature_list), np.shape(target), 'Shapes of feature list and target')
#     # define ordinal encoding
#     # encoder = OrdinalEncoder()
#     encoder = LabelEncoder()
#     # encoder = OneHotEncoder(sparse = False)
#     Y = encoder.fit_transform(target)
#     Y = np.reshape(Y,(len(Y),))
#     print(np.shape(Y))
#     # print(Y)

#     X_train, X_test, y_train, y_test = train_test_split(feature_list, Y)
#     pca = PCA().fit(X_train)
#     n_comp_hold = np.where(pca.explained_variance_ratio_.cumsum() > 0.95)
#     n_comp_list = list(n_comp_hold)
#     n_comp = len(n_comp_list[0])
#     # n_comp = len(np.where(pca.explained_variance_ratio_.cumsum() > 0.95))
#     print(n_comp, 'No. of PCA components > 95%')
#     pca = PCA(n_components = n_comp).fit(X_train)
#     X_train_pca = pca.transform(X_train)
#     classifier = SVC(probability=True).fit(X_train_pca, y_train)
#     X_test_pca = pca.transform(X_test)
#     predictions = classifier.predict(X_test_pca)
#     print(classification_report(y_test, predictions))
#     classifier_list.append(classifier)
#     pca_list.append(pca)


person_count = []
vehicle_count = []
avg_Batchcount_person =[]
avg_Batchcount_vehicel = []
track_person = []
track_vehicle = []
batch_person_id = []
detect_count = []
act_output = []

@torch.no_grad()
def run(
        source,
        queue1,
        queue2,
        queue3,
        queue4,
        queue5,
        yolo_weights=WEIGHTS / '27Sep_2022.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        save_vid=True,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'Nats_output/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, devices, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(devices)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # video file
            if source.endswith(VID_FORMATS):
                txt_file_name = p.stem
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

                # Print results
                for c in det[:, -1].unique():
                    global vehicle_count , license
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if names[int(c)] == "Person" :
                        print(f"{n}","line 338")
                        person_count.append(int(f"{n}"))
                        print("person detected")
                    if names[int(c)] == "Vehicle":
                       vehicle_count.append(int(f"{n}"))
                       print("vehicel detected")

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(det.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                crop_img = save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                                
                                # #face-id prediction
                                # image2 = cv2.cvtColor(crop_img, cv.COLOR_BGR2RGB)
                                # # Resize image to 224x224 size
                                # image3 = cv2.resize(image2, (224, 224)).reshape(-1, 224, 224, 3)
                                # # We need to preprocess imageto fulfill ResNet50 requirements
                                # image = preprocess_input(image3)
                                # prediction_list = []
                                # prediction_majority_list = []
                                # # prediction_prob_list = []
                                # for i_1 in range(NL): #NL
                                #     feature_list = []
                                #     # print('Layer No.: ', i)
                                #     model_layer = Model(inputs = model_1.inputs, outputs = model_1.layers[i_1].output)
                                #     features = model_layer.predict(image)
                                #     n_features = features.shape[-1]

                                #     for item in range(n_features):
                                #         # print(item, 'Line 83')
                                #         img = features[0, :, :, item]
                                #         mean, std = img.mean(), img.std()
                                #         if std==0.0:
                                #             # print('Std Dev 0 was encountered')
                                #             continue
                                #         img = (img - mean)/std
                                #         # clip pixel values to [-1,1]
                                #         img = np.clip(img, -1.0, 1.0)
                                #         # shift from [-1,1] to [0,1] with 0.5 mean
                                #         img = 255*(img + 1.0) / 2.0
                                #         # confirm it had the desired effect
                                #         mean, std = img.mean(), img.std()
                                #         # print(img, "positive global feature image")
                                #         # layer_feature_file = os.path.join(layer_path, str(item))
                                #         # # print(layer_feature_file + '.jpg')
                                #         # cv2.imwrite(layer_feature_file + '.jpg', img)
                                #         (row,col) = img.shape
                                #         img = np.reshape(img, row*col)
                                #         feature_list.append(img)
                                #     feature_list = np.array(feature_list)
                                #     # print(np.shape(feature_list), 'Shape of feature list')
                                #     X_test_pca = pca_list[i_1].transform(feature_list)
                                #     predictions = classifier_list[i_1].predict(X_test_pca)
                                #     Y_name = encoder.inverse_transform(predictions)
                                #     c = Counter(Y_name)
                                #     value, count = c.most_common()[0]
                                #     prediction_list.append(Y_name)
                                #     prediction_majority_list.append(value)
                                # c = Counter(prediction_majority_list)
                                # value, count = c.most_common()[0]
                                # print(value, 'Majority prediction for: ', f'{id}/{p.stem}.jpg')
                                
                LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')

            else:
                #strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
           
            # Save results (image with detections)
            if save_vid:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

            # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        
    #people Count
    sum_count = 0
    for x in person_count:
        sum_count += int(x)
        # if int(x) % 2 == 0:
        #     track_person.append(0)
        # else:
        #     track_person.append(1)
    try :
        avg = ceil(sum_count/len(person_count))
        avg_Batchcount_person.append(str(avg))
    except ZeroDivisionError:
        avg_Batchcount_person.append("0")
        print("No person found ")
        
    for iten in avg_Batchcount_person:
        for i in range(int(iten[0])):
            track_person.append("1")
        
    sum_count = 0
    for x in vehicle_count:
        sum_count += int(x)
        # if int(x) % 2 == 0:
        #     track_vehicle.append(0)
        # else:
        #     track_vehicle.append(1)
    try :
        avg = ceil(sum_count/len(vehicle_count))
        avg_Batchcount_vehicel.append(str(avg))
    except ZeroDivisionError:
        avg_Batchcount_vehicel.append("0")
        print("No Vehicle found ")
    
    for iten in avg_Batchcount_vehicel:
        for i in range(int(iten[0])):
            track_vehicle.append("1")
        
    if len(person_count) > 0 or len(vehicle_count) > 0 :
        detect_count.append(1)
    else:
        detect_count.append(0)

    queue1.put(avg_Batchcount_person)
    queue2.put(avg_Batchcount_vehicel)
    queue3.put(detect_count)
    queue4.put(track_person)
    queue5.put(track_vehicle)

    avg_Batchcount_person.clear()
    avg_Batchcount_vehicel.clear()
    detect_count.clear()
    track_person.clear()
    track_vehicle.clear()

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


if __name__ == "__main__":
    run(source="gray_scale.mp4")
    
