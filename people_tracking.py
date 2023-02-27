import numpy 
import cv2
import glob
import time
from os.path import join as opjoin
import numpy as np
import random
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from datetime import datetime
from numpy.lib import type_check
from turbojpeg import TurboJPEG 
from icecream import ic
import sys
import csv
import swift_save
import uuid
import json

# define zone boundary
with open(R"TAIWAI_000000.json", 'r') as fid:
    zone_info = json.load(fid)
shapes = zone_info['shapes']
zone1 = shapes[0]['points']
zone2 = shapes[1]['points']

debug_mode = True
draw_car_rectangle = debug_mode
draw_carplate_not_within_car = debug_mode
write_label_txt = False
write_jpeg = True


def save(image, oname, jpeg_engine):
    out_file = open(oname, 'wb')
    out_file.write(jpeg_engine.encode(image, quality=97))
    out_file.close()


def load_img(fname, yolo_w_h, jpeg_engine):
    in_file = open(fname, 'rb')
    img = jpeg_engine.decode(in_file.read())
    in_file.close()
    img_1 = cv2.resize(img, yolo_w_h, interpolation=cv2.INTER_AREA)
    og_height = img.shape[0]
    og_width = img.shape[1]
    return img, img_1, og_width, og_height, fname


def draw_label(fname, img):
    label_fname = os.path.splitext(fname)[0] + '.txt'
    if os.path.exists(label_fname):
        with open(label_fname, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            new_file = ''
            for row in spamreader:
                cls = int(row[0])
                if cls == 0:
                    x, y, w, h = [float(row[1])*img.shape[1], float(row[2])*img.shape[0], float(row[3])*img.shape[1], float(row[4])*img.shape[0] ]
                    left, right, top, bottom = int(x - w/2), int(x + w/2), int(y - h/2), int(y + h/2)
                    cv2.ellipse(img, (int(x),int(y)), (int(w/2),int(h/2)), 0, 0, 360, (55, 200, 200), 1)
                elif cls == 1:
                    x, y, w, h = [float(row[1])*img.shape[1], float(row[2])*img.shape[0], float(row[3])*img.shape[1], float(row[4])*img.shape[0] ]
                    left, right, top, bottom = int(x - w/2), int(x + w/2), int(y - h/2), int(y + h/2)
                    cv2.ellipse(img, (int(x),int(y)), (int(w/2),int(h/2)), 0, 0, 360, (255, 150, 20), 1)



def match_label(fname, img, carplate_area, area_list, area_list_all_label):
    missing_cnt = 0
    total_cnt = 0
    label_fname = os.path.splitext(fname)[0] + '.txt'
    if os.path.exists(label_fname):
        with open(label_fname, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            new_file = ''
            for row in spamreader:
                cls = int(row[0])
                #if cls == 0:
                if True:
                    x, y, w, h = [float(row[1])*img.shape[1], float(row[2])*img.shape[0], float(row[3])*img.shape[1], float(row[4])*img.shape[0] ]
                    left, right, top, bottom = int(x - w/2), int(x + w/2), int(y - h/2), int(y + h/2)
                    cv2.ellipse(img, (int(x),int(y)), (int(w/2),int(h/2)), 0, 0, 360, (55, 200, 200), 1)
                    total_cnt += 1
                    area = (bottom-top) * (right-left)
                    area_list_all_label.append(area)
                    if carplate_area[top:bottom, left:right, :].sum() == 0:
                        missing_cnt += 1
                        area_list.append(area)
    return missing_cnt, total_cnt


def load_image_proc(files, yolo_w_h, output_q):
    from multiprocessing import shared_memory
    import numpy as np
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_dll = os.path.join( dir_path , R"turbojpeg.dll")
    jpeg_engine = TurboJPEG(path_dll)

    max_mem_size = 40*3072*1536
    shm_list = []
    for n in range(2*3):
        shm = shared_memory.SharedMemory(create=True, size=max_mem_size)
        shm_list.append(shm)

    shm_cnt = 0
    for fn in files:
            in_file = open(fn, 'rb')
            img = jpeg_engine.decode(in_file.read())
            in_file.close()
            if img.shape[0] > yolo_w_h[1]:
                img_1 = cv2.resize(img, yolo_w_h, interpolation=cv2.INTER_AREA)
            else:
                img_1 = cv2.resize(img, yolo_w_h, cv2.INTER_LINEAR)     # because shrinking image using cv2.INTER_AREA sucks, significant aliasing is observed.

            og_height = img.shape[0]
            og_width = img.shape[1]
            shm_a = shm_list[2*shm_cnt]
            shm_b = shm_list[2*shm_cnt+1]
            shm_cnt += 1
            if shm_cnt > 2:
                shm_cnt = 0
            npy_view = np.ndarray(img.shape, dtype=img.dtype, buffer=shm_a.buf)
            npy_view[:] = img[:]
            npy_view2 = np.ndarray(img_1.shape, dtype=img_1.dtype, buffer=shm_b.buf)
            npy_view2[:] = img_1[:]
            output_q.put([shm_a.name, img.shape, shm_b.name, img_1.shape, og_width, og_height, fn])
        #except:
        #    print('Error in load_image_proc()  ', fn)
            
    output_q.put(None)
    output_q.put(None)

class precision_recall():
    def __init__(self):
        self.miss_cnt = 0
        self.total_label_cnt = 0
        self.n_false_pos = 0
        self.n_total_pred = 0
        self.area_list = []
        self.area_list_all_label = []


def compute_iou(a_xy_min, a_xy_max, b_xy_min, b_xy_max):
    min_inter_xy = np.maximum(a_xy_min, b_xy_min)
    max_inter_xy = np.minimum(a_xy_max, b_xy_max)
    inter_wh = max_inter_xy - min_inter_xy
    inter_wh = np.maximum( inter_wh, [0.,0.] )
    inter_area = inter_wh[0] * inter_wh[1]
    a_wh = a_xy_max - a_xy_min
    b_wh = b_xy_max - b_xy_min
    union_area = a_wh[0] * a_wh[1] + b_wh[0] * b_wh[1] - inter_area
    return inter_area/union_area


def predict_track(files, output_folder, prog_que=None):

    history_1 = []
    history_2 = []
    history_3 = []
    history_4 = []

    n_person_in = 0
    n_person_out = 0
    swift = swift_save.Swift_save()

    #os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_dll = os.path.join( dir_path , R"turbojpeg.dll")
    jpeg_engine = TurboJPEG(path_dll)

    import predictor_oo
    predictor = predictor_oo.darknet_oo()
    predictor.loadnet('the.cfg', 'the.weights', 'obj.data')
    yolo_h = predictor.network_height()
    yolo_w = predictor.network_width()
    yolo_w_h = (yolo_w, yolo_h)

    load_image_q = mp.Queue(1)

    proc_load = mp.Process(target=load_image_proc, args=(files, yolo_w_h, load_image_q,))
    proc_load.start()
    bn = 0
    while True:
        
        img_ls = load_image_q.get()
        if img_ls is None:
            break

        shm_a = shared_memory.SharedMemory( img_ls[0] )
        img = np.ndarray(img_ls[1], dtype=np.uint8, buffer=shm_a.buf)
        img = img//2

        cv2.polylines(img, [np.array(zone1).astype(int)], isClosed=True, color=(255,50,0), thickness=2)
        cv2.polylines(img, [np.array(zone2).astype(int)], isClosed=True, color=(0x00,0xa5,0xff), thickness=2)

        shm_b = shared_memory.SharedMemory( img_ls[2] )
        img_1 = np.ndarray(img_ls[3], dtype=np.uint8, buffer=shm_b.buf)

        og_wh = np.array([img_ls[4], img_ls[5]])
        fname = img_ls[6]

        results = predictor.performDetect( img_1, thresh=0.4 )
        boxes = []
        for k in range(len(results)):
            results[k][2][0] *= og_wh[0]/yolo_w
            results[k][2][1] *= og_wh[1]/yolo_h
            results[k][2][2] *= og_wh[0]/yolo_w
            results[k][2][3] *= og_wh[1]/yolo_h
            box_xy = np.array([results[k][2][0], results[k][2][1]])
            box_wh = np.array([results[k][2][2], results[k][2][3]])
            boxes.append( [box_xy, box_wh, '', ''] )            
        n_person = 0
        for n in range(len(results)):
            x,y,w,h = results[n][2]
            left, right, top, bottom = [int(x-w/2), int(x+w/2), int(y-h/2), int(y+h/2)]
            if left < 0:
                left = 0
            n_person += 1
            cv2.rectangle(img, (left,top), (right,bottom), (0,255,0), 2)
        n_overlap = 0
        for idx_a, box_a in enumerate(boxes):
            box_a_xy = box_a[0]
            box_a_wh = box_a[1]
            box_a_xy_min = box_a_xy - box_a_wh/2
            box_a_xy_max = box_a_xy + box_a_wh/2
            hit = False
            for box_b in history_1:
                box_b_name = box_b[2]
                box_b_xy = box_b[0]
                box_b_wh = box_b[1]
                box_b_xy_min = box_b_xy - box_b_wh/2
                box_b_xy_max = box_b_xy + box_b_wh/2
                iou = compute_iou(box_a_xy_min, box_a_xy_max, box_b_xy_min, box_b_xy_max)
                if iou > 0.4:
                    n_overlap += 1
                    boxes[idx_a][2] = box_b_name
                    boxes[idx_a][3] = box_b[3]
                    hit = True
            if not hit:
                boxes[idx_a][2] = uuid.uuid4()
            # show id on person head            
            cv2.putText(img, str(boxes[idx_a][2])[-4:].upper() , (box_a_xy.astype(int)) , cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2, cv2.LINE_AA )
        for person in history_4:
            xy = person[0]
            id = person[2]
            for h, person2 in enumerate(boxes):
                xy2 = person2[0]
                id2 = person2[2]
                last_event = person2[3]
                if id==id2:
                    from_zone_1 = cv2.pointPolygonTest( np.array(zone1).astype(int), xy , False) > 0
                    from_zone_2 = cv2.pointPolygonTest( np.array(zone2).astype(int), xy , False) > 0
                    to_zone_1 = cv2.pointPolygonTest( np.array(zone1).astype(int), xy2 , False) > 0
                    to_zone_2 = cv2.pointPolygonTest( np.array(zone2).astype(int), xy2 , False) > 0
                    if from_zone_1 and to_zone_2 and last_event != '1_to_2':
                        n_person_in += 1
                        last_event = '1_to_2'
                        boxes[h][3] = last_event
                    elif from_zone_2 and to_zone_1 and last_event != '2_to_1':
                        n_person_out += 1
                        last_event = '2_to_1'
                        boxes[h][3] = last_event
        cv2.putText(img, 'UP:%d'%n_person_out , (50,150) , cv2.FONT_HERSHEY_PLAIN, 4., (255,100,60), 2, cv2.LINE_AA )
        cv2.putText(img, 'DOWN :%d'%n_person_in ,  (50,200)  , cv2.FONT_HERSHEY_PLAIN, 4., (255,100,60), 2, cv2.LINE_AA )
        history_4 = history_3.copy()
        history_3 = history_2.copy()
        history_2 = history_1.copy()
        history_1 = boxes.copy()

        if bn > len(files)-20:
            img = img.astype(float) * (len(files)-bn)/20
            img = img.astype(np.uint8)

        bn += 1
        ic(bn)

        fname_body = os.path.split(fname)[-1]
        output_fname = os.path.join(output_folder, fname_body)        
        swift.imwrite(output_fname, img)
        #thumb = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), cv2.INTER_AREA)
        #cv2.imshow('',thumb)
        #cv2.waitKey(15)

    time.sleep(1.0)
    proc_load.join()
    swift.close()




if __name__=='__main__':
    inp_dir  = R"F:\TAIWAI\long sequence"
    outp_dir = R"result"
    if len(sys.argv) > 2:
        inp_dir = sys.argv[1]
        outp_dir = sys.argv[2]

    if not os.path.exists(outp_dir):
        os.mkdir(outp_dir)
    
    files = glob.glob(os.path.join(inp_dir, '*.jpg'))
    assert len(files) > 0
    files.sort()
    predict_track( files, outp_dir, None )

