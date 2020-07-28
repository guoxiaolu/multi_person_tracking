import os
import cv2
import numpy as np
import json
from tracker import Tracker
from tracking import Tracking
from libs.detector_yolov4 import model_init as detector_init, inference as detector_inference
from libs.reid_dgnet import ReID
from libs.pose_alpha import AlphaPose
from libs.base import run_time
import torch
if torch.cuda.is_available():
    from libs.pwc_opticalflow import Pwcnet

class MultiObjectTracking:
    def __init__(self, dcname, dmname, config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name, ap_name, pwc_name, drate, ishape):
        detector_init(dcname, dmname)
        # REID init
        self.reid = ReID(config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name)
        # pose init
        self.ap = AlphaPose(ap_name)
        self.drate = drate
        self.pwc_name = pwc_name
        # max_tracker number
        self.mtracker_id = 0
        self.trackers = []
        # optical flow
        if pwc_name is None:
            # Tracking init, optical flow init
            self.tracking = Tracking(self.drate, ishape)
        else:
            nshape = ((ishape[0] // 64 + int(ishape[0] % 64 > 0)) * 64, (ishape[1] // 64 + int(ishape[1] % 64 > 0)) * 64, ishape[2])
            pwc_opticalflow = Pwcnet(self.pwc_name, w=nshape[1], h=nshape[0])
            # Tracking init, optical flow init
            self.tracking = Tracking(self.drate, ishape, pwc_opticalflow, nshape)
        self.latest_detection_frame = None
        self.last_frame = None

    def FeedFirst(self, img):
        dboxes = detector_inference(img)
         # pose
        if len(dboxes) != 0:
            npdboxes = np.array(dboxes)
            appts = self.ap.inference(img, npdboxes[:,0:4], npdboxes[:,4:5])
        else:
            appts = []
        for i, dbox in enumerate(dboxes):
            self.mtracker_id += 1
            nt = Tracker(self.mtracker_id)
            # reid feature
            # 未做边界处理!!!!
            roi = img[dbox[1]:dbox[3], dbox[0]:dbox[2]]
            feature = self.reid.GetFeature(roi)
            appt = appts[i]
            nt.update(dbox, 1, feature, appt)
            self.trackers.append(nt)
        # optical flow
        if self.pwc_name is None:
            # when detection, update optical flow features
            self.tracking.good_feature_track_cpu(img, self.trackers)

        # save latest detection frame
        self.latest_detection_frame = img.copy()
        self.last_frame = img.copy()
        dict_data={}
        for tracker in self.trackers:
            if tracker.state == 0:
                continue
            dtracker = {}
            dtracker['bbox_body'] = list(map(float, tracker.latest_box))
            dtracker['pose'] = tracker.pose_pts
            dtracker['fea_body'] = tracker.mfeature.tolist()[0]
            dict_data[tracker.tracker_id] = dtracker
        return dict_data

    def Feed(self, img, fid):
        # detection
        if fid % self.drate == 0:
            dboxes = detector_inference(img)
            features = []
            for dbox in dboxes:
                roi = img[dbox[1]:dbox[3], dbox[0]:dbox[2]]
                feature = self.reid.GetFeature(roi)
                features.append(feature)
            if self.pwc_name is None:
                # tracking, optical flow
                old_gray = cv2.cvtColor(self.latest_detection_frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.tracking.optical_flow_cpu(old_gray, frame_gray, self.trackers, fid)
            else:
                self.tracking.optical_flow_gpu(self.last_frame, img, self.trackers, fid)
            print('tracking')
            if len(dboxes) != 0:
                npdboxes = np.array(dboxes)
                appts = self.ap.inference(img, npdboxes[:,0:4], npdboxes[:,4:5])
            else:
                appts = []
            # tracking
            trackers, mtracker_id = self.tracking.tracking(self.trackers, dboxes, features, appts, fid, self.mtracker_id)
            self.mtracker_id = mtracker_id
            if self.pwc_name is None:
                # when detection, update optical flow features
                self.tracking.good_feature_track_cpu(img, trackers)
            # save latest detection frame
            self.latest_detection_frame = img.copy()
            print('detection and matching:%d'%len(self.trackers))
        else:
            if self.pwc_name is None:
                # tracking, optical flow
                old_gray = cv2.cvtColor(self.latest_detection_frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.tracking.optical_flow_cpu(old_gray, frame_gray, self.trackers, fid)
            else:
                self.tracking.optical_flow_gpu(self.last_frame, img, self.trackers, fid)
            print('tracking')
        self.last_frame = img.copy()
        dict_data={}
        for tracker in self.trackers:
            if tracker.state == 0:
                continue
            dtracker = {}
            dtracker['bbox_body'] = list(map(float, tracker.latest_box))
            dtracker['pose'] = tracker.pose_pts
            dtracker['fea_body'] = tracker.mfeature.tolist()[0]
            dict_data[tracker.tracker_id] = dtracker
        return dict_data
    def DisplayRes(self,img,frame_data):
        img_dis=img.clone()
        return img_dis

@run_time
def test_video(fname, dcname, dmname, config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name, ap_name, pwc_name=None, drate=5):
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        print('read video file failed')
    
    # frame count
    fid = 0
    # person detection of 1st frame
    res, img = cap.read()
    fid += 1
    if res == False:
        print('read over')
        return
    jsondata={}
    ishape = img.shape
    mot = MultiObjectTracking(dcname, dmname, config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name, ap_name, pwc_name, drate, ishape)
    dict_data = mot.FeedFirst(img)
    jsondata[fid] = dict_data

    while True:
        res, img = cap.read()
        fid += 1
        # if fid > 10:
        #     break
        if res==False:
            print('read over')
            break
        print('fid: %d'%(fid))
        dict_data = mot.Feed(img, fid)
        jsondata[fid] = dict_data
    cap.release()

if __name__ == '__main__':
    fname = '/mnt/sda/guoxiaolu/testvideo/a2.mp4'
    # yolov4
    dcname = '/mnt/sda/guoxiaolu/qdh_mot/models/yolov4.cfg'
    dmname = '/mnt/sda/guoxiaolu/qdh_mot/models/yolov4.weights'
    # reid
    checkpoint_gen = '/mnt/sda/guoxiaolu/qdh_mot/models/gen_00100000.pt'
    checkpoint_id = '/mnt/sda/guoxiaolu/qdh_mot/models/id_00100000.pt'
    config_name = '/mnt/sda/guoxiaolu/qdh_mot/models/config.yaml'
    trainer_pth_name = '/mnt/sda/guoxiaolu/qdh_mot/models/net_last.pth'
    trainer_config_name = '/mnt/sda/guoxiaolu/qdh_mot/models/opts.yaml'
    # pose
    ap_name = '/mnt/sda/guoxiaolu/qdh_mot/models/duc_se.pth'
    # pwc-net, only gpu
    pwc_name = '/mnt/sda/guoxiaolu/qdh_mot/models/pwcnet.pth'
    test_video(fname, dcname, dmname, config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name, ap_name, drate=1)