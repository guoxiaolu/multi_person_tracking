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

@run_time
def test_video(fname, config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name, \
            ap_name, pwc_name=None, drate=5, is_visualize=False, downsample_rate=1):
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        print('read video file failed')
    
    # frame count
    fid = 0
    # max_tracker number
    mtracker_id = 0
    trackers = []
    # person detection of 1st frame
    res, img = cap.read()
    fid += 1
    if res == False:
        print('read over')
        return
    # REID init
    reid = ReID(config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name)
    # pose init
    ap = AlphaPose(ap_name)
    dboxes = detector_inference(img)
    if len(dboxes) != 0:
        npdboxes = np.array(dboxes)
        appts = ap.inference(img, npdboxes[:,0:4], npdboxes[:,4:5])
    else:
        appts = []
    for i, dbox in enumerate(dboxes):
        mtracker_id += 1
        nt = Tracker(mtracker_id)
        # reid feature
        # 未做边界处理!!!!
        roi = img[dbox[1]:dbox[3], dbox[0]:dbox[2]]
        feature = reid.GetFeature(roi)
        appt = appts[i]
        nt.update(dbox, fid, feature, appt)
        trackers.append(nt)
    
    ishape = img.shape
    if pwc_name is None:
        # Tracking init, optical flow init
        tracking = Tracking(drate, ishape)
        # when detection, update optical flow features
        tracking.good_feature_track_cpu(img, trackers)
    else:
        # gpu, pwcnet init
        nshape = ((ishape[0] // 64 + int(ishape[0] % 64 > 0)) * 64, (ishape[1] // 64 + int(ishape[1] % 64 > 0)) * 64, ishape[2])
        pwc_opticalflow = Pwcnet(pwc_name, w=nshape[1], h=nshape[0])

        # Tracking init, optical flow init
        tracking = Tracking(drate, ishape, pwc_opticalflow, nshape)

    # save latest detection frame
    latest_detection_frame = img.copy()
    last_frame = img.copy()

    basename = os.path.basename(fname)
    extnames = os.path.splitext(basename)
    if is_visualize:
        fourcc = cv2.VideoWriter_fourcc(*'mpeg')
        vname = os.path.join('./', extnames[0] + '.mp4')
        out = cv2.VideoWriter(vname, fourcc, 25, (img.shape[1] // downsample_rate, img.shape[0] // downsample_rate))
        # Create some random colors
        colors = np.random.randint(0,255,(500,3)).tolist()
        visualize(img.copy(), out, trackers, fid, colors, downsample_rate)
    
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total frame number: %d'%(frame_num))
    while True:
        res, img = cap.read()
        fid += 1
        # if fid > 10:
        #     break
        if res==False:
            print('read over')
            break
        print('fid: %d'%(fid))
        # detection
        if fid % drate == 0:
            dboxes = detector_inference(img)
            features = []
            for dbox in dboxes:
                roi = img[dbox[1]:dbox[3], dbox[0]:dbox[2]]
                feature = reid.GetFeature(roi)
                features.append(feature)
            if pwc_name is None:
                # tracking, optical flow
                old_gray = cv2.cvtColor(latest_detection_frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                tracking.optical_flow_cpu(old_gray, frame_gray, trackers, fid, is_visualize=is_visualize)
            else:
                tracking.optical_flow_gpu(last_frame, img, trackers, fid, is_visualize=is_visualize)
            print('tracking')
            # if is_visualize:
            #     visualize(img.copy(), out, trackers, fid, colors)
            
            if len(dboxes) != 0:
                npdboxes = np.array(dboxes)
                appts = ap.inference(img, npdboxes[:,0:4], npdboxes[:,4:5])
            else:
                appts = []
            # tracking
            trackers, mtracker_id = tracking.tracking(trackers, dboxes, features, appts, fid, mtracker_id)
            if pwc_name is None:
                # when detection, update optical flow features
                tracking.good_feature_track_cpu(img, trackers)
            # save latest detection frame
            latest_detection_frame = img.copy()
            print('detection and matching:%d'%len(trackers))
        else:
            if pwc_name is None:
                # tracking, optical flow
                old_gray = cv2.cvtColor(latest_detection_frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                tracking.optical_flow_cpu(old_gray, frame_gray, trackers, fid, is_visualize=is_visualize)
            else:
                tracking.optical_flow_gpu(last_frame, img, trackers, fid, is_visualize=is_visualize)
            print('tracking')
        last_frame = img.copy()

        if is_visualize:
            visualize(img.copy(), out, trackers, fid, colors, downsample_rate)
    
    if is_visualize:
        out.release()
    cap.release()

    # output json
    jname = os.path.join('./', extnames[0] + '.json')
    wf = open(jname, 'w')
    jdict = {}
    for tracker in trackers:
        history_boxes = tracker.history_boxes
        keys = history_boxes.keys()
        values = history_boxes.values()
        nvalues = [list(map(float, v)) for v in values]
        nhb = dict(zip(keys, nvalues))
        jdict[tracker.tracker_id] = {"state": tracker.state, "latest_fid": tracker.latest_fid,
            "latest_box": list(map(float, tracker.latest_box)), "history_cpts":tracker.history_cpts,
             "history_boxes":nhb, "mfeature": tracker.mfeature.tolist()[0]}
    json.dump(jdict, wf)
    wf.close()

@run_time
def visualize(frame, out, trackers, fid, colors, downsample_rate=1):
    nframe = frame.copy()
    print(len(trackers))
    # pose points
    l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
    p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                (77,222,255), (255,156,127), 
                (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
    for tracker in trackers:
        if tracker.latest_fid != fid or tracker.state == 0:
            continue
        history_cpts = tracker.history_cpts
        tracker_id = tracker.tracker_id
        color = colors[tracker_id]
        latest_box = tracker.latest_box
        cv2.rectangle(nframe, (latest_box[0], latest_box[1]), (latest_box[2], latest_box[3]), color, 2 * downsample_rate)
        cv2.putText(nframe,'%d'%(tracker_id),(latest_box[0], latest_box[1]),cv2.FONT_HERSHEY_COMPLEX,1,color,2 * downsample_rate)
        if len(history_cpts) == 0:
            break

        cv2.circle(nframe, history_cpts[fid], 5,color, -1)
        for i in range(1, fid):
            if i in history_cpts and i + 1 in history_cpts:
                cv2.line(nframe, history_cpts[i], history_cpts[i + 1], color, 2)

        # pose pts
        # Draw keypoints
        pose_pts = tracker.pose_pts
        for i, pt in enumerate(pose_pts):
            if pt[2] < 0.05:
                continue
            cv2.circle(nframe, (int(pt[0]), int(pt[1])), 2, p_color[i], -1)
        for i, lp in enumerate(l_pair):
            p0 = pose_pts[lp[0]]
            p1 = pose_pts[lp[1]]
            if p0[2] < 0.05 or p1[2] < 0.05:
                continue
            cv2.line(nframe, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), line_color[i], 2)
            
        # optical flow
    #    of = tracker.optical_flow
    #    if of is not None:
    #        of = of.tolist()
    #        for pt in of:
    #            cv2.line(nframe, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color, 1)
    # if fid >= 1:
    #     cv2.imshow('frame', nframe)
    #     cv2.waitKey()
    if downsample_rate > 1:
        wnframe = cv2.resize(nframe, (nframe.shape[1] // downsample_rate, nframe.shape[0] // downsample_rate))
        out.write(wnframe)
    else:
        out.write(nframe)

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
    detector_init(dcname, dmname)
    test_video(fname, config_name, checkpoint_gen, checkpoint_id, trainer_pth_name, trainer_config_name, ap_name, drate=1, is_visualize=True, downsample_rate=1)
