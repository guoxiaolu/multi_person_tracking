import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from tracker import Tracker
from libs.reid_dgnet import GetDis
from libs.base import run_time

def compute_iou(box1, box2, return_all=False):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积
 
    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    if not return_all:
        return iou
    else:
        return iou, area, s1, s2

def calc_area(box):
    return (box[3] - box[1]) * (box[2] - box[0])

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[:, 0]
    yg = g[:, 1]
    vg = g[:, 2]

    xd = d[:, 0]
    yd = d[:, 1]
    vd = d[:, 2]
    dx = xd - xg
    dy = yd - yg
    e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d) / 2 + np.spacing(1)) / 2
    if in_vis_thre is not None:
        ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
        e = e[ind]
    ious = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

class Tracking:
    def __init__(self, drate, ishape, pwc_opticalflow=None, nshape=None):
        # optical flow init of cpu
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.1,
                            minDistance = 7,
                            blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # optical flow init `of gpu
        self.pwc_opticalflow = pwc_opticalflow
        self.nshape = nshape
        self.drate = drate
        self.ishape = ishape
    
    @run_time
    def tracking(self, trackers, dboxes, features, appts, fid, mtracker_id,
             dist_th=0.1, iou_th=0.5, dist_resume_th=0.1, resume_times_th=10,
            area_th=2, iou_th_single=0.75, no_detection_frames_th=50, other_overlap_th=0.15, is_oks=True):
        ntrackers = trackers
        nmtracker_id = mtracker_id
        # Hungarian algorithm
        # valid trackers number
        vtn = 0
        # key is valid trackers sequence id, value is all trackers sequence id
        valid_id = {}
        for i, tracker in enumerate(trackers):
            if tracker.state == 0:
                continue
            valid_id[vtn] = i
            vtn += 1
        # valid tracker is existed
        if vtn > 0:
            dist_matrix = np.ones((len(dboxes), vtn))
            iou_matrix = np.ones((len(dboxes), vtn))
            oks_matrix = np.ones((len(dboxes), vtn))
            for i, dbox in enumerate(dboxes):
                feature = features[i]
                dvtn = 0
                for tracker in trackers:
                    if tracker.state == 0:
                        continue
                    iou = compute_iou(tracker.latest_box[:4], dbox[:4])
                    iou_matrix[i, dvtn] = iou
                    if is_oks:
                        # oks = oks_iou(np.array(tracker.pose_pts[:17]), np.array(appts[i][:17]),
                        #     calc_area(tracker.latest_box), calc_area(dbox))
                        oks = oks_iou(np.array(tracker.pose_pts[:17]), np.array(appts[i][:17]),
                            calc_area(tracker.latest_box), calc_area(dbox), in_vis_thre=0.05)
                        oks_matrix[i, dvtn] = oks
                    # calculate feature distance
                    distance = GetDis(tracker.mfeature, feature)
                    dist_matrix[i, dvtn] = distance
                    dvtn += 1
            # Hungarian algorithm, dist matrix
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            rind = row_ind.tolist()
            cind = col_ind.tolist()
            # Hungarian algorithm, iou matrix
            if is_oks:
                iou_matrix = oks_matrix
            row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
            # rind_iou = row_ind.tolist()
            cind_iou = col_ind.tolist()

            # # Hungarian algorithm, iou matrix
            # row_ind, col_ind = linear_sum_assignment(oks_matrix, maximize=True)
            # # rind_iou = row_ind.tolist()
            # cind_oks = col_ind.tolist()

            # determine which is dependent, which is not
            # which box has been updated
            dbox_markers = [0] * len(dboxes)
            # which tracker has been updated
            tracker_markers = [0] * len(trackers)
            # which valid tracker has been updated
            valid_tracker_markers = [0] * vtn
            for i, ri in enumerate(rind):
                ci = cind[i]
                distance = dist_matrix[ri, ci]
                # ri of rind and rind_iou is same
                ci_iou = cind_iou[i]
                iou = iou_matrix[ri, ci_iou]
                # if distance < dist_th:
                if ci == ci_iou and distance < dist_th * 1.5 and iou > iou_th / 2:
                    ti = valid_id[ci]
                    dbox = dboxes[ri]
                    feature = features[ri]
                    appt = appts[ri]
                    # dbox overlap with other tracker
                    is_other_overlap = False
                    ri_iou_list = iou_matrix[ri,:].tolist()
                    for idx, ri_iou in enumerate(ri_iou_list):
                        if idx == ci_iou:
                            continue
                        if ri_iou > other_overlap_th:
                            is_other_overlap = True
                            break
                    trackers[ti].update(dbox, fid, feature, appt, is_track=False, is_other_overlap=is_other_overlap)
                    dbox_markers[ri] = 1
                    tracker_markers[ti] = 1
                    valid_tracker_markers[ci] = 1
                    continue
                elif ci != ci_iou and iou > iou_th and dist_matrix[ri, ci_iou] < dist_th:
                    ti = valid_id[ci_iou]
                    dbox = dboxes[ri]
                    feature = features[ri]
                    appt = appts[ri]
                    # dbox overlap with other tracker
                    is_other_overlap = False
                    ri_iou_list = iou_matrix[ri,:].tolist()
                    for idx, ri_iou in enumerate(ri_iou_list):
                        if idx == ci_iou:
                            continue
                        if ri_iou > other_overlap_th:
                            is_other_overlap = True
                            break
                    trackers[ti].update(dbox, fid, feature, appt, is_track=False, is_other_overlap=is_other_overlap)
                    dbox_markers[ri] = 1
                    tracker_markers[ti] = 1
                    valid_tracker_markers[ci_iou] = 1
                    continue
                elif ci != ci_iou and distance < dist_th and iou_matrix[ri, ci] > iou_th:
                    ti = valid_id[ci]
                    dbox = dboxes[ri]
                    feature = features[ri]
                    appt = appts[ri]
                    # dbox overlap with other tracker
                    is_other_overlap = False
                    ri_iou_list = iou_matrix[ri,:].tolist()
                    for idx, ri_iou in enumerate(ri_iou_list):
                        if idx == ci:
                            continue
                        if ri_iou > other_overlap_th:
                            is_other_overlap = True
                            break
                    trackers[ti].update(dbox, fid, feature, appt, is_track=False, is_other_overlap=is_other_overlap)
                    dbox_markers[ri] = 1
                    tracker_markers[ti] = 1
                    valid_tracker_markers[ci] = 1
                    continue
                
            # add logic, try to match killed tracker
            ktn = 0
            # key is valid trackers sequence id, value is all trackers sequence id
            killed_id = {}
            for i, tracker in enumerate(trackers):
                if tracker.state != 0 or (fid - tracker.latest_fid) > max(50, self.drate * resume_times_th):
                # if tracker.state != 0:
                    continue
                killed_id[ktn] = i
                ktn += 1
            if dbox_markers.count(0) != 0 and ktn != 0:
                dist_matrix_resume = np.ones((dbox_markers.count(0), ktn))
                kdm = 0
                # key is unmatched dbox idx, value is dboxes idx
                unmatched_dbox_id = {}
                for i, dm in enumerate(dbox_markers):
                    if dm != 0:
                        continue
                    unmatched_dbox_id[kdm] = i
                    dbox = dboxes[i]
                    # try to match
                    feature = features[i]
                    dktn = 0
                    for j, tracker in enumerate(trackers):
                        if tracker.state != 0 or (fid - tracker.latest_fid) > max(50, self.drate * resume_times_th):
                        # if tracker.state != 0:
                            continue
                        # calculate feature distance
                        distance = GetDis(tracker.mfeature, feature)
                        dist_matrix_resume[kdm, dktn] = distance
                        dktn += 1
                    kdm += 1
                # Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(dist_matrix_resume)

                rind = row_ind.tolist()
                cind = col_ind.tolist()
                for i, ri in enumerate(rind):
                    ci = cind[i]
                    distance = dist_matrix_resume[ri, ci]
                    # similarity must be very high
                    if distance < dist_resume_th:
                        ti = killed_id[ci]
                        di = unmatched_dbox_id[ri]
                        dbox = dboxes[di]
                        feature = features[di]
                        appt = appts[di]

                        # area compare
                        darea = calc_area(dbox)
                        latest_box = trackers[ti].latest_box
                        larea = calc_area(latest_box)
                        # predict box compare
                        latest_fid = trackers[ti].latest_fid
                        delta_fid = fid - latest_fid
                        pred_velocity = trackers[ti].pred_velocity
                        vof = [pred_velocity[0] * delta_fid, pred_velocity[1] * delta_fid]
                        pbox = [int(latest_box[0] + vof[0]), int(latest_box[1] + vof[1]), 
                            int(latest_box[2] + vof[0]), int(latest_box[3] + vof[1])]
                        iou = compute_iou(dbox[:4], pbox)
                        # resume or not
                        if min(darea, larea) * area_th > max(darea, larea) and iou > 0.001:
                            trackers[ti].resume()
                            trackers[ti].update(dbox, fid, feature, appt, is_track=False)
                            dbox_markers[di] = 1
                            tracker_markers[ti] = 1
            
            # new tracker
            for i, dm in enumerate(dbox_markers):
                if dm != 0:
                    continue        
                # new tracker, using valid_tracker_markers info
                iou_list = iou_matrix[i,:].tolist()
                dist_list = dist_matrix[i,:].tolist()
                mark = True
                marked_vtm = {}
                vcid = None
                for cid, vtm in enumerate(valid_tracker_markers):
                    # tracker not matched before and iou is large
                    if vtm == 0 and (iou_list[cid] > iou_th or (iou_list[cid] > 0.01 and dist_list[cid] < dist_th / 2)):
                        mark = False
                        vcid = cid
                        break
                    # iou area / dbox or iou area / tracker.latest_box is high
                    if vtm == 0:
                        ti = valid_id[cid]
                        dbox = dboxes[i]
                        latest_box = trackers[ti].latest_box
                        iou, area, darea, lbarea = compute_iou(dbox[:4], latest_box[:4], True)
                        # if dist_list[cid] < dist_th and (area / darea > iou_th_single or area / lbarea > iou_th):
                        # seems overlap area in tracker cannot be too large, so using a smaller thresholld, as the detected box will be changed (part/full)
                        diou = area / darea
                        lbiou = area / lbarea
                        if diou > iou_th_single or lbiou > iou_th:
                            mark = False
                            # not creedy, otherwise mis-match
                            marked_vtm[cid] = diou + lbiou
                if not mark:
                    if len(marked_vtm) != 0:
                        vcid = sorted(marked_vtm.items(), key=lambda item: item[1], reverse=True)[0][0]
                    ti = valid_id[vcid]
                    dbox = dboxes[i]
                    feature = features[i]
                    appt = appts[i]
                    # dbox overlap with other tracker
                    is_other_overlap = False
                    ri_iou_list = iou_list
                    for idx, ri_iou in enumerate(ri_iou_list):
                        if idx == ti:
                            continue
                        if ri_iou > other_overlap_th:
                            is_other_overlap = True
                            break
                    trackers[ti].update(dbox, fid, feature, appt, is_track=False, is_other_overlap=is_other_overlap)
                    valid_tracker_markers[vcid] = 1
                else:
                    # new tracker
                    nmtracker_id += 1
                    nt = Tracker(nmtracker_id)
                    dbox = dboxes[i]
                    feature = features[i]
                    appt = appts[i]
                    nt.update(dbox, fid, feature, appt, is_track=False)
                    ntrackers.append(nt)
            # mark which tracker is not be detected
            for i, tm in enumerate(tracker_markers):
                if tm != 0 or ntrackers[i].state == 0:
                    continue
                ntrackers[i].update_no_detection_times()
                # ntrackers[i].update_no_detection_times(no_detection_frames_th // self.drate)
        else:
            # new tracker
            for i, dbox in enumerate(dboxes):
                nmtracker_id += 1
                nt = Tracker(nmtracker_id)
                feature = features[i]
                appt = appts[i]
                nt.update(dbox, fid, feature, appt, is_track=False)
                ntrackers.append(nt)
        return ntrackers, nmtracker_id
    
    @run_time
    def good_feature_track_cpu(self, frame, trackers, tracker_fpt_num_th=100):
        # extract from whole image
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # 先从整幅图中拿特征，也可从tracker中拿
        # pts = cv2.goodFeaturesToTrack(gray, mask = None, **self.feature_params)
        # pts = pts.tolist()
        # markers = [0] * len(pts)
        # for tracker in trackers:
        #     # all pts have been marked
        #     if sum(markers) == len(markers):
        #         break
        #     # update
        #     box = tracker.latest_box
        #     # empty
        #     tracker.empty_feature_pt()
        #     # judge pt in box or not
        #     for i, pt in enumerate(pts):
        #         if len(tracker.feature_pts) > tracker_fpt_num_th:
        #             break
        #         if markers[i] == 1:
        #         # if markers[i] == 1:
        #             continue
        #         pt = pt[0]
        #         if pt[0] < box[2] and pt[0] > box[0] and pt[1] < box[3] and pt[1] > box[1]:
        #             tracker.update_feature_pt(pt)
        #             markers[i] = 1
        # extract from tracker
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for tracker in trackers:
            if tracker.state == 0:
                continue
            box = tracker.latest_box
            bbox = [max(0, min(self.ishape[1], box[0])), max(0, min(self.ishape[0], box[1])),
                    max(0, min(self.ishape[1], box[2])), max(0, min(self.ishape[0], box[3])), box[4]]
            roi = gray[bbox[1]:bbox[3] - 1, bbox[0]:bbox[2] - 1]
            pts = cv2.goodFeaturesToTrack(roi, mask = None, **self.feature_params)
            if pts is None:
                continue
            tracker.empty_feature_pt()
            pts = pts.tolist()
            for pt in pts:
                tracker.update_feature_pt(pt[0])
    @run_time
    def optical_flow_cpu(self, old_gray, frame_gray, trackers, fid,
         fpt_num_th=36, is_visualize=False, is_history_conf=True, 
         velocity_sudden_change_th=2, velocity_sudden_change_norm_th=5, velocity_sudden_change_angle_th=0.8):
        if len(trackers) == 0:
            return None
        # extract all feature pts in trackers
        p0 = []
        trackerid_p0 = []
        for tracker in trackers:
            if tracker.state == 0:
                continue
            p0 += tracker.feature_pts
            pose_pts = np.array(tracker.pose_pts)[:,:2].tolist()
            p0 += pose_pts
            trackerid_p0.append([tracker.tracker_id, len(p0)])
        # p0为空，加异常判断!!!
        # calculate optical flow
        if len(p0) == 0:
            return None
        ap0 = np.expand_dims(np.array(p0), 1).astype('float32')
        ap1, ast, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, ap0, None, **self.lk_params)
        p1 = np.squeeze(ap1, 1).tolist()
        st = ast.squeeze().tolist()
        if not isinstance(st, list):
            st = [st]

        init_pos = 0
        for value in trackerid_p0:
            trackerid, p0_pos = value
            tracker_st = np.array(st[init_pos:p0_pos])
            tracker_pt0 = np.array(p0[init_pos:p0_pos])[tracker_st == 1]
            tracker_pt1 = np.array(p1[init_pos:p0_pos])[tracker_st == 1]
            init_pos = p0_pos

            pt_delta_valid = tracker_pt1 - tracker_pt0
            if sum(tracker_st) < fpt_num_th:
                vector_optical_flow = None
            else: 
                mean_val = np.mean(pt_delta_valid, axis=0)
                vector_optical_flow = tuple(list(map(int, mean_val)))

            # find this tracker by trackerid, then update predicted box
            for tracker in trackers:
                if tracker.tracker_id != trackerid:
                    continue
                latest_box = tracker.latest_box
                pv = tracker.pred_velocity
                dv = tracker.velocity
                if vector_optical_flow is None:
                    # pvof = (0, 0)
                    pvof = pv
                else:
                    # prevent velocity sudden change
                    pv_norm = np.linalg.norm(np.array(pv))
                    vof_norm = np.linalg.norm(np.array(vector_optical_flow))
                    cosangle = np.array(pv).dot(np.array(vector_optical_flow)) / (pv_norm * vof_norm)
                    angle = np.arccos(cosangle)
                    if (pv_norm > velocity_sudden_change_norm_th or vof_norm > velocity_sudden_change_norm_th) and \
                        (vof_norm > pv_norm * velocity_sudden_change_th or \
                            (angle > velocity_sudden_change_angle_th or angle < 0 - velocity_sudden_change_angle_th)):
                        pvof = pv
                    else:
                        if np.std(pt_delta_valid) > 2:
                            pvof = pv
                        else:
                            # prevent velocity sudden change
                            dv_norm = np.linalg.norm(np.array(dv))
                            cosangle = np.array(dv).dot(np.array(vector_optical_flow)) / (dv_norm * vof_norm)
                            angle = np.arccos(cosangle)
                            if (dv_norm > velocity_sudden_change_norm_th or vof_norm > velocity_sudden_change_norm_th) and \
                                (vof_norm > dv_norm * velocity_sudden_change_th or \
                                    (angle > velocity_sudden_change_angle_th or angle < 0 - velocity_sudden_change_angle_th)):
                                pvof = pv
                            else:
                                pvof = vector_optical_flow
                            # pvof = vector_optical_flow
                nbox = [int(latest_box[0] + pvof[0]), int(latest_box[1] + pvof[1]),
                        int(latest_box[2] + pvof[0]), int(latest_box[3] + pvof[1]), 0]
                is_boundary = False
                if nbox[3] >= self.ishape[0] or nbox[2] >= self.ishape[1] or nbox[0] <= 0 or nbox[1] <= 0:
                    is_boundary = True
                # pose pts
                pose_p0 = p0[p0_pos-18:p0_pos]
                pose_p1 = p1[p0_pos-18:p0_pos]
                # add confidence as 0
                # pose_p1 = np.hstack((np.array(pose_p1), np.zeros((18,1)))).tolist()
                npose = []
                pose_pts = tracker.pose_pts
                for i, pp1 in enumerate(pose_p1):
                    if pp1[0] < nbox[0] or pp1[0] > nbox[2] or pp1[1] < nbox[1] or pp1[1] > nbox[3]:
                        pp0 = pose_p0[i]
                        if is_history_conf:
                            npose.append([pp0[0] + pvof[0], pp0[1] + pvof[1], pose_pts[i][2]])
                        else:
                            npose.append([pp0[0] + pvof[0], pp0[1] + pvof[1], 0])
                    else:
                        if is_history_conf:
                            npose.append([pp1[0] , pp1[1], pose_pts[i][2]])
                        else:    
                            npose.append([pp1[0] , pp1[1], 0])
                if is_visualize:
                    tracker.update(nbox, fid, pose_pts=npose, optical_flow=np.hstack((tracker_pt0, tracker_pt1)), ishape=self.ishape, is_boundary=is_boundary)
                else:
                    tracker.update(nbox, fid, pose_pts=npose, ishape=self.ishape, is_boundary=is_boundary)
                # update pose points
                # tracker.update_pose_pts(pose_p1)
                # update feature points, 速度太慢
                nfpts = tracker_pt1.tolist()
                nfpts = [pt for pt in nfpts if pt[0] < nbox[2] and pt[0] > nbox[0] and pt[1] < nbox[3] and pt[1] > nbox[1]]
                tracker.update_feature_pt_list(nfpts)
                break
    @run_time
    def optical_flow_gpu(self, old, frame, trackers, fid, is_history_conf=True, is_pose_filter=True, is_visualize=False, 
                    velocity_sudden_change_th=2, velocity_sudden_change_norm_th=5, velocity_sudden_change_angle_th=0.8):
        ishape = old.shape
        nold = np.zeros(self.nshape, np.uint8)
        nframe = np.zeros(self.nshape, np.uint8)
        nold[:ishape[0],:ishape[1],:] = old
        nframe[:ishape[0],:ishape[1],:] = frame
        oflow = self.pwc_opticalflow.Pre(nold, nframe)
        scale = self.nshape[0] // oflow.shape[0]
        noflow = cv2.resize(oflow, (self.nshape[1], self.nshape[0])) * scale
        
        for tracker in trackers:
            if tracker.state == 0:
                continue
            latest_box = tracker.latest_box
            pose_pts = tracker.pose_pts
            npose = []
            # pose_pts
            for ppt in pose_pts:
                pi = int(min(max(ppt[1], 0), ishape[0] - 1))
                pj = int(min(max(ppt[0], 0), ishape[1] - 1))
                tvof = tuple(map(int, noflow[pi, pj].tolist()))
                if is_history_conf:
                    nppt = [ppt[0] + tvof[0], ppt[1] + tvof[1], ppt[2]]
                else:
                    nppt = [ppt[0] + tvof[0], ppt[1] + tvof[1], 0]
                npose.append(nppt)
            if is_pose_filter:
                ltpi = min(min(npose[5][0], npose[6][0]), min(npose[11][0], npose[12][0]))
                ltpi = int(min(max(ltpi, 0), ishape[0] - 1))
                ltpj = min(min(npose[5][1], npose[6][1]), min(npose[11][1], npose[12][1]))
                ltpj = int(min(max(ltpj, 0), ishape[1] - 1))
                rdpi = max(max(npose[5][0], npose[6][0]), max(npose[11][0], npose[12][0]))
                rdpi = int(min(max(rdpi, 0), ishape[0] - 1))
                rdpj = max(max(npose[5][1], npose[6][1]), max(npose[11][1], npose[12][1]))
                rdpj = int(min(max(rdpj, 0), ishape[1] - 1))
                tmp = noflow[ltpi:rdpi, ltpj:rdpj]
            else:
                ltpi = min(max(latest_box[1], 0), ishape[0] - 1)
                ltpj = min(max(latest_box[0], 0), ishape[1] - 1)
                rdpi = min(max(latest_box[3], 0), ishape[0] - 1)
                rdpj = min(max(latest_box[2], 0), ishape[1] - 1)
                tmp = noflow[ltpi:rdpi, ltpj:rdpj]
            
            pv = tracker.pred_velocity
            dv = tracker.velocity
            if len(tmp.tolist()) == 0:
                pvof = pv
            else:
                vof = tuple(map(int, np.mean(np.mean(tmp, axis=0), axis=0).tolist()))
                # prevent velocity sudden change
                pv_norm = np.linalg.norm(np.array(pv))
                vof_norm = np.linalg.norm(np.array(vof))
                cosangle = np.array(pv).dot(np.array(vof)) / (pv_norm * vof_norm)
                angle = np.arccos(cosangle)
                if (pv_norm > velocity_sudden_change_norm_th or vof_norm > velocity_sudden_change_norm_th) and \
                    (vof_norm > pv_norm * velocity_sudden_change_th or \
                        (angle > velocity_sudden_change_angle_th or angle < 0 - velocity_sudden_change_angle_th)):
                    pvof = pv
                else:
                    # prevent velocity sudden change
                    dv_norm = np.linalg.norm(np.array(dv))
                    cosangle = np.array(dv).dot(np.array(vof)) / (dv_norm * vof_norm)
                    angle = np.arccos(cosangle)
                    if (dv_norm > velocity_sudden_change_norm_th or vof_norm > velocity_sudden_change_norm_th) and \
                        (vof_norm > dv_norm * velocity_sudden_change_th or \
                            (angle > velocity_sudden_change_angle_th or angle < 0 - velocity_sudden_change_angle_th)):
                        pvof = pv
                    else:
                        pvof = vof
            nbox = [int(latest_box[0] + pvof[0]), int(latest_box[1] + pvof[1]), 
                int(latest_box[2] + pvof[0]), int(latest_box[3] + pvof[1]), 0]
            if is_visualize:
                of = np.hstack((np.array(pose_pts), np.array(npose)))
                tracker.update(nbox, fid, pose_pts=npose, optical_flow=of, ishape=self.ishape)
            else:
                tracker.update(nbox, fid, pose_pts=npose, ishape=self.ishape)
            
                
            

