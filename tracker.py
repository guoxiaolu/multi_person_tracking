
class Tracker:
    def __init__(self, tracker_id):
        self.latest_box = []
        self.history_boxes = {}
        self.latest_fid = 0
        self.history_cpts = {}
        self.history_velocity = {}
        self.tracker_id = tracker_id
        self.mfeature = None
        self.state = 1
        self.feature_pts = []
        self.no_detection_times = 0
        self.pose_pts = []
        self.pred_velocity = [0, 0]
        # this velocity is credible
        self.velocity = [0, 0]
        # for debug
        self.optical_flow = None
    def update(self, box, cur_frame_no, feature=None, pose_pts=None, optical_flow=None,
             is_track=True, ishape=None, is_boundary=False, is_other_overlap=False):
        if ishape is None:
            self.latest_box = box
        else:
            nbox = [max(0, min(ishape[1], box[0])), max(0, min(ishape[0], box[1])),
                    max(0, min(ishape[1], box[2])), max(0, min(ishape[0], box[3])), box[4]]
            self.latest_box = nbox
            if nbox[3] == nbox[1] or nbox[2] == nbox[0]:
                self.state = 0
        self.history_boxes[cur_frame_no] = box
        self.latest_fid = cur_frame_no
        self.history_cpts[cur_frame_no] = ((box[2] + box[0]) // 2, (box[3] + box[1]) // 2)
        if is_track:
            # when tracking, using pred_velocity (e.g. optical flow) to calculate velocity
            # if self.no_detection_times != 0:
            #     # keep same
            #     pass
            # elif cur_frame_no - 1 in self.history_cpts:
            # box in boundary
            if is_boundary:
                # keep same
                pass
            else:
                if cur_frame_no - 1 in self.history_cpts:
                    self.pred_velocity = [self.history_cpts[cur_frame_no][0] - self.history_cpts[cur_frame_no - 1][0],
                                            self.history_cpts[cur_frame_no][1] - self.history_cpts[cur_frame_no - 1][1]]
                else:
                    self.pred_velocity = [0, 0]
        else:
            # when matching, using detected box to calculate velocity
            # using multi-frames to prevent velocity sudden change
            last_detection_fid = []
            for tfid in range(self.latest_fid - 1, 0, -1):
                if tfid not in self.history_boxes:
                    break
                if self.history_boxes[tfid][-1] > 0.1:
                    last_detection_fid.append(tfid)
                    if len(last_detection_fid) >= 3:
                        break
            if len(last_detection_fid) != 0:
                # bbox will change, so use bbox to calculate pred_velocity
                # unvalid condition, bbox area changes a lot
                lfd = last_detection_fid[0]
                cur_frame_no_width = self.history_boxes[cur_frame_no][2] - self.history_boxes[cur_frame_no][0]
                last_detection_fid_width = self.history_boxes[lfd][2] - self.history_boxes[lfd][0]
                delta_width = abs(cur_frame_no_width - last_detection_fid_width)
                cur_frame_no_height = self.history_boxes[cur_frame_no][3] - self.history_boxes[cur_frame_no][1]
                last_detection_fid_height = self.history_boxes[lfd][3] - self.history_boxes[lfd][1]
                delta_height = abs(cur_frame_no_height - last_detection_fid_height)
                if (delta_height + delta_width) / (cur_frame_no_height + cur_frame_no_width) < 0.05:
                    pred_velocity = [(self.history_cpts[cur_frame_no][0] - self.history_cpts[lfd][0]) / (cur_frame_no - lfd),
                                        (self.history_cpts[cur_frame_no][1] - self.history_cpts[lfd][1]) / (cur_frame_no - lfd)]
                    for ldf in last_detection_fid:
                        pred_velocity = [pred_velocity[0] + self.history_velocity[ldf][0], pred_velocity[1] + self.history_velocity[ldf][1]]
                    pred_velocity = [pred_velocity[0] / (len(last_detection_fid) + 1), pred_velocity[1] / (len(last_detection_fid) + 1)]
                    self.pred_velocity = pred_velocity
                self.velocity = self.pred_velocity
        self.history_velocity[cur_frame_no] = self.pred_velocity

        # feature is None: tracking; feature is not None: detection
        if feature is not None:
            if not is_other_overlap:
                self.mfeature = feature
            else:
                self.mfeature = (self.mfeature + feature) / 2
            self.no_detection_times = 0
        if pose_pts is not None:
            self.pose_pts = pose_pts
        if optical_flow is not None:
            self.optical_flow = optical_flow
    def update_feature_pt(self, fpts):
        self.feature_pts.append(fpts)
    def update_feature_pt_list(self, fpts):
        self.feature_pts = fpts
    def empty_feature_pt(self):
        self.feature_pts = []
    def update_no_detection_times(self, no_detection_times_th=10):
        # no detection times > threshold -> tracker killed
        self.no_detection_times += 1
        if self.no_detection_times >= no_detection_times_th:
            self.state = 0
    def resume(self):
        self.state = 1