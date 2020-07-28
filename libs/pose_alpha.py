import numpy as np
import cv2
import torch
import torch.utils.data as data
from .SPPE.src.main_fast_inference import InferenNet_fast
from .SPPE.src.utils.img import cropBox, im_to_torch
from .SPPE.src.utils.eval import getPrediction
from .alphapose_vis import vis_frame
from .SPPE.src.pPose_nms import pose_nms
from .base import run_time

class Mscoco(data.Dataset):
    def __init__(self, train=False, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = 320
        self.inputResW = 256
        self.outputResH = 80
        self.outputResW = 64
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

def crop_from_dets(img, boxes, inps, pt1, pt2, inputResH=320, inputResW=256):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img, upLeft, bottomRight, inputResH, inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2

class AlphaPose():
    '''
    Nose, LEye, REye, LEar, REar
    LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
    LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck(is calculated by LShoulder, RShoulder)
    kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
    kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
    connect points
    l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
    '''
    def __init__(self, model_path, is_fast=True):
        pose_dataset = Mscoco()
        if is_fast:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset, model_path)
        if torch.cuda.is_available():
            self.pose_model.cuda()
        self.pose_model.eval()
    @run_time
    def inference(self, img, boxes=None, scores=None):
        '''
        img like img = cv2.imread(img_name)
        '''
        timg = im_to_torch(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if boxes is None and scores is None:
            boxes = torch.FloatTensor(np.array([[0, 0, img.shape[1], img.shape[0]]]))
            scores = torch.FloatTensor(np.array([[1.0]]))
        elif isinstance(boxes, np.ndarray) and isinstance(scores, np.ndarray):
            boxes = torch.FloatTensor(boxes)
            scores = torch.FloatTensor(scores)
        else:
            return None
        inps = torch.zeros(len(boxes), 3, 320, 256)
        pt1 = torch.zeros(len(boxes), 2)
        pt2 = torch.zeros(len(boxes), 2)
        inps, pt1, pt2 = crop_from_dets(timg, boxes, inps, pt1, pt2)
        if torch.cuda.is_available():
            inps = inps.cuda()
            pt1 = pt1.cuda()
            pt2 = pt2.cuda()
        hm = self.pose_model(inps)
        preds_hm, preds_img, preds_scores = getPrediction(
                            hm, pt1, pt2, 320, 256, 80, 64)
        nresult = []
        slen = preds_scores.shape[0]
        for i in range(slen):
            kp_preds = preds_img[i,:,:]
            kp_scores = preds_scores[i,:]
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
            if torch.cuda.is_available():
                kp_preds = kp_preds.cpu()
                kp_scores = kp_scores.cpu()
            kp_preds = kp_preds.detach().numpy()
            kp_scores = kp_scores.detach().numpy()
            kp = np.hstack((kp_preds, kp_scores)).tolist()
            nresult.append(kp)
        return nresult


        # # nms will combine overlap box, e.g. boxes has 3 object, result has 1 object.
        # result = pose_nms(boxes, scores, preds_img, preds_scores)
        # # return result and visualize directly
        # # return result
        # nresult = []
        # for ro in result:
        #     kp_preds = ro['keypoints']
        #     kp_scores = ro['kp_score']
        #     proposal_score = ro['proposal_score']
        #     kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
        #     kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
        #     kp_preds = kp_preds.detach().numpy()
        #     kp_scores = kp_scores.detach().numpy()
        #     kp = np.hstack((kp_preds, kp_scores)).tolist()
        #     nresult.append(kp)
        # return nresult

if __name__ == '__main__':
    img_name = '/Users/guoxiaolu/work/tracking/libs/timg.jpeg'
    boxes = np.array([[109.1425,  30.8479, 316.1574, 729.1960],
        [319.7101,  51.3892, 546.6592, 682.3468],
        [ 41.2098, 116.4684,  67.1232, 204.5955],
        [237.4613,  97.4263, 295.3131, 223.8628],
        [ 48.2343, 114.2056,  73.1248, 180.0681],
        [ 56.9143, 109.6427,  76.1288, 172.7608],
        [ 63.3201, 113.2064,  80.4211, 170.6993],
        [ 40.6339, 122.2506,  63.6091, 175.0461]])
    scores = np.array([[0.9971],
        [0.9017],
        [0.7021],
        [0.6044],
        [0.4463],
        [0.2875],
        [0.0816],
        [0.0747]])
    model_path = '/Users/guoxiaolu/work/tracking/models/duc_se.pth'
    # img_name = '/Users/guoxiaolu/work/tracking/libs/person.jpg'
    model = AlphaPose(model_path)
    img = cv2.imread(img_name)
    # result = model.inference(img)
    result = model.inference(img, boxes, scores)

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
    # Draw keypoints
    for pose_pts in result:
        for i, pt in enumerate(pose_pts):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, p_color[i], -1)
        for i, lp in enumerate(l_pair):
            p0 = pose_pts[lp[0]]
            p1 = pose_pts[lp[1]]
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), line_color[i], 2)
    cv2.imwrite('./alphapose.jpg', img)
    # result = {
    #             'imgname': img_name,
    #             'result': result
    #         }
    # nimg = vis_frame(img, result)
    # cv2.imwrite('./alphapose.jpg', nimg)
