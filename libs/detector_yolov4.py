import torch
import cv2
from .tool_yolov4.darknet2pytorch import Darknet
from .tool_yolov4.utils import do_detect
from .base import run_time

def model_init(cname, mname):
    global model
    global use_cuda
    model = Darknet(cname)
    model.print_network()
    model.load_weights(mname)
    print('Loading weights from %s... Done!' % (mname))
    use_cuda = 0
    if torch.cuda.is_available():
        use_cuda = 1
        model.cuda()

@run_time
def inference(img, conf_thresh=0.5, nms_thresh=0.4, wh_th=32):
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    boxes = do_detect(model, sized, conf_thresh, nms_thresh, use_cuda)
    rboxes = []
    # currently, only support person 
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) < 7:
            print('box length is wrong')
            return None
        cls_id = box[6]
        if cls_id != 0:
            continue
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        if x2 - x1 < wh_th / 4 or y2 - y1 < wh_th:
            continue
        cls_conf = box[5]
        rbox = [x1, y1, x2, y2, cls_conf]
        rboxes.append(rbox)
    return rboxes

if __name__ == '__main__':
    cname = '/Users/guoxiaolu/work/tracking/models/yolov4.cfg'
    mname = '/Users/guoxiaolu/work/tracking/models/yolov4.weights'
    iname = '/Users/guoxiaolu/work/tracking/libs/timg.jpeg'
    img = cv2.imread(iname)
    model_init(cname, mname)
    boxes = inference(img)
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    cv2.imshow('frame',img)
    cv2.waitKey()