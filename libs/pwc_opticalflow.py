import cv2
import PIL
import torch
import numpy as np
import math
#from lib.OpticalFlow.PwcNet.run import Network
from .pwcnet.model import Network

class Pwcnet():
    def __init__(self, model_path, w=None, h=None):
        
        self.moduleNetwork = Network(model_path).cuda().eval()
        if w is None and h is None:
            scale=4
            self.w=64*4*scale
            self.h=64*3*scale
        else:
            self.w = w
            self.h = h
    def Preprocess(self,img_bgr):
        img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        img_rgb=img_rgb.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        img_tensor=torch.FloatTensor(img_rgb)
        intWidth=img_tensor.size(2)
        intHeight=img_tensor.size(1)        
        img_tensor=img_tensor.cuda().view(1, 3, intHeight, intWidth)
        return img_tensor
        
    def Pre(self,img0,img1):
        img1=cv2.resize(img1,(self.w,self.h))
        img0=cv2.resize(img0,(self.w,self.h))
        intHeight, intWidth=img0.shape[0],img0.shape[1]
        img0_t=self.Preprocess(img0)
        img1_t=self.Preprocess(img1)
        out=self.moduleNetwork(img0_t,img1_t)
        #tensorFlow = 20.0 * torch.nn.functional.interpolate(input=out, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
        tensorFlow = 20.0*out
        t_op=tensorFlow[0, :, :, :].cpu().numpy()
        res=t_op.transpose((1,2,0))
        return res

if __name__ == '__main__':
    # pwcnet output has been downsampled
    model_path = '/mnt/sda/guoxiaolu/qdh_mot/models/pwcnet.pth'

    # # two images
    # fn_0='/mnt/sda/guoxiaolu/qdh_mot/libs/first.png'
    # fn_1='/mnt/sda/guoxiaolu/qdh_mot/libs/second.png'
    # img0=cv2.imread(fn_0)
    # img1=cv2.imread(fn_1)

    # ishape = img0.shape
    # nshape = ((ishape[0] // 64 + int(ishape[0] % 64 > 0)) * 64, (ishape[1] // 64 + int(ishape[1] % 64 > 0)) * 64, ishape[2])
    # model = Pwcnet(model_path, w=nshape[1], h=nshape[0])
    # nimg0 = np.zeros(nshape, np.uint8)
    # nimg1 = np.zeros(nshape, np.uint8)
    # nimg0[:ishape[0],:ishape[1],:] = img0
    # nimg1[:ishape[0],:ishape[1],:] = img1
    # oflow = model.Pre(nimg0, nimg1)
    # scale = nshape[0] // oflow.shape[0]
    # noflow = cv2.resize(oflow, (nshape[1], nshape[0])) * scale
    
    # nframe = nimg1.copy()
    # noshape = noflow.shape
    # for i in range(0, noshape[0], 20):
    #     for j in range(0, noshape[1], 20):
    #         vc = tuple(map(int, noflow[i,j].tolist()))
    #         if abs(vc[0]) < 5 and abs(vc[1]) < 5:
    #             continue
    #         now = (j, i)
    #         before = (now[0] - vc[0], now[1] - vc[1])
    #         cv2.line(nframe, before, now, (0, 0, 255), 1)
    #         cv2.circle(nframe, now, 2, (255, 0, 0), -1)
    # cv2.imwrite('./result.jpg', nframe)

    # video
    video_path = '/mnt/sda/guoxiaolu/testvideo/inputFiles/v7_10.39.241.22.avi'
    cap = cv2.VideoCapture(video_path)
    res, img = cap.read()
    img0 = img
    fid = 0
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    vname = './test.mp4'
    out = cv2.VideoWriter(vname, fourcc, 25, (img.shape[1], img.shape[0]))

    while(True):
        res, img1 = cap.read()
        if res==False:
            print('read over')
            break
        fid += 1
        if fid > 1000:
            break
        print (fid)
        ishape = img0.shape
        nshape = ((ishape[0] // 64 + int(ishape[0] % 64 > 0)) * 64, (ishape[1] // 64 + int(ishape[1] % 64 > 0)) * 64, ishape[2])
        model = Pwcnet(model_path, w=nshape[1], h=nshape[0])
        nimg0 = np.zeros(nshape, np.uint8)
        nimg1 = np.zeros(nshape, np.uint8)
        nimg0[:ishape[0],:ishape[1],:] = img0
        nimg1[:ishape[0],:ishape[1],:] = img1
        oflow = model.Pre(nimg0, nimg1)
        scale = nshape[0] // oflow.shape[0]
        noflow = cv2.resize(oflow, (nshape[1], nshape[0])) * scale
        
        nframe = img1.copy()
        noshape = noflow.shape
        for i in range(0, noshape[0], 20):
            for j in range(0, noshape[1], 20):
                vc = tuple(map(int, noflow[i,j].tolist()))
                # if abs(vc[0]) < 5 and abs(vc[1]) < 5:
                #     continue
                now = (j, i)
                before = (now[0] - vc[0], now[1] - vc[1])
                cv2.line(nframe, before, now, (0, 0, 255), 1)
                cv2.circle(nframe, now, 1, (255, 0, 0), -1)

        out.write(nframe)
        img0 = img1
    out.release()
