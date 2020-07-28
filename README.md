本程序针对multi-person tracking任务，集成如下方法，并在内部逻辑做修改：
https://github.com/Tianxiaomo/pytorch-YOLOv4.git
https://github.com/NVlabs/DG-Net
https://github.com/sniklaus/pytorch-pwc
https://github.com/Amanbhandula/AlphaPose
https://github.com/microsoft/human-pose-estimation.pytorch
models下载地址：https://pan.baidu.com/s/1OeL3H5_ng-Xs_hIaTkrXHQ 提取码: rmh3。下载模型放置到./models目录下
模型包含：
yolov4检测模型：yolov4.cfg，yolov4.weights
dgnet-reid模型：gen_00100000.pt，id_00100000.pt，config.yaml，net_last.pth，opts.yaml
pwcnet光流模型（GPU下）：pwcnet.pth
alphapose人体关键点模型：duc_se.pth

main.py可直接运行，可以可视化，即输入已视频文件，在当前目录下输出结果mp4文件
main_interface.py可按约定标准json接口输出，但不可可视化

本框架基本接近deepsort框架，每帧均作光流，预测速度；设定帧频率下做人体检测，pose检测，人体匹配逻辑采用reid模型和oks（有人体关键点）/iou（只依靠人体检测狂）。同时，如果设定帧数未检测到人体，tracker消失，但如果一定帧数，重新检测到，即再恢复id。一旦检测到人体，立即创建tracker，未要求检测到一定帧才创建tracker。
cpu光流采用opencv稠密光流，gpu/cpu下可通用；gpu光流采用模型，建议逐帧检测。二者输出结果略有不同，使用gpu光流时，输出tracker数略多。

模块化设计，可对应替换相应模块