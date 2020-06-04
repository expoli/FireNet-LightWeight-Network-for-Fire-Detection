import argparse

from detection.FireDetection import FireDetectioner
from tfsettings.gpu import InitGpu

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=False, help='path to test dataset')
ap.add_argument('-m', '--model', required=True, help='saved model file path')
ap.add_argument('-c', '--categories', required=False, help='categories')
ap.add_argument('-gui', '--gui_flag', required=False, default=0, help='the gui out flag')
ap.add_argument('-url', '--rtsp_url', required=False, default=0, help='the rtsp url')
args = vars(ap.parse_args())


def main():
    InitGpu.InitGpu().init()
    model_path = args['model']
    dataset_path = args['dataset']
    categories = args['categories']
    gui_flag = args['gui_flag']
    rtsp_url = args['rtsp_url']
    if rtsp_url:
        firedetectioner = FireDetectioner(modelPath=model_path, gui_flag=gui_flag, rtsp_url=rtsp_url)
    else:
        video_path = dataset_path + '/' + categories
        firedetectioner = FireDetectioner(modelPath=model_path, video_path=video_path, gui_flag=gui_flag)
    
    firedetectioner.detection()


if __name__ == '__main__':
    main()
