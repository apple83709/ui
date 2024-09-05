from flask import Flask
from flask_cors import CORS
import os
from pypylon import pylon
import cv2

app = Flask(__name__)
CORS(app)

@app.route('/dir', methods=['GET'])
def get_directories():
    current_directory = os.getcwd()
    directories = os.listdir(f'{current_directory}/models')
    return {'dir': directories}


@app.route('/pred', methods=['GET'])
def get_prediction():
    # delete all files in directory: images/partno1/pred/*.*
    os.system('del /S /Q .\\images\\partno1\\pred\\*.*')
    get_sample()

    return {'prediction': [
        {'position': 'N', 'fail_count': 3},
        {'position': 'E', 'fail_count': 1},
        {'position': 'S', 'fail_count': 0},
        {'position': 'W', 'fail_count': 2},
    ]}

def get_sample():
    device_info = {}
    device_info['2676017D53E2'] = 'N'
    device_info['EEEEEEEEEEEE'] = 'E'
    device_info['SSSSSSSSSSSS'] = 'S'
    device_info['2676017D0FF9'] = 'W'

    instance = pylon.TlFactory.GetInstance()

    # Get all the available devices
    devices = instance.EnumerateDevices()

    for i, device in enumerate(devices):
        # 創建相機對象
        # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        print(f'Create camera {i}')
        camera = pylon.InstantCamera(instance.CreateDevice(devices[i]))
        guid = camera.GetDeviceInfo().GetDeviceGUID()
        print(f'Create guid {guid} = position: {device_info[guid]}')
        # 打開相機
        camera.Open()

        # 調整相機曝光時間
        # camera.ExposureAuto.SetValue('Off')
        # camera.ExposureTime.SetValue(10000)

        # 拍照存檔
        camera.StartGrabbing()
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data
            image = grabResult.Array
            img_path = f'images\\partno1\\sample\\{device_info[guid]}.png'
            cv2.imwrite(img_path, image)
            print(img_path)
        grabResult.Release()
        camera.StopGrabbing()


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
