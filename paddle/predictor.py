import sys
import json
import boto3
import os
import warnings
import numpy as np
from paddleocr import PaddleOCR
from flask import jsonify
import cv2
import flask
import requests

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# Flask应用
app = flask.Flask(__name__)

# 确保新模型的参数存在
for i in ['/opt/program/inference/ch_PP-OCRv4_det_infer.tar',
          '/opt/program/inference/ch_PP-OCRv4_rec_infer.tar',
          '/opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer']:
    if os.path.exists(i):
        print(f"<<<< pretrained model exists for: {i}")
    else:
        print(f"<<< make sure the model parameters exist for: {i}")
        break

# 列出模型目录下的文件
print("<<< files under /opt/ml/model", os.listdir('/opt/ml/model/'))
print("Start loading models!")

# 使用 PP-OCRv3 检测模型和 PP-OCRv4 识别及分类模型
ocr = PaddleOCR(det_model_dir='/opt/program/inference/ch_PP-OCRv4_det_infer.tar',  
                rec_model_dir='/opt/program/inference/ch_PP-OCRv4_rec_infer.tar',  
                #lang='en',  # 指定语言为英文
                use_pdserving=False)  # 加载模型到内存
print("Models loaded successfully!")

def bbox_main(type, imgpath, detect='paddle'):
    if detect == 'paddle':
        if type == 'img_path':
            img = cv2.imread(imgpath)
            img_shape = img.shape
            print(f"<<< img shape: {img_shape}")
            result = ocr.ocr(imgpath, rec=True)
            print(result)
        elif type == 'img':
            img_shape = imgpath.shape
            print(f"<<< img shape: {img_shape}")
            result = ocr.ocr(imgpath, rec=True)
            print(f"<<< image ocr result as below: {result}")

        # 检查并展开嵌套的结果列表
        if len(result) == 1 and isinstance(result[0], list):
            result = result[0]

        # 将NumPy数据类型转换为Python原生数据类型
        res2 = {
            'label': [str(i[1][0]) for i in result],          # 文本内容
            'confidence': [float(i[1][1]) for i in result],   # 置信度
            'bbox': [[[float(coord) for coord in point] for point in i[0]] for i in result]  # 边框坐标
        }
        img_shape = [int(dim) for dim in img_shape]  # 转换为Python的int

        print(f"<<< image ocr res2 as below: {res2}")

        return res2, img_shape
    else:
        return

@app.route('/ping', methods=['GET'])
def ping():
    """检查容器是否健康"""
    health = 1
    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

# 定义自定义的JSON编码器
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list, tuple)):
            return [self.default(item) for item in obj]
        else:
            return super(MyEncoder, self).default(obj)

# 修改后的invocations函数
@app.route('/invocations', methods=['POST'])
def invocations():
    """执行推理"""
    print("================ INVOCATIONS =================")
    print(f"Content-Type: {flask.request.content_type}")

    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)

        image_url = data['image_url']
        print(f"Image URL: {image_url}")

        # 下载图像
        response = requests.get(image_url)
        if response.status_code == 200:
            img_data = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("Image downloaded and decoded successfully!")

            try:
                res, img_shape = bbox_main('img', img, detect='paddle')
            except Exception as e:
                print(f"Error during inference: {e}")
                return flask.Response(response='Error during inference', status=500)

            inference_result = {
                'label': res['label'],
                'confidence': res['confidence'],
                'bbox': res['bbox'],
                'shape': img_shape
            }

            print(f"<<< image ocr inference_result as below: {inference_result}")

            return flask.Response(response=json.dumps(inference_result, ensure_ascii=False, cls=MyEncoder),
                                  status=200,
                                  mimetype='application/json')

        else:
            return flask.Response(response='Failed to download image from the provided URL',
                                  status=400, mimetype='application/json')

    else:
        return flask.Response(response='This predictor only supports JSON data with image URLs',
                              status=415, mimetype='text/plain')