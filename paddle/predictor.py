import sys
import json
import boto3
import os
import warnings
import numpy as np
from paddleocr import PaddleOCR
import cv2
import flask
import requests

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# Flask应用
app = flask.Flask(__name__)

# 确保新模型的参数存在
for i in ['/opt/program/inference/en_PP-OCRv3_det_infer',  # 使用 PP-OCRv3 英文检测模型
          '/opt/ml/model',  # 使用 PP-OCRv4 识别模型，模型路径指向 /opt/ml/model
          '/opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer']:  # 使用 PP-OCRv4 分类模型
    if os.path.exists(i):
        print(f"<<<< pretrained model exists for: {i}")
    else:
        print(f"<<< make sure the model parameters exist for: {i}")
        break

# 列出模型目录下的文件
print("<<< files under /opt/ml/model", os.listdir('/opt/ml/model/'))
print("Start loading models!")

# 使用 PP-OCRv3 检测模型和 PP-OCRv4 识别及分类模型
ocr = PaddleOCR(det_model_dir='/opt/program/inference/en_PP-OCRv3_det_infer',  # 英文检测模型
                rec_model_dir='/opt/program/inference/en_PP-OCRv4_rec_infer',  # 英文识别模型
                use_pdserving=False)  # 加载模型到内存
print("Models loaded successfully!")

# JSON编码器
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# 主推理函数
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

        # 保存结果
        res2 = {
            'label': [i[1][0] for i in result],
            'confidence': [i[1][1] for i in result],
            'bbox': [i[0] for i in result]
        }
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
                'confidences': res['confidence'],
                'bbox': res['bbox'],
                'shape': img_shape
            }
            _payload = json.dumps(inference_result, ensure_ascii=False, cls=MyEncoder)
            return flask.Response(response=_payload, status=200, mimetype='application/json')

        else:
            return flask.Response(response='Failed to download image from the provided URL',
                                  status=400, mimetype='application/json')

    else:
        return flask.Response(response='This predictor only supports JSON data with image URLs',
                              status=415, mimetype='text/plain')