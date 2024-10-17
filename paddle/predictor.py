import sys
import json
import boto3
import os
import warnings
import numpy as np
from paddleocr import PaddleOCR
import cv2
import flask

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# Flask应用
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

# 确保新模型的参数存在
for i in ['/opt/program/inference/ch_PP-OCRv4_server_det_infer',  # 使用最新的 PP-OCRv4 检测模型
          '/opt/ml/model',
          '/opt/program/inference/ch_PP-OCRv4_server_cls_infer']:  # 使用最新的 PP-OCRv4 分类模型
    if os.path.exists(i):
        print(f"<<<< pretrained model exists for: {i}")
    else:
        print(f"<<< make sure the model parameters exist for: {i}")
        break

# 列出模型目录下的文件
print("<<< files under /opt/ml/model", os.listdir('/opt/ml/model/'))
print("Start loading models!")

# 使用最新的 PP-OCRv4 模型
ocr = PaddleOCR(det_model_dir='/opt/program/inference/ch_PP-OCRv4_server_det_infer',
                rec_model_dir='/opt/ml/model',
                rec_char_dict_path='/opt/program/ppocr_keys_v1.txt',
                cls_model_dir='/opt/program/inference/ch_PP-OCRv4_server_cls_infer',
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
            print(result)

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

        bucket = data['bucket']
        image_uri = data['image_uri']
        download_file_name = image_uri.split('/')[-1]
        print(f"Download file name: {download_file_name}")

        s3_client.download_file(bucket, image_uri, download_file_name)
        print('Download finished!')

        try:
            res, img_shape = bbox_main('img_path', download_file_name, detect='paddle')
        except Exception as e:
            print(e)

        inference_result = {
            'label': res['label'],
            'confidences': res['confidence'],
            'bbox': res['bbox'],
            'shape': img_shape
        }
        _payload = json.dumps(inference_result, ensure_ascii=False, cls=MyEncoder)

        os.remove(download_file_name)
        return flask.Response(response=_payload, status=200, mimetype='application/json')

    elif flask.request.content_type == 'image/jpeg':
        data = flask.request.data
        data_np = np.fromstring(data, dtype=np.uint8)
        data_np = cv2.imdecode(data_np, cv2.IMREAD_UNCHANGED)
        data_np = cv2.cvtColor(data_np, cv2.COLOR_BGR2RGB)

        try:
            res, img_shape = bbox_main('img', data_np, detect='paddle')
        except Exception as e:
            print(e)

        inference_result = {
            'label': res['label'],
            'confidences': res['confidence'],
            'bbox': res['bbox'],
            'shape': img_shape
        }
        _payload = json.dumps(inference_result, ensure_ascii=False, cls=MyEncoder)
        return flask.Response(response=_payload, status=200, mimetype='application/json')

    else:
        return flask.Response(response='This predictor only supports JSON data and JPEG image data',
                              status=415, mimetype='text/plain')