#FROM registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7
FROM paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6
#FROM registry.baidubce.com/paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6

ENV LANG=en_US.utf8
ENV LANG=C.UTF-8

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN pip3 install --upgrade pip

## install flask
#RUN pip3 install networkx==2.3 flask gevent gunicorn boto3 paddleocr==2.0.1
RUN pip3 install networkx flask gevent gunicorn boto3 paddleocr==2.6.1

RUN pip3 install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

#add folder
#RUN git clone -b release/2.1 https://github.com/PaddlePaddle/PaddleOCR.git /opt/program/
RUN git clone -b release/2.6.1 https://github.com/PaddlePaddle/PaddleOCR.git /opt/program/

#download model for inference
RUN mkdir /opt/program/inference/
RUN mkdir /opt/ml/
RUN mkdir /opt/ml/model/
RUN cd /opt/program/inference/

RUN wget -P /opt/program/inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar && tar -xf /opt/program/inference/ch_ppocr_server_v2.0_det_infer.tar -C /opt/program/inference/ && rm -rf /opt/program/inference/ch_ppocr_server_v2.0_det_infer.tar
RUN wget -P /opt/program/inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar && tar -xf /opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer.tar -C /opt/program/inference/ && rm -rf /opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer.tar

RUN wget -P /opt/program/inference/ https://paddleocr.bj.bcebos.com/PP-OCRv4/ch_PP-OCRv4_server_det_infer.tar && tar -xf /opt/program/inference/ch_PP-OCRv4_server_det_infer.tar -C /opt/program/inference/ && rm -rf /opt/program/inference/ch_PP-OCRv4_server_det_infer.tar

RUN wget -P /opt/program/inference/ https://paddleocr.bj.bcebos.com/PP-OCRv4/ch_PP-OCRv4_mobile_cls_infer.tar && tar -xf /opt/program/inference/ch_PP-OCRv4_mobile_cls_infer.tar -C /opt/program/inference/ && rm -rf /opt/program/inference/ch_PP-OCRv4_mobile_cls_infer.tar
### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# Set up the program in the image
COPY paddle/* /opt/program/
RUN chmod +x /opt/program/serve
WORKDIR /opt/program



