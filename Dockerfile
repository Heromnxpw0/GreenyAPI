FROM python:3.9

WORKDIR /src

ADD ./ /src

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.11/MindSpore/unified/x86_64/mindspore-2.2.11-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 8000 

CMD ["python", "api.py"]