FROM python:3.9

WORKDIR /src

ADD ./ /src

RUN chmod 777 95.ckpt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 

CMD ["python", "api.py"]