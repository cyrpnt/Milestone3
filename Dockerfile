FROM python:3.8.10

COPY . /app

WORKDIR /app/code

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install numpy==1.22.3
RUN pip install tensorflow==2.9.1
RUN pip install tqdm==4.66.1
RUN pip install keras==2.9.0
RUN pip install scikit-learn==1.3.2
RUN pip install pillow
RUN pip install flask
