FROM python:3.7
LABEL maintainer "Shaham Farooq <shaham.farooq@mail.utoronto.ca>"


RUN apt-get update

RUN mkdir /app
WORKDIR /app
COPY . /app

RUN mkdir ~/.aws


RUN cp /app/aws/config ~/.aws/
RUN cp /app/aws/credentials ~/.aws/



RUN pip install --no-cache-dir -r requirements.txt
# RUN pip3 install opencv-python

# RUN apt-get install libmagickwand-dev
# RUN pip install Wand

ENV FLASK_ENV="docker"
EXPOSE 5000