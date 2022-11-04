# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# # Install pip requirements
# COPY requirements.txt ./
# RUN pip3 install -r ./requirements.txt

# Copy local code to the container image.
ENV APP_HOME ./       
#app
WORKDIR $APP_HOME

COPY ./ ./

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install Flask
RUN pip3 install gunicorn
RUN pip3 install opencv-python
RUN pip3 install six
RUN pip3 install tensorflow-cpu
RUN pip3 install PIL

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the web service on container startup.
CMD exec gunicorn --bind :8080 --workers 1 --threads 1 --timeout 0 main:app
