# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-bullseye

# # Install pip requirements
# COPY requirements.txt ./
# RUN pip3 install -r ./requirements.txt

RUN pip3 install Flask
RUN pip3 install gunicorn
RUN pip3 install opencv-python

# Copy local code to the container image.
ENV APP_HOME ./app
WORKDIR $APP_HOME

COPY ./ ./

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the web service on container startup.
CMD exec gunicorn --bind :8080 --workers 1 --threads 1 --timeout 0 main:app
