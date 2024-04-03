#this line instructs Docker to start with an official Python runtime image for version 3.8 with the "slim" variant. 
FROM python:3.9-slim

#Sets the working directory inside the container to /docker_app. This is the directory where subsequent commands will be executed.
WORKDIR /docker_app

#Copies the contents of the current directory (where the Dockerfile is located) into the container's /docker_app directory.
COPY app.py /docker_app/app.py
COPY model.joblib /docker_app/model/model.joblib
COPY f_df.joblib /docker_app/f_df.joblib
COPY requirements.txt /docker_app/requirements.txt
COPY src/ /docker_app/src

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#expose to port 
EXPOSE 8080

#Specifies the default command to run when the container starts. In this example, it runs the Python script app.py. 
#The CMD instruction provides defaults for an executing container, but it can be overridden when running the container.
CMD ["python", "app.py"]