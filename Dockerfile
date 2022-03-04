# Jupyter Deep Learning Notebook
FROM continuumio/anaconda3:latest

# needed for opencv-python
RUN RUN apt-get -y update && apt-get -y install ffmpeg libsm6 libxext6

RUN conda install -y jupyter
RUN pip install scikit-learn seaborn pandas tensorflow opencv-python

RUN mkdir /sjsu

# Go into working dir /sjsu
WORKDIR /sjsu

# copy dev host contents of cmpe257_ld/ to working dir in container
COPY . ./cmpe257_ld

# start jupyter notebook
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
