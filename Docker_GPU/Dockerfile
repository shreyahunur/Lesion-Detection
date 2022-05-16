# Jupyter Deep Learning Notebook
FROM continuumio/anaconda3:latest

# needed for opencv-python
RUN apt-get -y update && apt-get -y install ffmpeg libsm6 libxext6

RUN apt-get -y update
RUN pip install scikit-learn seaborn pandas tensorflow-gpu==2.3.0 opencv-python lxml jupyter xgboost torch pykeops sentence-transformers
RUN wget https://github.com/jgm/pandoc/releases/download/2.17.1.1/pandoc-2.17.1.1-1-amd64.deb
RUN dpkg -i pandoc-2.17.1.1-1-amd64.deb
RUN apt-get install -y texlive-xetex texlive-fonts-recommended texlive-plain-generic

RUN mkdir /sjsu

# Go into working dir /sjsu
WORKDIR /sjsu

# copy dev host contents of cmpe257_ld/ to working dir in container
COPY . ./cmpe257_ld

# start jupyter notebook
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
