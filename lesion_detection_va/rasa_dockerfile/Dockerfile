FROM continuumio/anaconda3:latest

# conda install: cudnn opencv cudatoolkit==10.1.243
# pip install: tqdm matplotlib tensorflow==2.7.1 opencv-python==4.1.1.26 lxml tqdm absl-py easydict pillow

RUN apt-get -y update
RUN apt-get -y install ffmpeg libsm6 libxext6
RUN conda install -y jupyter
RUN pip install rasa==3.1.0 rasa-sdk==3.1.1 scikit-learn scrapy scrapydo lxml tqdm absl-py easydict pillow opencv-python

EXPOSE 5005
ENTRYPOINT ["/bin/bash"]
