FROM continuumio/anaconda3:latest

RUN apt-get -y update
RUN conda install -y jupyter
RUN pip install rasa==3.1.0 rasa-sdk==3.1.1 scikit-learn scrapy scrapydo tqdm

EXPOSE 5005
ENTRYPOINT ["/bin/bash"]
