# Deploy CMPE257 LD TensorFlow Docker Container

1\. Build Docker image from Dockerfile:

~~~bash
# Create proj/ folder
mkdir -p $HOME/Documents/GitHub/
cd $HOME/Documents/GitHub/

# Clone cmpe257 Lesion Detection project repo to GitHub folder
git clone git@github.com:shreyahunur/Lesion-Detection.git
cd Lesion-Detection

# Build cmpe257_ld docker image with Anaconda3, TensorFlow Jupyter Notebook, 
# if on linux, prepend sudo
docker build -t cmpe257_ld:dev .
~~~

2\. Deploy cmpe257 Lesion Detection TensorFlow Docker container:

~~~bash
# volume mount created from dev host curr proj $PWD to /sjsu/cmpe257 in container
# if on linux, prepend sudo
docker run --name cmpe257_ld_dev -p 8888:8888 -it -v $PWD:/sjsu/cmpe257_ld cmpe257_ld:dev
docker run --name cmpe257_ld_dev -p 8888:8888 -it cmpe257_ld:dev
~~~


