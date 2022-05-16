# Deploy Rasa Lesion Detection VA

## Software on Backend

The following software was launched in Docker containers based on Docker Desktop version: 4.3.2

- Rasa 3.1.0 (Rasa Chatbot Server)
- Rasa 3.1.1 SDK (Rasa Actions Server)
- Ngrok version latest (Exposes Rasa VA to Internet)
- Tensorflow-GPU 2.5.0

## Action Items

- [x] Ability to **understand 8 user utterances/queries** (including reasonable variations of those queries) 4 from Caregiver and 4 from Rasa.
    - [ ] Create Physician Story Path
    - [ ] Create Doctor Story Path
    - [ ] Create Surgeo Story Path
    - [ ] Create Endoscopist Story Path

- [x] Ability to use Rasa Custom Actions to deploy at least **2 Deep Learning Models to perform Polyp Recognition**. (At least one of these APIs must be outside of what is built into Rasa i.e time/weather)
    - [ ] Custom Action: Deploys **Keras CNN Polyp Classifier**
    - [ ] Custom Action: Deploys **YoloV4 Polyp Detector**
    - [ ] Custom Action: Deploys **Faster RCNN Polyp Segmentor**

- [ ] Were we able to deploy our Deep Learning models into a production environment?

- [ ] Deploy Rasa App on Slack

## Way to Update Rasa

Order Rasa goes for Files to Answer User comment/question/etc:

- Rasa: nlu.yml --> stories.yml --> domain.yml

1\. I will start by greeting Rasa: **`hi`**. Then that example greet will be linked to **intent: `greet`**. Reference **`nlu.yml`**

- Me: `hi`
- Rasa: for `hi` example --> **`nlu.yml`** --> `intent: greet`

2\. Rasa goes to **`stories.yml`** for intents and actions to know how to respond to **intent: `greet`** with an **action: `utter_greet`**.

- Rasa: for `nlu.yml`'s `intent: greet` example -->  `stories.yml` --> `action: utter_greet`

3\. Rasa goes to **`domain.yml`** to know which action to say for **utter_greet**, which Rasa finds to say **text: `"Hey! How are you?"`**. We see that back in our chat window.

- Rasa: goes to `domain.yml` for response to `utter_greet` --> `text: "Hey! How are you?"`


## Building a RASA Assistant in Conda or Docker

### Setup RASA Project

~~~bash
# Setup RASA GPU Project
cd Lesion-Detection\lesion_detection_va


# DEMO
docker run --name rasa-init --gpus all -it --privileged -v C:\Users\james\Documents\GitHub\Lesion-Detection\lesion_detection_va:/app rasa_3.1.0_rasa_sdk_1.1.1:dev

# go into docker container, rasa init
rasa init --no-prompt

# Yours
docker run --name rasa-init --gpus all -it --privileged -v ${PWD}\GitHub\Lesion-Detection\lesion_detection_va:/app rasa/rasa:3.1.0-full init --no-prompt

cd ${PWD}\GitHub\Lesion-Detection\lesion_detection_va
rasa init --no-prompt
~~~

### Train RASA Model

~~~bash
# Train Rasa Model
docker run --name rasa-train --gpus all -it --privileged -v ${PWD}\GitHub\Lesion-Detection\lesion_detection_va:/app rasa/rasa:3.1.0-full 

# DEMO
docker run --name rasa-train --gpus all -it --privileged -v  C:\Users\james\Documents\GitHub\Lesion-Detection\lesion_detection_va:/app rasa_3.1.0_rasa_sdk_1.1.1:dev

rasa train --domain domain.yml --data data --out models


cd ${PWD}\GitHub\Lesion-Detection\lesion_detection_va
rasa train --domain domain.yml --data data --out models
~~~


### Talk to your Virtual Assistant From Shell

~~~bash
# Talk to your Virtual Assistant From Shell
docker run --name rasa-shell --gpus all -it --privileged -p 5005:5005 -v ${PWD}\GitHub\Lesion-Detection\lesion_detection_va:/app rasa/rasa:3.1.0-full shell

cd ${PWD}\GitHub\Lesion-Detection\lesion_detection_va
rasa shell
~~~

### Talk to your Virtual Assistant From Slack

To launch Rasa server, so our Virtual Assistant is just running in the Docker container, you do docker run followed by Rasa **run** command, so then external clients like Unity in our case can interact with our Virtual Assistant.

~~~bash
# Talk to your Virtual Assistant from Unity
docker run --name rasa-run --gpus all -it --privileged -p 5005:5005 -v ${PWD}\GitHub\Lesion-Detection\lesion_detection_va:/app rasa/rasa:3.1.0-full run


# DEMO
docker run --name rasa-run --gpus all -it --privileged -p 5005:5005 -v  C:\Users\james\Documents\GitHub\Lesion-Detection\lesion_detection_va:/app --net colon-cancer-mayoclinic rasa_3.1.0_rasa_sdk_1.1.1:dev

cd app
rasa run



cd ${PWD}\GitHub\Lesion-Detection\lesion_detection_va
rasa run
~~~

### Adding Custom Actions

IMPORTANT: For your custom actions to be called, you need at least 2 story paths where they are called. If you only have 1, it wont be called.

1\. Build a custom action using Rasa SDK by editing **`actions/actions.py`**:

~~~python
import requests
import json
from rasa_sdk import Action

class ActionJoke(Action):
    def name(self):
        return "action_joke"

    def run(self, dispatcher, tracker, domain):
        request = requests.get("http://api.icndb.com/jokes/random").json() # make an api call
        joke = request["value"]["joke"] # extract a joke from returned json response
        dispatcher.utter_message(text=joke) # send the message back to the user
        return []

~~~

1\. In **`data/stories.yml`**, replace **`utter_cheer_up`** with the custom action **`action_joke`** tell your bot to use this new action:

2\. In **`domain.yml`**, add a section for custom actions, including your
new action

~~~yml
actions:
  - action_joke
~~~

3\. After updating your domain and stories, you must retrain your model

~~~bash
docker run --name rasa-train --gpus all -it --privileged -v ${PWD}\GitHub\Lesion-Detection\lesion_detection_va:/app rasa/rasa:3.1.0-full train --domain domain.yml --data data --out models
~~~

Your custom actions will run on a separate server from your Rasa server.

5\. Create a network to connect the two containers: docker network create my-project

~~~bash
# Test
docker network create my-project

# MVP Project
docker network create colon-cancer-mayoclinic
~~~

6\. Run the custom actions with the following command: 

~~~bash

# MVP Project
docker run -d --name cc-action-server --gpus all -it --privileged -p 5055:5055 -v ${PWD}\GitHub\Lesion-Detection\lesion_detection_va\actions:/app/actions --net colon-cancer-mayoclinic rasa/rasa-sdk:3.1.1

# DEMO
docker run --name ld-action-server --gpus all -it --privileged -p 5055:5055 -v  C:\Users\james\Documents\GitHub\Lesion-Detection\lesion_detection_va\actions:/app/actions --net colon-cancer-mayoclinic rasa_3.1.0_rasa_sdk_1.1.1:dev

cd app/actions
rasa run actions



cd ${PWD}\GitHub\Lesion-Detection\lesion_detection_va\actions

cd C:\Users\james\Documents\GitHub\Lesion-Detection\lesion_detection_va\actions
rasa run actions
~~~


- **`d`**: Runs container in detached mode, so you can run the Rasa container in same window
- **`v $(pwd):/app`**: Mounts your project directory into Docker container, so that action server can run the code in **`actions`** folder
- **`net my-project`**: Run server on a specific network, so Rasa container can find it 
- **`--name rasa-action-server`**: Gives server a specific name for Rasa server to reference
- **rasa/rasa-sdk:3.1.1**: Uses Rasa SDK image with tag 3.1.1
  
Run **`docker stop rasa-action-server`**. Run **`docker ps`** to see all currently running containers

7\. To instruct Rasa server to use action server, tell Rasa its location by adding this endpoint to your **`endpoints.yml`** and referencing the **`--name`** you gave the server (in our example above **`rasa-action-server`**):

~~~yml
action_endpoint:
    url: "http://rasa-action-server:5055/webhook"
~~~

8\. Now we can talk to our Rasa Chatbot via shell or from our Unity Client

~~~bash
# Test: Talk to your Virtual Assistant From Shell
docker run --name rasa-shell --gpus all -it --privileged -p 5005:5005 -v ${PWD}\GitHub\Lesion-Detection\lesion_detection_va:/app --net my-project rasa/rasa:3.1.0-full shell
~~~

Talk to our Rasa Chatbot from Slack:

~~~bash
# DEMO
docker run --name rasa-run --gpus all -it --privileged -p 5005:5005 -v  C:\Users\james\Documents\GitHub\Lesion-Detection\lesion_detection_va:/app --net colon-cancer-mayoclinic rasa_3.1.0_rasa_sdk_1.1.1:dev

cd app
rasa run
~~~

## Expose Rasa Chatbot App on Internet using Ngrok

Rasa Chatbot server is up and running on **port 5005** in our Docker container.

Launch **ngrok** Docker container to create a tunnel from the newly assigned public URL to port **5005** on the Rasa Chatbot server container called **rasa-run** (at which the **Rasa Chatbot Application** is handling requests.

~~~bash
docker run -d -p 4040:4040 --privileged --net colon-cancer-mayoclinic --name ngrok-integ-rasa wernight/ngrok ngrok http rasa-run:5005

# NOTE: Rasa run server must be running before using ngrok
rasa run

ngrok http 5005

~~~

The following public urls are examples of what you see going to **http://localhost:4040/inspect/http**

~~~md
### No requests to display yet

To get started, make a request to one of your tunnel URLs:

http://ea38-130-65-254-19.ngrok.io
https://ea38-130-65-254-19.ngrok.io

~~~

You can go to ngrok web page to see the public url, so now our Rasa Chatbot GI Cancers Application can be accessed from any client anywhere in the world:

## Talk to Rasa Virtual Assistant from Slack

To make our Rasa Chatbot available to Slack, use the **ngrok URL** followed by webhooks followed by Slack channel followed by webhook.

~~~bash
http://ea38-130-65-254-19.ngrok.io/webhooks/slack/webhook
~~~

## Appendix: Build a Custom Docker Image Rasa & Scrapy

If you need to build a custom Docker image with Rasa, Rasa SDK Actions Server and Scrapy Web Spider, run the following command:

~~~bash
cd C:\Users\james\Documents\GitHub\Lesion-Detection\lesion_detection_va\rasa_dockerfile

docker build -t rasa_3.1.0_rasa_sdk_3.1.1:dev .
~~~





