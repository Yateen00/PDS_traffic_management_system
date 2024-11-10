# Traffic System Management at a Single Intersection Using PPO Reinforcement Learning

## Project Overview

This project focuses on optimizing traffic flow at a single intersection using Proximal Policy Optimization (PPO) reinforcement learning. By training an AI model to manage signal timing dynamically, we aim to reduce vehicle wait times and improve overall traffic efficiency. This repository is a modified version of [sumo-rl by LucasAlegre](https://github.com/LucasAlegre/sumo-rl.git), customized to enhance its applicability to our objectives.  
It also contains the YOLO prototype we plan to use for getting the observation space.

## Team Members

All team members are students at Sardar Patel Institute of Technology.

- [Jai Parameshwaran](https://github.com/paramj-sudo)
- [Nikhil Vishwakarma](https://github.com/Vishwakarma-Nikhil)
- [Vinayak Yadav](https://github.com/vinayakyadav2709)
- [Yateen Vaviya](https://github.com/Yateen00)

## Setup Instructions

### Prerequisites

Ensure that Python is installed on your system. This setup was tested to work with Python 3.10.12.

### YOLO Model Requirements

To use the YOLO model, download the following files and place them inside the `YOLO` folder:

- **yolov3.cfg**: [Download here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- **coco.names**: [Download here](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
- **yolov3.weights**: [Download here](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights)

### Create Virtual Environments

1. **Set up a virtual environment for the `YOLO` folder:**

   ```bash
   cd YOLO
   python3 -m venv yolo-venv
   source yolo-venv/bin/activate
   ```

   After activating the virtual environment, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   To deactivate the virtual environment once you’re done:

   ```bash
   deactivate
   ```

2. **Set up a virtual environment for the `model` folder:**

   ```bash
   cd ../model
   python3 -m venv model-venv
   source model-venv/bin/activate
   ```

   Similarly, install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   To deactivate this virtual environment when finished:

   ```bash
   deactivate
   ```

### Switching Between Virtual Environments

- When working with files in the `YOLO` folder, activate the `yolo-venv`:

  ```bash
  cd YOLO
  source yolo-venv/bin/activate
  ```

- When working with files in the `model` folder, activate the `model-venv`:

  ```bash
  cd ../model
  source model-venv/bin/activate
  ```

Always ensure the correct virtual environment is active before running commands in each respective folder.
