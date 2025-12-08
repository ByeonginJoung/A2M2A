# A2M2A

---

### Overview

This repository introduces an A2M2A, which is an approach for 'Audio-to-MR video-to-Anime stylization'.
First, the audio to MR video conversion model should be trained and the converted results are utilized to stylization, by using [RAFT](https://github.com/princeton-vl/RAFT) to get deformation field and deform the converted MR video.

---


### Dataset Preparation by downloading the training data for audio-to-MR video.

To train the audio-to-MR video conversion model, please download [USC-Span dataset](https://sail.usc.edu/span/75speakers/).

After download dataset, you have to modify the path for dataset in the config files in 'config' folder.

---

### Environment:

We used following environment.

```bash
Ubuntu 20.04
cuda 12.2
Python 3.9.23
GCC 9.4.0
Pytorch 2.1.0+cu121
```

You can use your own cuda/torch versions.

---

### Installation

Clone this repository and install submodules, proper packages.

```bash
git clone --recursive https://github.com/byeonginjoung/A2M2A.git
cd A2M2A
pip install -r requirements.txt
```

---

### Usage

To train the audio2MR conversion model, you can run following script:

```bash
bash run_train.sh
```

This scripts run for all 75 speakers. If you want to use specific scene, you can modify environment variable "SUB_NAME_ARRAY".

After training for conversion model, the input audio is required in 'demo_items' folder. 

When the demo item is prepared, you can inference the entire pipleline by running the script:

```bash
bash run_pipeline.sh
```

You may have to change the environment variables in the bash file, run_pipeline.sh, to use your own audio file and pretrain weights.