# Which Action Caused This

*Reasoning about world change to infer what happened.*


This project aims at building a predictor able to infer an action based on its pre and post conditions and is part of a Master Degree Thesis in Computer Scient at *Università Degli Studi di Napoli "Federico II"*.

This is still in a very early stage, more information will be added here at a later time.

## Prerequisites

Here the necessary steps to set your environment up to run this project.

### Dataset

Before running this project, you need to gather a **dataset** of action pre and post conditions, using [this tool](https://github.com/imatrisciano/ai2thor-hugo/).

---

### Python environment
Please create a python virtual environment and install all the required python dependencies by running:

```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

---

### Ollama
[Ollama](https://ollama.com/) is the local llm inference engine used in this project. In order to run the chatbot, ollama is required, please install it using the [latest instructions](https://ollama.com/download) from their website. 

If you wish to switch to another inference engine, e.g. if you wish to use an online model, you can do so by installing the corresponding ollama packages and modifying `WACTChatBot`'s initialization code

After installing ollama, make sure the chosen large language model's files are available by downloading them:
```bash
ollama pull qwen3:1.7b
```

You can test that the model is working by running:
```bash
ollama run qwen3:1.7b
```

---

### AMD ROCM Quirks
If you are using ROCM and the program hangs while moving a tensor/model to your device, you might want to set the following environment variable `HSA_ENABLE_SDMA=0`
