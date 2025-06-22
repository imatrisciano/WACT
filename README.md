# Which Action Caused This

*Reasoning about world change to infer what happened.*


This project aims at building a predictor able to infer an action based on its pre and post conditions and is part of a Master Degree Thesis in Computer Scient at *Università Degli Studi di Napoli "Federico II"*.

This is still in a very early stage, more information will be added here at a later time.

## Prerequisites
Before running this project, you need to gather a **dataset** of action pre and post conditions, using [this tool](https://github.com/imatrisciano/ai2thor-hugo/).


### Quirks
If you are using ROCM and the program hangs while moving a tensor/model to your device, you might want to set the following enviroment variable `HSA_ENABLE_SDMA=0`
