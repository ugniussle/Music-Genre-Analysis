# Content-Based Music Genre Analysis and Recommender

_Welcome to my project!_

In this project, I will explore the ability of convolutional neural networks to extract features from spectrograms of songs. A model will then be trained on these features to perform classification of their respective genres. I will also play around with a system and perform some analysis on the Beatles Discography.

## Table of Contents
---

- [Project Motivation](#project-motivation)
- [Workflow](#workflow)
- [Requirements](#Requirements)
---

<a id='project-motivation'></a>
## Project Motivation

The feeling that we get when we listen to music - the very sound waves that vibrate our ear drums and fire up our brains in response - is as subjective as it gets. There are no two ways about it: one person might enjoy the soft and ethereal tunes of dream pop, while another would enjoy the intensity and power of industrial metal. Generally, _people tend to stick with and listen to just a few musical genres that they favour or have an affinity with_, be it pop, rock, jazz, or classical music, to name a few. 

What defines a music genre? Does the answer lie in the instruments that play in the song? Is it defined by the speed in which the music plays? Or is it the message that the singer is trying to get across? Maybe, it is a combination of all of those elements, underpinned by an unspoken rule that permeates through everybody's consciousness? Without any information - the title of the song, the artist's name, the date the song was released - a person would easily be able to identify the genre of a song in a few seconds to a high degree of accuracy.

This brings me to my project. __What I am trying to do in this project is to develop a model to perform accurate classification of music genres to a high degree of accuracy. The only data that will be available to the model is the raw audio file itself - no metadata or other information will be given other than the actual audio for classification.__

#### Why is this important?

Parties that would benefit most from robust models to extract audio information is are music companies like Spotify, SoundCloud, or Apple. For example, Spotify receives [thousands](https://expandedramblings.com/index.php/spotify-statistics/) of new tracks a day, and having a model to extract audio features from the tracks would be invaluable for classification. In particular, their playlist generation and music recommendation system benefits from augmenting a traditional [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) based recommender with content-based recommenders to aid with the [cold start problem](https://en.wikipedia.org/wiki/Cold_start) for new tracks that are recently uploaded. 

Aside from the music industry, there could be significant benefits to be had with other industries. With this, it is possible that any audio could potentially be run through a similar model, and important features could be extracted and analysed or classified.

---
<a id='workflow'></a>
## Workflow of this Repository.

This repository consists of a few notebooks (found in the [notebooks](/Notebooks) folder) and models (found in the [models](/Models) folder). 

This is the general workflow of the project:

![](/Images/notebookworkflow.jpg)

The ```extractingfeatures.ipynb ``` notebook will generate and extract the features to be trained on the models.

The ``` loadXandY.ipynb ``` notebook is the next notebook that will load the pickle files generated from  ```extractingfeatures.ipynb ``` and prepare the data for feeding into the convolutional model. 

Next, the modelling notebooks (with names starting with "conv"  eg: ```convpool.ipynb```) are used to train the model and show the results from the training. There is also a notebook ```resnet-batch50.ipynb``` which utlises transfer learning by finetuning a pre-trained ResNet50 model.

The test notebook ```checkingtestsetresnet.ipynb``` is also used to evaluate and perform analysis on the test set using the ResNet50 architecture. A simple content-based recommender was also created in this notebook that mapped the tracks which had the closest predictions to any given track.

The notebook ```Beatles.ipynb``` explores the Beatles discography by running the audio files through the classifier. An exploration was also done by finding audio files with the closest link based on the probability distribution of the class predictions.

Further information and theory is shown inside the jupyter notebooks, so please feel free to hop inside to learn more!

The notebooks should be viewed in this order:

* ```extractingfeatures.ipynb```
* ```loadXandY.ipynb```
* ```convpoolnoaug.ipynb```
* ```convpoolwithaug.ipynb```
* ```convconvpoolnoaug.ipynb```
* ```convconvpoolwithaug.ipynb```
* ```convconvpoolwithaug_80epochs.ipynb```
* ```resnet-batch50.ipynb```
* ```checkingtestsetresnet.ipynb```

---

<a id='Requirements'></a>
## Requirements

The general requirements are listed here. Typically, using ```pip``` would work well in installing these packages. Further details on the packages used are stated in the jupyter notebooks.
```python``` packages:
* ```PIL```
* ```librosa```
* ```keras```
* ```tensorflow```
* ```pandas```
* ```numpy```
* ```sklearn```

---











