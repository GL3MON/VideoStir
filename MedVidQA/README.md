###########################################################
The MedVidQA Dataset
###########################################################

Initial Release, Jan 2022



Relevant files:
README.md (this file)
train.json (the training annotations)
val.json (the validation annotations)
test.json (the testing annotations)

###########################################################
train.json/val.json/test.json
###########################################################
The json file contains the following fields:

- sample_id:
Unique identifier for each sample.
- question:
Medical or health-related question
- answer_start:
The start of the visual answer to question in MM:SS format
- answer_end:
The end of the visual answer to question in MM:SS format
- answer_start_second:
The start of the visual answer to question in seconds
- answer_end_second:
The end of the visual answer to question in seconds
- video_length:
The length of the video in seconds
- video_id:
Unique identifier for YouTube video where visual answer exists.
- video_link:
Link to download YouTube video


This can be loaded into python as:

>>> import json
>>> with open('train.json', 'r') as rfile:
>>>     data_items = json.load(rfile)


Due to copyright issues, we cannot directly share the videos as a part of this dataset. The videos can be downloaded using pytube library (https://github.com/pytube/pytube):

>>> from pytube import YouTube
>>> YouTube('https://youtu.be2lAe1cqCOXo').streams.first().download()



###########################################################
Dataset statistics
###########################################################
Training dataset:
{'Total videos': 800, 'Total questions': 2710}

Validation Dataset:
{'Total videos': 49, 'Total questions': 145}

Test Dataset:
{'Total videos': 50, 'Total questions': 155}


###########################################################
CHANGELOG
###########################################################
1/10/2022
Initial release


###########################################################