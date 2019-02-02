To Download NEU emotional corpus
http://www.coe.neu.edu/Research/AClab/Speech%20Data/

Download pre-procesed data and trained weights from the following drive link:
https://drive.google.com/open?id=1gs12BDqg2eJbrEiHQ_KZRp1scmj3xkmv

Note: "Jenie Pre-processed" and "Trained Weights" are to be downloaded in the current directory to maintain hard-coded directory structure.

To pre-process the audio waveforms, edit path in the audio_preprocess.py code to the path of particular emotion and then run: python audio_preprocess.py
Similarly, to change the emotion in the training process, edit the train.py file to the hard-coded path for the jenie pre-processed emotion
To start the training, run: python train.py
To test the performance of the trained weights, run: python synth.py --emotion=<ENTER EMOTION HERE>
Example : emotion=amused

audio.wav file will be generated in trained weight directory. 
Generated audio samples: https://soundcloud.com/aditya-mukewar-801416775/sets/emotional-speech-synthesis-samples

