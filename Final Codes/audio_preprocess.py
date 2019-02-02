#Import all the required libraries
#Importing all the audio utility functions
import audio_utils
#Import for file handling
import os
#For basic math operation
import numpy as np
#To construct argument parser
import argparse

#Write Metadata details file name, number of utterences to text file
def write_metadata(metadata, out_dir):
#Parameters: metadata: Collection of metadata for all the audio files in in_dir
#            out_dir: Diretory path where all the mel and linear frequency spectrum are written

  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[2] for m in metadata])
  hours = frames * 12.5 / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max(m[2] for m in metadata))

#Preprocess a single utterance audio - text pair.
#Writes linear and mel-scale spectrograms to disk.
def process_utterance(odir, index, wav_path, text):
  
#Parameters: out_dir: The directory to write the spectrograms into
#            index: The numeric index to use in the spectrogram filenames.
#            wav_path: Path to the audio file containing the speech input
#            text: The text spoken in the input audio file

#Returns:(spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  

  # Load the audio to a numpy array:
  wav = audio_utils.load_audio(wav_path)
  print(wav_path.split('\\')[-1].split('.')[0])
  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio_utils.spectrogram(wav).astype(np.float32)

  n_frames = spectrogram.shape[1]
  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio_utils.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'jeine-spec-%05d.npy' % index
  mel_filename = 'jeine-mel-%05d.npy' % index
  np.save(os.path.join(odir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(odir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)


def main():

    #Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--emotion', required = True, 
                    help = 'The kind of emotion of which the wav files are to be preprocessed.')    
    args = vars(ap.parse_args())

    #I/O directory paths
    in_dir = 'jeine\\' + args['emotion']
    out_dir = './jenie_Processed//' +  args['emotion'] + '//'
    os.makedirs(out_dir, exist_ok=True)

    #Load the metadata csv file and parse text and wav filename
    metadata = []
    index = 1
    with open(os.path.join(in_dir, 'Metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[1]
            metadata.append(process_utterance(out_dir, index, wav_path, text))
            index += 1

    write_metadata(metadata, out_dir)

if __name__ == '__main__':
    main()