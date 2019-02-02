import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


src_path = './Trained Weights/'

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(src_path + 'model.ckpt-' + args.emotion)
  base_path = get_output_base_path(src_path)
  i = 0
  while(1):
    text = input('Enter the text to synthesize:')
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))
    print('[INFO] Synthesized audio file is written at ' + path)
    u = input('Do you want to continue? N or Y')

    if u == 'N' :
      break


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--emotion', required=True, help='Type of emotion to synthesize.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
