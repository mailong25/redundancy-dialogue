import os
import argparse
os.environ['TRANSFORMERS_CACHE'] = '/home/people/20202939/scratch/trans_cache/'
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, required=False, type=str)
parser.add_argument('--generate', action='store_true')
args = parser.parse_args()

if args.path is not None:
    ckpt_files = [p for p in os.listdir(args.path) if 'model.checkpoint' in p and 'model.checkpoint.' not in p]
    ckpt_files = [os.path.join(args.path, p) for p in ckpt_files]
    ckpt_files = sorted(ckpt_files)[2:]

for file in ckpt_files:
    MODEL_PATH = os.path.join(args.path, 'model')
    LOG_PATH = 'logs/' + '_'.join(file.split('/')[-2:])
    print(file)
    print(MODEL_PATH)
    print(LOG_PATH)
    
    if args.generate:
        try:
            os.remove(MODEL_PATH)
            print("Deleted:", MODEL_PATH)
        except:
            pass
        
        os.system('cp ' + file + ' ' + MODEL_PATH)
        os.system('python interactive.py --model ' + MODEL_PATH + ' --log ' + LOG_PATH + ' --blocking')
    
    os.system('python eval.py --log ' + LOG_PATH)