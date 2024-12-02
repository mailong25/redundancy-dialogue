import argparse
import time, os, torch

parser = argparse.ArgumentParser()
parser.add_argument("--path", default='', required=True, type=str)
args = parser.parse_args()

CKPT_PATH = os.path.join(args.path, 'model.checkpoint')

count = 0
while True:
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
        checkpoint = {'model': checkpoint['model']}
        count += 1
        torch.save(checkpoint, CKPT_PATH.replace('model.checkpoint','model.checkpoint' + str(count)))
        os.remove(CKPT_PATH)
    time.sleep(10)
