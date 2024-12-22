import json
import os
import numpy as np
import time
json_file = "captions_val2014.json"
num = []

with open(json_file, "r", encoding="utf-8") as f:
    file = json.load(f)
    for key in file:
        print(key)
    print(file[0])
    print(len(file['annotations']))
    for j in file['annotations']:
        if int(j['image_id']) == 358765:
            print(j['caption'])
     #   if int(j['image_id']) == 358765:
      #      print(j)

