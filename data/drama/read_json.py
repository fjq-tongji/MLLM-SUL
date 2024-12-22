import json

file = 'integrated_v7.json'
file_ = 'drama_captions4eval.json'
file_1 = 'drama_llama_adapter_3_r2.json'
file_2 = './rolisp_dataset_compare/drama_llama_adapter_ROLISP.json'
#with open(file, 'r') as f:
 #   k = json.load(f)
  #  for ll in k:
   #     if ll['id'] == 'titanAclip_807_000180Aframe_000180':
    #        print(ll)


with open(file, 'r') as f:
    k = json.load(f)
    kk = k #['annotations']
    for kkk in kk:
        if kkk['id'] == 'titanAclip_137_000282Aframe_000282':
            print(kkk['Caption_new'])


    print(len(kk))

    for each in k:
        b = each['bbox']
        for p in b:
            if p > 1.0:
                print(each['id'])











