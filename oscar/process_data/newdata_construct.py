import json
import matplotlib.pyplot as plt  
import os
import numpy as np
import sng_parser
from pprint import pprint
from tqdm import tqdm 
import spacy
nlp = spacy.load("en_core_web_sm")
IMAGE_ROOT = '../datasets/images/'

def read_json(file):
    datas = []
    with open(file, 'r', encoding='utf-8') as fin:
        datas = json.load(fin)
    return datas

def show_img(imgid, IMAGE_ROOT):
    img_path = IMAGE_ROOT
    split = 'val'
    img_path = os.path.join(img_path, split+'2014', 'COCO_val2014_'+'0'*(12-len(imgid))+imgid+'.jpg')
    print(img_path)
    # show img
    img = plt.imread(img_path)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()

def match(robj, objs):
    for gobj in objs:
        if (robj in gobj) or (gobj in robj):
            return gobj
    return None

def deledup(rels):
    for k,v in rels.items():
        rels[k] = []
        for vi in v:
            if vi not in rels[k]:
                rels[k].append(vi)
    new_rels = {}
    for k,v in rels.items():
        if len(v) > 0:
            new_rels[k] = v
    return new_rels

def rfilter(rels, cap):
    new_rels = {}
    for k, v in rels.items():
        new_v = []
        for vi in v:
            if vi[1]+' '+vi[2] in ['in front']:
                continue
            if vi[0] in ['that']:
                continue
            if (vi[2] in cap) or (vi[2] in ['it']):
                continue
            new_v.append(vi)
        if len(new_v) > 0:
            new_rels[k] = new_v
    return new_rels

def afilter(attrs):
    new_attrs = {}
    for k, v in attrs.items():
        new_v = []
        for vi in v:
            if vi[0] in ['other','several']:
                continue
            new_v.append(vi)
        if len(new_v) > 0:
            new_attrs[k] = new_v
    return new_attrs
    

if __name__=="__main__":
    # read caption json
    cap_path = '../datasets/annotation/ref_captions.json'
    with open(cap_path, 'r', encoding='utf-8') as fin:
        all_captions = json.load(fin)
     
    # get split image ids
    split_root = '../datasets/annotation/'
    img_ids = []
    for split in ['trn','val','tst']:
        split_path = split_root + '{}_names.npy'.format(split)
        img_ids.extend(np.load(split_path))
    # get caption set
    captions = {}
    id2name = {}
    for k in img_ids:
        new_k = str(int(k.split('_')[-1].split('.jpg')[0]))
        captions[new_k] = all_captions[k]
        id2name[new_k] = k

    # load scene graph
    path = '../datasets/ref_cap2sng.json'
    cap2sg = read_json(path)
    '''
    sample in cap2sg as follows:
    key: image id 487008 
    value:
    {
        'a room with a couch chair and television':
        {
            'objects': ['room', 'chair', 'television'],
            'relations': [('room', 'with', 'television'), ('room', 'with', 'chair')],
            'attributes': {'room': [], 'chair': ['couch'], 'television': []}, 
        }
    }
    '''
    elements = {}
    example = 0
    for imgid, caps in tqdm(captions.items()):
        cap_lens = [len(cap) for cap in caps]
        general_cap = '' # find the shortest cap as the general cap
        for cap in caps:
            if len(cap) == min(cap_lens):
                general_cap = cap
                break
        rest_caps = caps.copy()
        rest_caps.remove(general_cap)    
        
        new_cap = ''
        sgs = cap2sg[imgid]
        gsg = sgs[general_cap]
        rels = {}
        attrs = {}
        for rcap in rest_caps:
            rsg = sgs[rcap]
            robj = rsg['objects']
            rrel = rsg['relations']
            rattr = rsg['attributes']

            robj2gobj = {}
            for obj in robj:
                gobj = match(obj,gsg['objects'])
                robj2gobj[obj] = gobj 
            # extend attr
            if len(rattr) > 0:
                for obj in robj:
                    if obj not in rattr.keys():
                        continue
                    gobj = robj2gobj[obj]
                    if gobj is not None:
                        if gobj not in attrs.keys():
                            attrs[gobj] = []
                        for new_attr in rattr[obj]: 
                            if (gobj not in gsg['attributes'].keys()) or (new_attr not in gsg['attributes'][gobj]):
                                attrs[gobj].append(new_attr)
                        
            # extend rel
            if len(rrel) > 0:
                for rel in rrel:
                    gobj = robj2gobj[rel[0]]
                    if gobj is not None:
                        if gobj not in rels.keys():
                            rels[gobj] = []
                        
                        new_rel = rel.copy()
                        new_rel[0] = gobj
                        if new_rel not in gsg['relations']:
                            rels[gobj].append(new_rel)
        # Remove duplicate information
        rels = deledup(rels)
        attrs = deledup(attrs)
        
        # Filter invalid information
        rels = rfilter(rels, general_cap)
        attrs = afilter(attrs)
        
        elements[imgid] = {}
        elements[imgid]['cap'] = general_cap
        elements[imgid]['rels'] = rels
        elements[imgid]['attrs'] = attrs
        example += 1


    # Based on the resulting REL and ATTR, to generate new-format data:
    # e.g. A couple of traffic lights on a pole with a blue sky. The traffic light is red .
    prmpts = {}
    morph2verb = {'Sing': ' is ', 'Plur': ' are '}
    count = 0
    for imgid, v in tqdm(elements.items()):
        raw_cap = v['cap']
        rels = v['rels']
        attrs = v['attrs']
        new_parts = []
        new_attrs = []
        new_rels = []
        for robj, rvs in rels.items():
            for r in rvs:
                sent = ' the ' + ' '.join(r) + '.' 
                new_parts.append(sent)
                new_rels.append(sent)
        for aobj, avs in attrs.items():
            morph = nlp(aobj)[-1].morph.get("Number")
            verb = morph2verb[morph[0]] if len(morph) > 0 else ' is '
            for a in avs:
                if a[1] == 'nummod':
                    sent =  ' the number of ' + aobj + verb + a[0]
                else:
                    sent = ' the ' + aobj + verb + a[0]
                new_parts.append(sent)
                new_attrs.append(sent)
        new_caps = [raw_cap + '.' + sent for sent in new_parts]
        new_attrs = [raw_cap + '.' + sent for sent in new_attrs]
        new_rels = [raw_cap + '.' + sent for sent in new_rels]
        
        prmpts[imgid] = {}
        prmpts[imgid]['raw_cap'] = raw_cap
        prmpts[imgid]['new_caps'] = new_caps
        prmpts[imgid]['new_attrs'] = new_attrs
        prmpts[imgid]['new_rels'] = new_rels
        count += len(new_caps)
        if len(new_caps) == 0:
            continue
    

    # convert original train.json/val.json/test.json
    root = '../datasets/coco_caption/'
    new_datas = {}
    id = 0
    for split in ['train','val']:
        path = os.path.join(root, split+'_caption.json')
        datas = json.load(open(path,'r'))
        prmpt_datas = []
        unchanged_caps = []
        raw_size = len(datas)
        imgids = set()
        for term in tqdm(datas):
            imgids.add(term['image_id'])
        imgids = list(imgids)
        for imgid in tqdm(imgids):
            new_caps = prmpts[imgid]['new_caps']
            if len(new_caps) > 0:
                for new_cap in new_caps:
                    prmpt = {}
                    prmpt['image_id'] = imgid
                    prmpt['id'] = id
                    id += 1
                    prmpt['caption'] = new_cap
                    prmpt_datas.append(prmpt)

        new_datas[split] = prmpt_datas
        data_size = len(prmpt_datas)   
        print("==>{}<==".format(split))
        print("raw data sample size:", len(datas))
        print("new data sample size:", data_size)
        print("unchanged image id num:", len(unchanged_caps))
        

        # save train/val/test_adj_prompts.json
        out_path = '../datasets/'+split+'_prefix_prompts.json'
        with open(out_path, 'w') as fout:
            json.dump(new_datas[split], fout)

        
        