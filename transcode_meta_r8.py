root='/work2/project/trojai-text8/meta-r8-roberta';
out='/work2/project/trojai-fuzzing-nlp/meta-r8-old';

import os
import torch

fnames=os.listdir(root);
#fnames=[os.path.join(root,fname) for fname in fnames];

for i,fname in enumerate(fnames):
    print('%d/%d %s'%(i,len(fnames),fname));
    data=torch.load(os.path.join(root,fname));
    tokens=[x+[1]*(8-len(x)) for x in data['trigger_tokens']];
    loss=torch.Tensor(data['loss']);
    loss=loss.view(loss.shape[0],-1).tolist();
    
    data2={'triggers':tokens,'loss':loss,'insert_locs':data['insert_locs'],'model_id':data['model_id']}
    torch.save(data2,os.path.join(out,fname));