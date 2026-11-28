"""Project I/O for the pinned author BTC, retaining its flat 170-label output."""
import torch
from torch import nn
from torch.nn import functional as F

from acr.vocabulary import QUALITIES
from acr.vendor.btc.btc_model import BTC_model

BTC_QUALITIES=('min','maj','dim','aug','min6','maj6','min7','minmaj7','maj7','7','dim7','hdim7','sus2','sus4')
BTC_UNKNOWN=168
BTC_N=169


def project_to_btc(labels):
    if labels.dtype!=torch.long or not ((labels==-100)|((labels>=0)&(labels<=168))).all():
        raise ValueError('Expected project int64 labels: -100, N=0 or 1..168')
    result=labels.clone();chord=labels>0
    quality=labels.new_tensor([BTC_QUALITIES.index(q) for q in QUALITIES])
    result[chord]=(labels[chord]-1)//14*14+quality[(labels[chord]-1)%14]
    result[labels==0]=BTC_N
    return result


def btc_to_project(labels):
    if labels.dtype!=torch.long or not ((labels>=0)&(labels<170)).all():
        raise ValueError('Expected BTC int64 labels 0..169')
    result=torch.full_like(labels,-100);chord=labels<168
    quality=labels.new_tensor([QUALITIES.index(q) for q in BTC_QUALITIES])
    result[chord]=1+labels[chord]//14*14+quality[labels[chord]%14]
    result[labels==BTC_N]=0
    return result


def btc_loss(logits,project_labels):
    labels=project_to_btc(project_labels)
    if logits.ndim!=3 or logits.shape[1]!=170 or labels.shape!=(logits.shape[0],logits.shape[2]):
        raise ValueError('Expected logits Bx170xT and labels BxT')
    return F.cross_entropy(logits,labels,ignore_index=-100) if (labels!=-100).any() else logits.sum()*0


class BTCFrame(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config['feature_size']!=144 or config['timestep']!=108 or config['num_chords']!=170 or config['probs_out'] is not True:
            raise ValueError('B0 adapter requires 144 bins, 108 frames and 170 raw logits')
        self.model=BTC_model(config)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(144,108) or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Expected finite float CQT Bx144x108')
        # The author forward reshapes labels even when probs_out returns logits only.
        unused=torch.zeros((len(x),108),dtype=torch.long,device=x.device)
        return self.model(x.transpose(1,2),unused).transpose(1,2)


@torch.no_grad()
def predict_frames(model,features):
    if features.ndim!=2 or features.shape[1]!=144 or not len(features):
        raise ValueError('Expected nonempty Tx144 normalized CQT')
    device=next(model.parameters()).device;outputs=[];was_training=model.training
    model.eval()
    try:
        for start in range(0,len(features),108):
            part=torch.as_tensor(features[start:start+108],dtype=torch.float32,device=device)
            x=torch.zeros((1,144,108),device=device);x[0,:,:len(part)]=part.T
            outputs.append(model(x)[0,:,:len(part)].cpu())
    finally:model.train(was_training)
    return torch.cat(outputs,dim=1)
