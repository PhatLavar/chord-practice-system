import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/acr'))
from btc_adapter import BTCFrame,btc_loss,project_to_btc,btc_to_project,predict_frames
from vendor.btc.utils.transformer_modules import MultiHeadAttention,_gen_bias_mask,LayerNorm


class BTCCompatibility(unittest.TestCase):
    def test_label_roundtrip_unknown_and_no_chord(self):
        labels=torch.arange(169,dtype=torch.long)
        self.assertTrue(torch.equal(btc_to_project(project_to_btc(labels)),labels))
        self.assertEqual(project_to_btc(torch.tensor([-100,0,1,2])).tolist(),[-100,169,1,0])
        self.assertEqual(btc_to_project(torch.tensor([168,169])).tolist(),[-100,0])
        with self.assertRaises(ValueError):project_to_btc(torch.tensor([169]))
        with self.assertRaises(ValueError):btc_to_project(torch.tensor([-100]))
        with self.assertRaises(ValueError):project_to_btc(torch.tensor([1.]))

    def test_ignored_target_loss(self):
        torch.manual_seed(1);logits=torch.randn(1,170,3,requires_grad=True)
        loss=btc_loss(logits,torch.tensor([[1,-100,0]]))
        expected=nn.functional.cross_entropy(logits[:,:,[0,2]],torch.tensor([[1,169]]))
        torch.testing.assert_close(loss,expected);loss.backward()
        self.assertTrue((logits.grad[:,:,1]==0).all())
        self.assertEqual(float(btc_loss(logits,torch.full((1,3),-100)).detach()),0.)

    def test_author_model_forward_backward_and_vendor_identity(self):
        manifest=json.loads(Path('src/acr/vendor/btc/UPSTREAM.json').read_text(encoding='utf-8'))
        for name,record in manifest['files'].items():
            raw=(Path('src/acr/vendor/btc')/name).read_bytes().replace(b'\r\n',b'\n')
            self.assertEqual(hashlib.sha256(raw).hexdigest(),record['vendored_sha256_lf'])
        cfg=json.loads(Path('configs/baselines/btc-b0.json').read_text())['model']
        torch.manual_seed(4);model=BTCFrame(cfg)
        logits=model(torch.randn(1,144,108))
        self.assertEqual(tuple(logits.shape),(1,170,108))
        loss=btc_loss(logits,torch.ones((1,108),dtype=torch.long));loss.backward()
        grad=model.model.self_attn_layers.embedding_proj.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
        with self.assertRaises(ValueError):model(torch.zeros(1,144,109))

    def test_tiling_preserves_real_frame_count_and_order(self):
        class Echo(nn.Module):
            def __init__(self):
                super().__init__();self.weight=nn.Parameter(torch.zeros(()))
            def forward(self,x):return x[:,:1,:].expand(-1,170,-1)
        features=np.zeros((109,144),dtype=np.float32);features[:,0]=np.arange(109)
        model=Echo();result=predict_frames(model,features)
        self.assertEqual(tuple(result.shape),(170,109))
        torch.testing.assert_close(result[0],torch.arange(109,dtype=torch.float32))
        self.assertTrue(model.training)

    def test_author_attention_directions_and_sample_std_norm(self):
        mask=_gen_bias_mask(4);x=torch.randn(1,4,8)
        for reverse in (False,True):
            bias=mask.transpose(2,3) if reverse else mask
            attention=MultiHeadAttention(8,8,8,8,2,bias,0.,True)
            _,weights=attention(x,x,x)
            forbidden=torch.tril(torch.ones(4,4,dtype=torch.bool),-1) if reverse else torch.triu(torch.ones(4,4,dtype=torch.bool),1)
            self.assertTrue((weights[:,:,forbidden]==0).all())
        norm=LayerNorm(8)
        torch.testing.assert_close(norm(x),(x-x.mean(-1,keepdim=True))/(x.std(-1,keepdim=True)+1e-6))
