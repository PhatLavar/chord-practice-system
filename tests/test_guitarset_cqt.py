import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/acr'))
try:
    from prepare_guitarset_cqt import extract, frame_labels, normalization, selected_records
    from train_guitarset_cqt import timeline
except ModuleNotFoundError as error:
    if error.name not in {'librosa','soundfile','mir_eval'}:raise
    extract=None
from teacher_f import FrameTeacher, losses
from teacher_e import EventTeacher


@unittest.skipIf(extract is None,'Install requirements-audio.txt')
class GuitarSetCQT(unittest.TestCase):
    def test_tone_and_frame_clock(self):
        cfg=json.loads(Path('configs/p2-guitarset-cqt.json').read_text())
        sr=44100;t=np.arange(3*sr)/sr
        x,times,duration=extract(np.sin(2*np.pi*440*t).astype(np.float32),sr,cfg)
        self.assertEqual(x.shape[1],144)
        self.assertLessEqual(abs(int(x.mean(0).argmax())-90),1)
        self.assertAlmostEqual(times[1]-times[0],2048/22050)
        self.assertLess(times[-1],duration)

    def test_boundaries_and_oov_mask(self):
        y=frame_labels(np.array([0,.5,1,1.5,2]),[
            dict(time=0,duration=1,value='C:maj'),dict(time=1,duration=1,value='C:(1,5)')])
        self.assertTrue((y[:2]>0).all());self.assertTrue((y[2:]==-100).all())

    def test_normalization_train_only(self):
        songs={'a':{'x':np.array([[1.,2.],[3.,4.]])},'b':{'x':np.array([[100.,200.]])}}
        roles={'a':'train','b':'validation'};first=normalization(songs,roles)
        songs['b']['x']*=100
        self.assertEqual(first,normalization(songs,roles));self.assertEqual(first['mean'],2.5)

    def test_no_test_recordings_selected(self):
        import csv
        with open('reports/p2/guitarset-labels/folds.csv') as f:rows=list(csv.DictReader(f))
        test={r['id'] for r in rows if r['fold']=='0' and r['partition']=='test'}
        self.assertFalse(test & selected_records(0).keys())
        self.assertEqual(len(selected_records(0,True)),4)

    def test_input_backward_compatibility_and_gradients(self):
        torch.manual_seed(9);a=FrameTeacher()
        torch.manual_seed(9);b=FrameTeacher(input_channels=24)
        self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()))
        for cls in [FrameTeacher,EventTeacher]:
            m=cls(input_channels=144);x=torch.randn(2,144,80)
            h=m(x,torch.ones(2,80,dtype=torch.bool)) if cls==EventTeacher else m(x)
            y=torch.ones((2,h[0].shape[-1]),dtype=torch.long)
            loss=sum(losses(h,y));loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.input.weight.grad.abs().sum()),0)

    def test_explicit_unknown_gaps(self):
        intervals,labels=timeline([dict(time=1,duration=1)],3,['C:maj'],'X')
        self.assertEqual(intervals,[[0.,1.],[1.,2.],[2.,3]])
        self.assertEqual(labels,['X','C:maj','X'])
        intervals,_=timeline([dict(time=0,duration=1.00001)],1,['C:maj'],'X')
        self.assertEqual(intervals,[[0.,1]])
        with self.assertRaises(ValueError):timeline([dict(time=0,duration=1.01)],1,['C:maj'],'X')
        intervals,labels=timeline([dict(time=0,duration=1)],1.00000000001,['C:maj'],'X')
        for a,b in intervals:
            self.assertGreater(float(f'{b:.17g}'),float(f'{a:.17g}'))

    def test_frame_context_matches_training_window(self):
        from train_f import Windows
        torch.manual_seed(3);model=FrameTeacher(input_channels=144).eval()
        song={'x':np.random.default_rng(3).normal(size=(90,144)).astype(np.float32),'y':np.ones(90,dtype=np.int64)}
        x,_=Windows({'a':song},108,model.context)[0]
        full=torch.nn.functional.pad(torch.from_numpy(song['x'].T.copy()).unsqueeze(0),(31,31))
        with torch.no_grad():
            a=model(x.unsqueeze(0));b=model(full)
        for left,right in zip(a,b):
            torch.testing.assert_close(left[:,:,31:121],right[:,:,31:121],rtol=1e-5,atol=1e-6)
