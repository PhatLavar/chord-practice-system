import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

try:
    from acr.data.guitarset.cqt import extract, frame_labels, normalization, selected_records
    from acr.evaluation.guitarset import timeline
except ModuleNotFoundError as error:
    if error.name not in {'librosa','soundfile','mir_eval'}:raise
    extract=None
from acr.models.tcn import FrameTeacher
from acr.models.components import losses
from acr.models.tcn import EventTeacher


@unittest.skipIf(extract is None,'Install requirements-audio.txt')
class GuitarSetCQT(unittest.TestCase):
    def test_repeat_seed_can_reuse_only_compatible_cache(self):
        from acr.training.guitarset import resolve_cache_config
        original=json.loads(Path('configs/p2/p2-guitarset-cqt-player.json').read_text())
        cfg=dict(original,seed=20261120,cache_config='configs/p2/p2-guitarset-cqt-player.json')
        self.assertEqual(resolve_cache_config(cfg),original)
        with self.assertRaises(ValueError):resolve_cache_config(dict(cfg,fold=1))
        with self.assertRaises(ValueError):resolve_cache_config(dict(cfg,learning_rate=.01))

    def test_repeat_aggregate_uses_paired_deltas_and_sample_std(self):
        from acr.experiments.guitarset.repeat import aggregate
        rows=[]
        for teacher in ['F','E']:
            for seed,a,b in [(1,.1,.3),(2,.3,.2)]:
                rows.append(dict(teacher=teacher,seed=seed,
                    natural={k:a for k in ['root','majmin','triads','tetrads','event_accuracy']},
                    transpose={k:b for k in ['root','majmin','triads','tetrads','event_accuracy']}))
        result=aggregate(rows)
        for teacher in ['F','E']:
            self.assertAlmostEqual(result[teacher]['tetrads']['natural']['mean'],.2)
            self.assertAlmostEqual(result[teacher]['tetrads']['paired_delta']['mean'],.05)
            self.assertAlmostEqual(result[teacher]['tetrads']['natural']['sample_std'],np.sqrt(.02))

    def test_cqt_transposition_labels_padding_and_no_wrap(self):
        from acr.data.augmentation import transpose_batch
        x=torch.arange(1,25,dtype=torch.float32).reshape(1,8,3)
        x[:,:,0]=0
        y=torch.tensor([[155,0,-100]])  # B:maj -> C:maj when shifted up one.
        up,label=transpose_batch(x,y,[1],2,-9.)
        self.assertEqual(label.tolist(),[[1,0,-100]])
        torch.testing.assert_close(up[:,2:,1:],x[:,:-2,1:])
        self.assertTrue((up[:,:2,1:]==-9).all())
        self.assertTrue((up[:,:,0]==0).all())
        down,label=transpose_batch(x,torch.tensor([[1]]),[-1],2,-9.)
        self.assertEqual(label.item(),155)
        torch.testing.assert_close(down[:,:-2,1:],x[:,2:,1:])
        self.assertTrue((down[:,-2:,1:]==-9).all())
        same,labels=transpose_batch(x,y,[0],2,-9.)
        torch.testing.assert_close(same,x);torch.testing.assert_close(labels,y)
        for shift in range(-5,7):
            original=torch.arange(1,169)[None,:]
            _,mapped=transpose_batch(torch.ones(1,144,1),original,[shift],2,-9.)
            self.assertTrue(torch.equal((mapped-1)%14,(original-1)%14))
            self.assertTrue(torch.equal((mapped-1)//14,((original-1)//14+shift)%12))

    def test_component_audit_explicit_references_without_audio(self):
        import tempfile
        from contextlib import redirect_stdout
        import io
        from acr.evaluation.guitarset import run
        from acr.models.components import targets
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);refs=root/'training_references.json'
            refs.write_text(json.dumps({'a':{'duration':1.,'events':[{'time':0.,'duration':1.,'value':'C:maj7'}]}}))
            values=targets(torch.tensor([11]))
            heads=[]
            for size,target in zip([2,12,6,5],values):
                h=np.full((size,2),-10,dtype=np.float32);h[int(target.item())]=10;heads.append(h)
            for teacher in ['F','E']:
                directory=root/teacher/'predictions';directory.mkdir(parents=True)
                (root/teacher/'metrics.json').write_text(json.dumps({'supported_events':1,'supported_oracle_event_accuracy':1.}))
                if teacher=='F':np.savez(directory/'a.npz',times=[0.,.5],**{f'logits_{j}':h for j,h in enumerate(heads)})
                else:np.savez(directory/'event_logits.npz',**{f'logits_{j}':h[:,0][None,:] for j,h in enumerate(heads)})
            with redirect_stdout(io.StringIO()):run(root,root/'audit',refs)
            result=json.loads((root/'audit/summary.json').read_text())
            for teacher in ['F','E']:
                self.assertEqual(result[teacher]['raw_all_heads_correct'],1)
                self.assertEqual(result[teacher]['extended_predicted_none'],0)
            self.assertFalse((root/'audit/review_candidates.csv').exists())

    def test_player_split(self):
        import csv
        manifest=Path('splits/guitarset-player-v1.csv')
        with manifest.open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len({r['id'] for r in rows}),360)
        expected={'train':{'00','01','02','03'},'validation':{'05'},'test':{'04'}}
        for part,players in expected.items():
            subset=[r for r in rows if r['partition']==part]
            self.assertEqual({r['player'] for r in subset},players)
            self.assertEqual({r['progression'] for r in subset},{'0','1','2'})
        self.assertEqual(sum(r['partition']=='excluded' for r in rows),2)
        selected=selected_records(0,False,manifest)
        self.assertEqual(len(selected),300)
        self.assertFalse(any(rid.startswith('04_') for rid in selected))
        self.assertNotEqual(selected,selected_records(0))

    def test_memorization_selection(self):
        from acr.experiments.guitarset.memorization import select_events
        events=[('a',0,1,0,5,1),('a',1,2,5,11,1),
                ('b',0,1,0,8,15),('c',0,1,0,6,2),
                ('d',0,1,0,8,-100),('e',0,1,0,8,0),('f',0,1,0,200,3)]
        selected=select_events(events)
        self.assertEqual(selected,[events[0],events[2],events[3]])
        self.assertEqual(selected,select_events(list(reversed(events))))
        with self.assertRaises(ValueError):select_events(events[4:])

    def test_tone_and_frame_clock(self):
        cfg=json.loads(Path('configs/p2/p2-guitarset-cqt.json').read_text())
        sr=44100;t=np.arange(3*sr)/sr
        x,times,duration=extract(np.sin(2*np.pi*440*t).astype(np.float32),sr,cfg)
        self.assertEqual(x.shape[1],144)
        self.assertLessEqual(abs(int(x.mean(0).argmax())-90),1)
        from acr.data.augmentation import transpose_batch
        shifted,_=transpose_batch(torch.from_numpy(x.T.copy())[None],torch.tensor([[1]]),[1],2,float(np.log(1e-6)))
        self.assertEqual(int(shifted[0].mean(1).argmax()),int(x.mean(0).argmax())+2)
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
        from acr.data.windows import Windows
        torch.manual_seed(3);model=FrameTeacher(input_channels=144).eval()
        song={'x':np.random.default_rng(3).normal(size=(90,144)).astype(np.float32),'y':np.ones(90,dtype=np.int64)}
        x,_=Windows({'a':song},108,model.context)[0]
        full=torch.nn.functional.pad(torch.from_numpy(song['x'].T.copy()).unsqueeze(0),(31,31))
        with torch.no_grad():
            a=model(x.unsqueeze(0));b=model(full)
        for left,right in zip(a,b):
            torch.testing.assert_close(left[:,:,31:121],right[:,:,31:121],rtol=1e-5,atol=1e-6)
