"""Checks for the event-teacher repeat controller."""
import csv,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/acr"))
from repeat_e import run,METRICS
from repeat_seeds import aggregate


class EventRepeatTests(unittest.TestCase):
    def test_event_pair_delta_direction(self):
        rows=[]
        for seed in (1,2,3):
            rows.append(dict(method="natural",seed=seed,**{k:.6 for k in METRICS}))
            rows.append(dict(method="balanced",seed=seed,**{k:.5 for k in METRICS}))
        result=aggregate(rows,metrics=METRICS,methods=("natural","balanced"))
        self.assertAlmostEqual(result["paired_delta"]["event_accuracy"]["mean"],-.1)
        self.assertEqual(result["paired_delta"]["event_accuracy"]["negative_seeds"],3)

    def test_four_new_jobs_and_reuse(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                Path("configs").mkdir()
                configs={"natural":{"seed":20261119,"sampling":"natural"},
                         "balanced":{"seed":20261119,"sampling":"quality_song_mixture"}}
                for method,cfg in configs.items():
                    Path(f"configs/{method}.json").write_text(json.dumps(cfg))
                plan=dict(seeds=[20261120,20261121],natural_config="configs/natural.json",balanced_config="configs/balanced.json")
                Path("plan.json").write_text(json.dumps(plan))
                ref=Path("reports/p2/f-reference/events.csv")
                ref.parent.mkdir(parents=True)
                ref.write_text("fixed reference")
                digest=hashlib.sha256(ref.read_bytes()).hexdigest()
                qualities=Path("reports/p0/qualities.csv")
                qualities.parent.mkdir(parents=True)
                qualities.write_text("quality,rare\nmaj,False\ndim,True\n")
                metrics=dict(event_accuracy=40/101,macro_quality_event_recall=.2,rare_macro_event_recall=0,
                             scored_events=101,F_reference_sha256=digest)
                counts="quality,metric,both_correct,F_only,E_only,both_wrong\nmaj,complete,30,20,10,40\ndim,complete,0,1,0,0\n"
                for method,path in (("natural","reports/p2/kaggle-natural"),("balanced","reports/p2/kaggle-oracle/full")):
                    base=Path(path);base.mkdir(parents=True)
                    (base/"run.json").write_text(json.dumps({"config":configs[method]}))
                    (base/"metrics.json").write_text(json.dumps(metrics))
                    dest=base.parent if method=="balanced" else base
                    (dest/"complementarity.csv").write_text(counts)

                def execute(args,check):
                    target=Path(args[args.index("--output")+1]);target.mkdir()
                    cfg=json.loads(Path(args[args.index("--config")+1]).read_text())
                    (target/"run.json").write_text(json.dumps({"config":cfg,"git_revision":"test"}))
                    (target/"metrics.json").write_text(json.dumps(metrics))
                    (target/"complementarity.csv").write_text(counts)

                with (
                    patch("repeat_e.subprocess.check_output",return_value="test"),
                    patch("repeat_e.subprocess.run",side_effect=execute) as launch,
                    redirect_stdout(StringIO()),
                ):
                    run(Path("plan.json"),Path("output"),"cpu")
                    self.assertEqual(launch.call_count,4)
                    run(Path("plan.json"),Path("output"),"cpu")
                    self.assertEqual(launch.call_count,4)
                result=json.loads(Path("output/summary.json").read_text())
                self.assertEqual(result["n"],3)
                self.assertEqual(result["delta_direction"],"balanced minus natural")
                with Path("output/per_seed.csv").open() as stream:
                    rows=list(csv.DictReader(stream))
                self.assertEqual(len(rows),6)
                self.assertTrue(all(int(row["rare_reference_events"])==1 for row in rows))
            finally:
                os.chdir(previous)


if __name__=="__main__":
    unittest.main()
