"""Matched-seed three-policy summary checks."""
import sys,unittest,json,tempfile,subprocess
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/acr"))
from compare_e_mixtures import summarize
from repeat_e import METRICS


class MixtureTests(unittest.TestCase):
    def test_three_policy_means_and_paired_directions(self):
        rows=[]
        for seed,offset in ((1,0),(2,.01),(3,-.01)):
            for method,value in (("natural",.6),("mixture25",.55),("balanced",.5)):
                rows.append(dict(method=method,seed=seed,**{key:value+offset for key in METRICS}))
        result=summarize(rows)
        self.assertEqual(result["n"],3)
        for key in METRICS:
            self.assertAlmostEqual(result["methods"]["mixture25"][key]["mean"],.55)
            self.assertAlmostEqual(result["methods"]["mixture25"][key]["sample_std"],.01)
            self.assertAlmostEqual(result["mixture25_minus_natural"][key]["mean"],-.05)
            self.assertAlmostEqual(result["mixture25_minus_balanced"][key]["mean"],.05)

    def test_notebook_trains_only_two_jobs_then_reuses_them(self):
        repo=Path(__file__).resolve().parents[1]
        nb=json.loads((repo/"notebooks/02e_teacher_e_mixture25_repeat.ipynb").read_text())
        source=next("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"]=="code" and "for seed in plan" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)
            (project/"configs").mkdir()
            for name in ("p2-mixture25-seeds.json","p2-mixture25.json"):
                (project/"configs"/name).write_text((repo/"configs"/name).read_text())

            def execute(args,check):
                config=json.loads(Path(args[args.index("--config")+1]).read_text())
                target=Path(args[args.index("--output")+1]);target.mkdir()
                (target/"run.json").write_text(json.dumps({"config":config,"git_revision":"test"}))
                (target/"metrics.json").write_text("{}")
                (target/"complementarity.csv").write_text("completed")
            scope=dict(project=project,subprocess=subprocess,sys=sys,json=json,
                       torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)))
            with (
                patch("subprocess.check_output",return_value="test"),
                patch("subprocess.run",side_effect=execute) as launch,
                redirect_stdout(StringIO()),
            ):
                exec(source,scope)
                self.assertEqual(launch.call_count,2)
                exec(source,scope)
                self.assertEqual(launch.call_count,2)
                configs=[json.loads(Path(call.args[0][call.args[0].index("--config")+1]).read_text())
                         for call in launch.call_args_list]
                self.assertEqual([c["seed"] for c in configs],[20261120,20261121])
                self.assertTrue(all(c["balanced_probability"]==.25 for c in configs))

    def test_missing_or_duplicated_policy_is_rejected(self):
        rows=[dict(method=m,seed=1,**{key:.5 for key in METRICS})
              for m in ("natural","mixture25","balanced")]
        for invalid in ([],rows[:2],rows+[rows[0]],[rows[0],rows[0],rows[2]]):
            with self.assertRaises(ValueError):summarize(invalid)


if __name__=="__main__":
    unittest.main()
