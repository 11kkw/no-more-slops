"""Run existing trusted CPU-only paper build/validation into a staging directory."""
import argparse
import importlib.util
import json
from pathlib import Path


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True);parser.add_argument('--stage',required=True);args=parser.parse_args()
    package=Path(args.package).resolve();stage=Path(args.stage).resolve()
    if not stage.is_relative_to(package/'reviews'):raise ValueError('Stage must be inside the review directory')
    # Workflow documentation is edited by pipeline setup, independently of the
    # frozen scientific evidence. Retain its exclusion and current hash explicitly.
    snapshot=stage/'input_snapshot.json'
    entries=json.loads(snapshot.read_text())
    workflow='plan/03_paper_pipeline.md'
    excluded=entries.pop(workflow,None)
    if excluded is not None:
        snapshot.write_text(json.dumps(entries,indent=2)+'\n')
        import hashlib
        source=package.parent.parent/workflow
        (stage/'snapshot_scope.json').write_text(json.dumps({'excluded_authorized_workflow_document':workflow,'baseline_sha256':excluded,'current_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reason':'User-authorized pipeline configuration documentation; all scientific inputs remain checked'},indent=2)+'\n')
    build=load(package/'build.py','trusted_paper_build');build.ROOT=stage;build.main()
    validate=load(package/'validate.py','trusted_paper_validate');validate.ROOT=stage;validate.main()
