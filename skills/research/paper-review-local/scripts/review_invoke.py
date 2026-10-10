"""Use existing Hermes CLI/auth with native page attachments, no upstream changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HERMES_HOME']='/home/happy5700x/.hermes/profiles/paper-reviewer'
os.environ['HERMES_ACTIVE_PROFILE']='paper-reviewer'
sys.path.insert(0,'/home/happy5700x/.hermes/hermes-agent')
from review_mcp import ROOT, safe


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--query-file',required=True);parser.add_argument('--package',required=True)
    args=parser.parse_args()
    package=safe(ROOT,args.package)
    prompt_path=safe(ROOT,args.query_file)
    images=sorted((package/'renders').glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
    if not images or len(images)>30:raise ValueError('Native PDF page images are required; maximum 30')
    for image in images:
        safe(package,image.relative_to(package))
        if image.stat().st_size>10_000_000:raise ValueError('Page image exceeds size limit')
    import cli
    from hermes_cli.config import load_config
    from agent.image_routing import decide_image_input_mode,build_native_content_parts
    config=load_config();model=config['model']
    mode=decide_image_input_mode(model['provider'],model['default'],config,requested_provider=model['provider'])
    if mode!='native':raise RuntimeError('Native image input unavailable; refuse to claim visual review')
    prompt=prompt_path.read_text(encoding='utf-8')
    parts,skipped=build_native_content_parts(prompt,[str(p) for p in images])
    if skipped or sum(p.get('type')=='image_url' for p in parts)!=len(images):raise RuntimeError('Not all rendered pages attached')
    manifest={'image_input_mode':mode,'attached_pages':len(images),'pages':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in images]}
    (prompt_path.parent/'native_image_inputs.json').write_text(json.dumps(manifest,indent=2))
    # Scoped wrapper around Hermes' image collection; no installed runtime edit.
    cli._collect_query_images=lambda query,image_arg=None:(query or '',images)
    cli.main(query=prompt,toolsets='mcp-paper_review_local',skills='paper-review-local',quiet=True,max_turns=24,run_budget=480)


if __name__=='__main__':main()
