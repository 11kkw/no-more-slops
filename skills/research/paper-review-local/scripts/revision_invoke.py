"""Fresh existing writer session, restricted to staging prose patches through MCP."""
import argparse
import os
from pathlib import Path
import sys
os.environ['HERMES_HOME']='/home/happy5700x/.hermes/profiles/paper'
os.environ['HERMES_ACTIVE_PROFILE']='paper'
sys.path.insert(0,'/home/happy5700x/.hermes/hermes-agent')
from review_mcp import ROOT,safe


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--query-file',required=True);args=parser.parse_args()
    prompt=safe(ROOT,args.query_file).read_text()
    import cli
    images=sorted((ROOT/'week1_clip_dcsm_surgery/paper/shortpaper/renders').glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
    cli._collect_query_images=lambda query,image_arg=None:(query or '',images)
    cli.main(query=prompt,toolsets='mcp-paper_revision_local',quiet=True,max_turns=10,run_budget=300)


if __name__=='__main__':main()
