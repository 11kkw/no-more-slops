"""Bounded writer/reviewer handoff. Never dispatch experiments or react to bot chat."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from dotenv import dotenv_values
from review_mcp import ROOT,safe,sha
from revision_mcp import context,validate_patches
import publish_review as publisher

PACKAGE=ROOT/'week1_clip_dcsm_surgery/paper/shortpaper'
REVIEWS=PACKAGE/'reviews'
PYTHON='/home/happy5700x/.hermes/hermes-agent/venv/bin/python'


def stamp():return datetime.now(timezone.utc).isoformat()


def write(path,data):publisher.write(path,data)


def notice(job,event,body,writer=False,files=None):
    receipt=job/(event+'_discord.json')
    previous=json.loads(receipt.read_text()) if receipt.exists() else {}
    if previous.get('status')=='sent':return previous
    profile='/home/happy5700x/.hermes/.env' if writer else '/home/happy5700x/.hermes/profiles/paper-reviewer/.env'
    token=dotenv_values(profile).get('DISCORD_BOT_TOKEN')
    bot=publisher.request('GET','/users/@me',token)
    expected='1503942820795388044' if writer else '1558443383176044556'
    if bot['id']!=expected:raise RuntimeError('Unexpected handoff sender')
    channel=publisher.request('GET','/channels/'+publisher.CHANNEL,token)
    if channel.get('guild_id')!=publisher.GUILD:raise RuntimeError('Unexpected handoff channel')
    marker='paper-feedback:'+job.parent.name+':'+event
    history=publisher.request('GET','/channels/'+publisher.CHANNEL+'/messages?limit=100',token)
    found=next((m for m in history if publisher.has_marker(m,marker,bot['id'])),None)
    if found:message=found
    elif previous.get('status') in ['sending','uncertain']:return previous
    else:
        payload={'content':body[:1900],'allowed_mentions':{'parse':[]},'nonce':hashlib.sha256(marker.encode()).hexdigest()[:24],'enforce_nonce':True,
                 'embeds':[{'description':'집필 ↔ 독립 검토 · 최대 두 번 수정 · 새 실험 없음','footer':{'text':marker}}]}
        if files:
            boundary='paper-feedback-'+uuid.uuid4().hex
            if sum(len(data) for _,data in files)>10000000:raise RuntimeError('Handoff attachments exceed limit')
            payload['attachments']=[{'id':i,'filename':name} for i,(name,data) in enumerate(files)]
            parts=[('--'+boundary+'\r\nContent-Disposition: form-data; name="payload_json"\r\n\r\n').encode(),json.dumps(payload,ensure_ascii=False).encode(),b'\r\n']
            for i,(name,data) in enumerate(files):
                parts.extend([('--'+boundary+'\r\nContent-Disposition: form-data; name="files['+str(i)+']"; filename="'+name+'"\r\nContent-Type: application/octet-stream\r\n\r\n').encode(),data,b'\r\n'])
            parts.append(('--'+boundary+'--\r\n').encode());payload=b''.join(parts);kind='multipart/form-data; boundary='+boundary
        else:kind='application/json'
        write(receipt,{'status':'sending','at':stamp()})
        message=publisher.request('POST','/channels/'+publisher.CHANNEL+'/messages',token,payload,kind)
    verified=publisher.request('GET','/channels/'+publisher.CHANNEL+'/messages/'+message['id'],token)
    if not publisher.has_marker(verified,marker,bot['id']):raise RuntimeError('Handoff message not verified')
    result={'status':'sent','message_id':message['id'],'sender_id':bot['id'],'message_url':f"https://discord.com/channels/{publisher.GUILD}/{publisher.CHANNEL}/{message['id']}",'at':stamp()}
    write(receipt,result);return result


def science_snapshot():
    paths=[]
    for relative in ['week1_clip_dcsm_surgery/paper/evidence','week1_clip_dcsm_surgery/experiment/results']:
        directory=ROOT/relative
        if directory.exists():paths.extend(p for p in directory.rglob('*') if p.is_file())
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}


def apply_revision(job,checks):
    proposal=json.loads((job/'revision_proposal.json').read_text())
    if proposal['status']=='no_safe_changes':return {'status':'no_safe_changes'}
    _,current,editorial=context(proposal['review_id'])
    if current['input_digest']!=checks['input_digest']:raise RuntimeError('Stale writer proposal')
    expected=validate_patches((PACKAGE/'shortpaper.md').read_text(),proposal['patches'],{f['id'] for f in editorial['findings']},(PACKAGE/'references.bib').read_text())
    if (job/'proposed_shortpaper.md').read_text()!=expected:raise RuntimeError('Staged manuscript does not match the validated proposal')
    before=science_snapshot()
    stage=job/('candidate-'+uuid.uuid4().hex[:12]);stage.mkdir(exist_ok=False)
    for path in PACKAGE.rglob('*'):
        relative=path.relative_to(PACKAGE)
        if any(part in ['reviews','__pycache__'] for part in relative.parts) or not path.is_file():continue
        safe(PACKAGE,relative);destination=stage/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    shutil.copy2(job/'proposed_shortpaper.md',stage/'shortpaper.md')
    command=['/home/happy5700x/.local/bin/uv','run','--offline','--no-project','--python','3.10','--with','pypandoc_binary','--with','matplotlib==3.10.7','--with','pymupdf',
             'python',str(Path(__file__).with_name('build_revision.py')),'--package',str(PACKAGE),'--stage',str(stage)]
    with (job/'build.log').open('w') as output:
        process=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT,timeout=240)
    if process.returncode:raise RuntimeError('Candidate CPU compilation or evidence validation failed; originals unchanged')
    validation=json.loads((stage/'build_validation.json').read_text())
    if validation['status'] not in ['passed','passed_with_warnings']:raise RuntimeError('Candidate validation failed')
    if sha(stage/'references.bib')!=sha(PACKAGE/'references.bib'):raise RuntimeError('Verified bibliography changed during build')
    from review_mcp import latex_lint
    lint=latex_lint(stage/'shortpaper.tex');write(job/'candidate_lint.json',lint)
    if lint['status'] not in ['passed','warnings_found']:raise RuntimeError('Required candidate linter failed')
    validation['mandatory_linter']={'status':lint['status'],'report':str(job/'candidate_lint.json')}
    validation['not_verified']=[item for item in validation.get('not_verified',[]) if item!='chktex unavailable']
    if (stage/'snapshot_scope.json').exists():validation['snapshot_scope']=json.loads((stage/'snapshot_scope.json').read_text())
    write(stage/'build_validation.json',validation)
    if science_snapshot()!=before:raise RuntimeError('Protected scientific evidence changed')
    context(proposal['review_id'])  # Reject concurrent author edits immediately before applying.
    eligible={'shortpaper.md','shortpaper.tex','shortpaper.pdf','shortpaper.log','shortpaper.out','shortpaper.aux','shortpaper.bbl','shortpaper.blg',
              'compile_stdout.txt','build_validation.json','pdf_text.txt','figure_data.json','figures/spatial_balanced.pdf','figures/spatial_balanced.png'}
    eligible.update(str(path.relative_to(stage)) for path in (stage/'renders').glob('page-*.png'))
    backup=job/'before';backup.mkdir(exist_ok=True)
    replacements=[]
    for relative in sorted(eligible):
        candidate=stage/relative;target=safe(PACKAGE,relative)
        if candidate.exists() and (not target.exists() or sha(candidate)!=sha(target)):
            saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():shutil.copy2(target,saved)
            replacements.append((relative,candidate,target,saved,target.exists()))
    manifest={'review_id':proposal['review_id'],'status':'applying','at':stamp(),'files':[{'path':relative,'before_sha256':sha(saved) if existed else None,'after_sha256':sha(candidate)} for relative,candidate,target,saved,existed in replacements],
              'protected_scientific_files':len(before),'numeric_evidence_preserved':True}
    write(job/'apply_state.json',manifest)
    try:
        for relative,candidate,target,saved,existed in replacements:
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.revision.tmp');shutil.copy2(candidate,temp);temp.replace(target)
    except Exception:
        for relative,candidate,target,saved,existed in replacements:
            if existed:shutil.copy2(saved,target)
        raise
    manifest.update(status='applied',at=stamp(),validation_status=validation['status'],candidate_linter_status=lint['status']);write(job/'apply_state.json',manifest)
    return manifest


def finish_staged(job,checks,state):
    proposal=json.loads((job/'revision_proposal.json').read_text())
    result=apply_revision(job,checks)
    counts=Counter(r['status'] for r in proposal['responses'])
    feedback=['# 집필 에이전트의 검토 응답','',proposal['summary'],'']
    for response in proposal['responses']:feedback.extend(['## '+response['finding_id']+' — '+response['status'],'',response['explanation'],'','근거: '+response['evidence'],''])
    response_file=job/'writer_response.ko.md';response_file.write_text('\n'.join(feedback))
    attachments=[('writer-response-'+proposal['review_id']+'.md',response_file.read_bytes())]
    if result['status']=='applied':attachments.append(('revised-shortpaper.pdf',(PACKAGE/'shortpaper.pdf').read_bytes()))
    msg=notice(job,'writer-response','**[논문 집필 에이전트] 피드백 응답 완료**\n'+proposal['summary'][:800]+'\n'+f'수정안 {proposal["patch_count"]}개 · 응답 상태 {dict(counts)}\n'+
               ('CPU 문서 빌드·기존 수치 검증·필수 린터를 통과한 수정본과 항목별 응답을 첨부합니다. Paper Reviewer가 수정본을 다시 검토합니다.' if result['status']=='applied' else '추가로 안전하게 반영할 수정이 없어 원고를 유지했습니다. 미해결 항목은 저자 확인이 필요합니다.'),writer=True,files=attachments)
    state.update(status='awaiting_recheck' if result['status']=='applied' else 'author_action_required',at=stamp(),patch_count=proposal['patch_count'],response_counts=dict(counts),writer_message_url=msg.get('message_url'),apply_result=result)
    if result['status']=='applied':
        write(REVIEWS/'latest_writer_response.json',{'review_id':proposal['review_id'],'input_digest':checks['input_digest'],'responses':proposal['responses'],'summary':proposal['summary'],'apply_state':str(job/'apply_state.json')})
    state.pop('reason',None);state.pop('error_type',None)
    write(job/'job_state.json',state);write(REVIEWS/'feedback_loop_state.json',state)
    return state


def tick():
    gate_path=REVIEWS/'gate.json'
    if not gate_path.exists():return
    gate=json.loads(gate_path.read_text())
    if gate.get('status') not in ['author_revision_required','author_decision_pending']:return
    watch=json.loads((REVIEWS/'watcher_state.json').read_text())
    if gate['input_digest'] not in watch.get('handled',[]):return
    task=json.loads((PACKAGE/'task_state.json').read_text()).get('task_id','default')
    round_count=watch.get('rounds',{}).get(task,0)
    directory=safe(ROOT,gate['review_directory'])
    job=directory/'writer_revision';job.mkdir(exist_ok=True);state_file=job/'job_state.json'
    previous=json.loads(state_file.read_text()) if state_file.exists() else {}
    if previous:return
    directory,checks,editorial=context(gate['review_id'])
    delivery=publisher.publish_latest(str(PACKAGE.relative_to(ROOT)))
    if delivery.get('status')!='sent':return
    required=[f for f in editorial['findings'] if f['priority']=='required']
    if round_count>=3 or not editorial['findings']:
        result={'status':'bounded_cycle_complete','review_id':gate['review_id'],'at':stamp(),'review_rounds':round_count,'remaining_required':[f['id'] for f in required],
                'reason':'Initial review plus at most two revision checks; unresolved author decisions remain.'}
        notice(job,'cycle-complete','**[Paper Reviewer] 자동 개선 루프 종료**\n'+f'검토 {round_count}회 완료 · 남은 필수 사항 {len(required)}건.\n'+
               ('남은 항목: '+', '.join(f['id'] for f in required)+'\n' if required else '')+'최신 피드백을 바탕으로 저자 확인이 필요한 사항을 결정해 주세요. 자동으로 저자 정보·실험 결과를 만들거나 arXiv에 제출하지 않습니다.')
        write(state_file,result);write(REVIEWS/'feedback_loop_state.json',result);return
    state={'status':'writer_running','review_id':gate['review_id'],'at':stamp(),'review_rounds':round_count}
    write(state_file,state);write(REVIEWS/'feedback_loop_state.json',state)
    prompt=(f'You are the existing paper writer, responding to independent review {gate["review_id"]}. '
            'Use only the provided paper_revision_local MCP tools. FIRST read_revision_context. Read relevant original source/evidence text as needed. '
            'Improve academic prose, attribution, table/caption wording and exposition using existing evidence only. '
            'Do NOT run experiments, recalculate or invent results, add unverified claims, invent author/affiliation/checkpoint/version metadata, submit papers, or message other bots. '
            'All scientific numbers and their order within each exact patch must remain unchanged; table numeric cells and figure paths remain unchanged. '
            'The trusted build uses the original Markdown structure (including Abstract, numbered first Introduction heading, two tables, and Figure 1). Keep those markers. '
            'Keep the original four body pages plus separate references. Replace repetitive prose concisely, avoid net expansion, and retain limitations without repeating them across sections. A build that adds body pages will be rejected. '
            'Author choices and missing evidence must stay unresolved and be explained in your Korean response. Do not claim these are fixed. '
            'Prepare at most 30 exact, uniquely matching prose patches, then call submit_prose_revision once with a Korean summary and one response for EVERY finding. '
            'Patch fields are finding_id,old,new,reason. Response fields are finding_id,status,explanation,evidence; status is addressed,partial,unresolved,not_applicable. '
            'You cannot edit originals directly. Deterministic CPU build, evidence checks, required ChkTeX and later independent PDF re-review happen after submission. '
            'If no safe changes exist, submit an empty patches list with honest responses. Do not merely explain changes without submitting the tool.')
    request=job/'writer_request.txt';request.write_text(prompt)
    try:
        notice(job,'writer-start','**[논문 집필 에이전트] 검토 피드백 반영 시작**\n'+f'원고 검토 {round_count}회차의 {len(editorial["findings"])}개 항목에 응답하고, 기존 근거로 수정 가능한 내용만 반영합니다.',writer=True)
        command=[PYTHON,str(Path(__file__).with_name('revision_invoke.py')),'--query-file',str(request)]
        with (job/'writer_session.log').open('w') as output:
            process=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT,timeout=360)
        if process.returncode or not (job/'revision_proposal.json').exists():raise RuntimeError('Writer did not submit a completed revision proposal')
        finish_staged(job,checks,state)
    except Exception as error:
        state.update(status='writer_revision_blocked',at=stamp(),error_type=type(error).__name__,reason=str(error))
        try:notice(job,'writer-blocked','**[논문 집필 에이전트] 자동 수정 보류**\n'+str(error)+'\n저자 확인 또는 파이프라인 점검이 필요합니다. 새 실험이나 무제한 재시도는 실행하지 않습니다.',writer=True)
        except Exception:pass
    write(state_file,state);write(REVIEWS/'feedback_loop_state.json',state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');parser.add_argument('--resume-staged');args=parser.parse_args()
    REVIEWS.mkdir(exist_ok=True)
    with (REVIEWS/'feedback_loop.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume_staged:
            directory,checks,_=context(args.resume_staged)
            job=directory/'writer_revision'
            state=json.loads((job/'job_state.json').read_text())
            if state.get('status')!='writer_revision_blocked':raise RuntimeError('Only a blocked staged revision can be resumed')
            if (job/'apply_state.json').exists():raise RuntimeError('Do not replay a partially committed revision')
            write(job/('recovery-before-'+uuid.uuid4().hex[:12]+'.json'),state)
            finish_staged(job,checks,state)
            return
        while True:
            try:tick()
            except Exception as error:write(REVIEWS/'feedback_loop_error.json',{'error_type':type(error).__name__,'reason':str(error),'at':stamp()})
            if args.once:return
            time.sleep(20)


if __name__=='__main__':main()
