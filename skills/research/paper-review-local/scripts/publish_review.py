"""Publish completed, current manuscript reviews to the authorized Discord channel."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
import fcntl
from dotenv import dotenv_values
from review_mcp import ROOT, safe, sha

CHANNEL = os.environ.get('PAPER_REVIEW_DISCORD_CHANNEL', '1558119941939535973')
GUILD = os.environ.get('PAPER_REVIEW_DISCORD_GUILD', '1558103834788765766')
API = 'https://discord.com/api/v10'


class DiscordError(RuntimeError):
    def __init__(self, status, retry_after=0):
        self.status=status;self.retry_after=retry_after
        super().__init__('Discord request failed: HTTP '+str(status))


def request(method, path, token, payload=None, content_type='application/json'):
    data=payload if isinstance(payload, bytes) else json.dumps(payload,ensure_ascii=False).encode('utf-8') if payload is not None else None
    headers={'Authorization':'Bot '+token,'User-Agent':'PaperReviewPipeline/0.3.1'}
    if data is not None:headers['Content-Type']=content_type
    req=urllib.request.Request(API+path,data=data,headers=headers,method=method)
    try:
        with urllib.request.urlopen(req,timeout=25) as result:return json.load(result) if result.status!=204 else {}
    except urllib.error.HTTPError as error:
        retry_after=0
        if error.code==429:
            try:retry_after=float(json.loads(error.read()).get('retry_after',60))
            except Exception:retry_after=60
        raise DiscordError(error.code,retry_after) from None


def credentials():
    for label,file in [('paper-reviewer','/home/happy5700x/.hermes/profiles/paper-reviewer/.env'),('Hermes-Agent','/home/happy5700x/.hermes/.env')]:
        token=dotenv_values(file).get('DISCORD_BOT_TOKEN')
        if token:return label,token
    raise RuntimeError('No Discord bot credential available')


def message_payload(editorial, checks):
    findings=editorial['findings'];required=[f for f in findings if f['priority']=='required']
    required_text='\n'.join(f"{i}. {f['suggested_correction']}" for i,f in enumerate(required,1)) or '필수 수정 지적 없음. 최종 제출 판단은 저자가 합니다.'
    exact=checks['local_similarity'];near=checks.get('near_similarity',{})
    no_candidates=exact['status']=='no_exact_candidates_in_supplied_sources' and near.get('status')=='no_near_candidates_in_supplied_sources'
    return {'content':'**[논문 검토 에이전트 · Paper Reviewer]**\n초안 검토가 완료되었습니다. 상세 피드백을 첨부합니다.',
            'nonce':hashlib.sha256((CHANNEL+editorial['review_id']).encode()).hexdigest()[:24], 'enforce_nonce':True,
            'allowed_mentions':{'parse':[], 'replied_user':False},
            'embeds':[{'title':'논문 초안 검토 피드백','description':editorial['summary'].split(';')[0][:500], 'color':15105570 if required else 3066993,
                       'fields':[{'name':'검토 결과','value':f"총 {len(findings)}건 · 필수 수정 {len(required)}건 · 권고 {sum(f['priority']=='recommended' for f in findings)}건 · 경미 {sum(f['priority']=='minor' for f in findings)}건"},
                                 {'name':'필수 수정','value':required_text[:1000]},
                                 {'name':'필수 린터','value':f"ChkTeX 실제 실행 · 경고 {checks.get('latex_lint',{}).get('diagnostic_count','미확인')}건. 경고는 문맥별 수정/오탐 판단이 필요합니다."},
                                 {'name':'무료 로컬 유사도 검사','value':f"확보한 원문 {exact['source_count']}개와 비교 · "+('설정한 동일·유사 문장 후보 없음.' if no_candidates else '상세 보고서의 후보/미실행 항목 확인 필요.')+' 전체 문헌의 표절률 또는 독창성 인증이 아닙니다.'}],
                       'footer':{'text':'paper-review:'+editorial['review_id']},'timestamp':datetime.now(timezone.utc).isoformat()}]}


def multipart(payload, feedback, review_id):
    boundary='paper-review-'+uuid.uuid4().hex
    payload=dict(payload,attachments=[{'id':0,'filename':'paper-review-'+review_id+'.md','description':'전체 논문 검토 피드백'}])
    lines=[('--'+boundary+'\r\nContent-Disposition: form-data; name="payload_json"\r\nContent-Type: application/json\r\n\r\n').encode(),
           json.dumps(payload,ensure_ascii=False).encode('utf-8'),b'\r\n',
           ('--'+boundary+'\r\nContent-Disposition: form-data; name="files[0]"; filename="paper-review-'+review_id+'.md"\r\nContent-Type: text/markdown; charset=utf-8\r\n\r\n').encode(),feedback,b'\r\n',('--'+boundary+'--\r\n').encode()]
    return b''.join(lines),'multipart/form-data; boundary='+boundary


def write(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def has_marker(message, marker, bot_id):
    return message.get('author',{}).get('id')==bot_id and any(e.get('footer',{}).get('text')==marker for e in message.get('embeds',[]))


def _publish_latest(package_relative):
    if os.environ.get('PAPER_REVIEW_DISCORD_ENABLED')!='1':return {'status':'disabled'}
    package=safe(ROOT,package_relative);gate_path=package/'reviews/gate.json'
    if not gate_path.exists():return {'status':'no_review'}
    gate=json.loads(gate_path.read_text())
    if gate.get('status') not in ['author_revision_required','author_decision_pending']:return {'status':'review_not_completed'}
    directory=safe(ROOT,gate['review_directory']);receipt=directory/'discord_delivery.json'
    previous=json.loads(receipt.read_text()) if receipt.exists() else {}
    if previous.get('status')=='sent':return previous
    if previous.get('retry_at',0)>time.time():return previous
    checks=json.loads((directory/'local_checks.json').read_text());editorial=json.loads((directory/'editorial_review.json').read_text())
    if editorial['input_digest']!=checks['input_digest'] or editorial['input_digest']!=gate['input_digest']:raise RuntimeError('Mismatched review input digest')
    if checks['latex_lint']['status'] not in ['passed','warnings_found']:raise RuntimeError('Required linter has not run')
    for entry in checks['input_files']:
        if sha(safe(ROOT,entry['path']))!=entry['sha256']:return {'status':'stale_review_not_published'}
    feedback=(directory/'feedback.ko.md').read_bytes()
    if len(feedback)>1_000_000:raise RuntimeError('Feedback attachment exceeds limit')
    label,token=credentials();bot=request('GET','/users/@me',token);channel=request('GET','/channels/'+CHANNEL,token)
    if channel.get('guild_id')!=GUILD or channel.get('id')!=CHANNEL:raise RuntimeError('Discord destination mismatch')
    marker='paper-review:'+editorial['review_id']
    # Recover a confirmed message after an interrupted process before posting again.
    history=request('GET','/channels/'+CHANNEL+'/messages?limit=100',token)
    found=next((message for message in history if has_marker(message,marker,bot['id'])),None)
    metadata={'review_id':editorial['review_id'],'input_digest':editorial['input_digest'],'channel_id':CHANNEL,'guild_id':GUILD,'sender_id':bot['id'],'sender_name':bot['username'],'credential_profile':label}
    if found:
        result=dict(metadata,status='sent',message_id=found['id'],message_url=f"https://discord.com/channels/{GUILD}/{CHANNEL}/{found['id']}",recovered=True)
        write(receipt,result);return result
    if previous.get('status') in ['sending','uncertain']:
        result=dict(metadata,status='uncertain',reason='No confirmed delivery found; inspect channel before retrying')
        write(receipt,result);return result
    body,content_type=multipart(message_payload(editorial,checks),feedback,editorial['review_id'])
    write(receipt,dict(metadata,status='sending',attempted_at=datetime.now(timezone.utc).isoformat()))
    try:message=request('POST','/channels/'+CHANNEL+'/messages',token,body,content_type)
    except DiscordError as error:
        result=dict(metadata,status='retry_later' if error.status==429 else 'failed',http_status=error.status,retry_at=time.time()+max(error.retry_after,60))
        write(receipt,result);return result
    except (TimeoutError,urllib.error.URLError):
        result=dict(metadata,status='uncertain',reason='Delivery outcome unknown; inspect channel before retrying')
        write(receipt,result);return result
    verified=request('GET','/channels/'+CHANNEL+'/messages/'+message['id'],token)
    if not has_marker(verified,marker,bot['id']) or not verified.get('attachments'):raise RuntimeError('Discord feedback confirmation failed')
    result=dict(metadata,status='sent',message_id=message['id'],message_url=f"https://discord.com/channels/{GUILD}/{CHANNEL}/{message['id']}",sent_at=datetime.now(timezone.utc).isoformat(),attachment_verified=True)
    write(receipt,result);return result


def publish_latest(package_relative):
    if os.environ.get('PAPER_REVIEW_DISCORD_ENABLED')!='1':return {'status':'disabled'}
    directory=safe(ROOT,package_relative)/'reviews';directory.mkdir(exist_ok=True)
    with (directory/'discord_delivery.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'publisher_busy'}
        return _publish_latest(package_relative)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True);args=parser.parse_args()
    try:print(json.dumps(publish_latest(args.package),ensure_ascii=False))
    except Exception as error:print(json.dumps({'status':'failed','error_type':type(error).__name__,'message':str(error)},ensure_ascii=False));raise SystemExit(1)
