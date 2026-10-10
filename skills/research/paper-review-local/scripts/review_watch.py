"""Watch one paper package and dispatch bounded isolated manuscript reviews."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl
from review_mcp import ROOT, audit, run_review, safe, VERSION
from publish_review import publish_latest


def write_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def signature(package):
    paths = [p for p in sorted(package.rglob('*')) if p.is_file() and 'reviews' not in p.relative_to(package).parts and '__pycache__' not in p.relative_to(package).parts]
    return hashlib.sha256(json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()]).encode()).hexdigest()


def deliver_feedback(package_relative, directory):
    try:
        result=publish_latest(package_relative)
        if result.get('status') in ['failed','uncertain']:
            write_json(directory/'discord_publish_error.json', result)
    except Exception as error:
        write_json(directory/'discord_publish_error.json', {'error':type(error).__name__,'message':str(error),'at':datetime.now(timezone.utc).isoformat()})


def dispatch(package_relative, state):
    report = audit(package_relative)
    digest = hashlib.sha256(json.dumps({'version': VERSION, 'files': report['input_files']}, sort_keys=True).encode()).hexdigest()
    package = safe(ROOT, package_relative)
    if digest in state.get('handled', []): return state
    task = json.loads((package/'task_state.json').read_text()).get('task_id', 'default') if (package/'task_state.json').exists() else 'default'
    rounds = state.setdefault('rounds', {})
    if rounds.get(task, 0) >= 3:
        write_json(package/'reviews/gate.json', {'status': 'revision_limit_reached', 'input_digest': digest, 'reason': 'Initial review plus two revision checks used; author decision needed.'})
        return state
    review_id = 'auto-' + digest[:20]
    report = run_review(package_relative, review_id)
    directory = Path(report['review_directory'])
    gate = {'review_id': review_id, 'input_digest': digest, 'checked_at': datetime.now(timezone.utc).isoformat(), 'status': 'awaiting_editorial_review', 'review_directory': str(directory)}
    if report['latex_lint']['status'] not in ['passed', 'warnings_found']:
        gate.update(status='blocked_required_linter', reason=report['latex_lint']['status'])
        write_json(package/'reviews/gate.json', gate)
        raise RuntimeError('Required ChkTeX did not run successfully')
    write_json(package/'reviews/gate.json', gate)
    prompt = (f'Review the frozen manuscript package {package_relative} using the paper-review-local skill. '
              f'FIRST call run_local_review with review_id {review_id}; ChkTeX is mandatory. '
              'Read the original manuscript, citation/source context and all rendered PDF pages attached as native images at the start. '
              'The attachment order matches page numbers. Do not treat MCP MEDIA path strings as pixels. '
              'Use the deterministic exact and near-string matches as candidates only, and state the local source coverage. '
              'Evaluate academic prose, claims, attribution, table/figure readability and arXiv preparation. '
              'Do not read writer self-review first. Do not change originals, run experiments, upload manuscripts or contact Discord. '
              'Finish by calling save_editorial_review with the returned input digest and structured findings, plus a concise Korean summary. '
              'Findings must contain id, location, evidence, priority, explanation, confidence, suggested_correction, all strings. '
              'State every unverified check. Do not certify plagiarism freedom or arXiv acceptance.')
    if (package/'reviews/latest_writer_response.json').exists():
        prompt += (' This is a revision recheck. First inspect the current original manuscript and all attached PDF pages independently. '
                   'Then read reviews/latest_writer_response.json and the previous editorial report it identifies through read_review_input. '
                   'Compare prior findings with the author response, verify actual changes, and report unresolved/new issues. '
                   'Do not treat the writer response as proof that an issue is resolved.')
    prompt_file = directory/'review_request.txt'; prompt_file.write_text(prompt, encoding='utf-8')
    env = dict(os.environ, HERMES_HOME='/home/happy5700x/.hermes/profiles/paper-reviewer', HERMES_ACTIVE_PROFILE='paper-reviewer')
    command = ['/home/happy5700x/.hermes/hermes-agent/venv/bin/python', str(Path(__file__).with_name('review_invoke.py')),
               '--query-file', str(prompt_file), '--package', package_relative]
    try:
        with (directory/'reviewer_session.log').open('w', encoding='utf-8') as output:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=540)
        editorial = directory/'editorial_review.json'
        if process.returncode or not editorial.exists():
            gate.update(status='reviewer_failed', exit_code=process.returncode, reason='No completed structured editorial report')
        else:
            saved = json.loads(editorial.read_text(encoding='utf-8'))
            priorities = Counter(f['priority'] for f in saved['findings'])
            gate.update(status='author_revision_required' if priorities['required'] else 'author_decision_pending', finding_counts=dict(priorities))
            feedback = ['# 논문 검토 피드백', '', saved['summary'], '',
                        '이 피드백은 현재 원고와 확보한 참고자료의 검토 결과입니다. 전체 문헌에 대한 표절 판정이나 arXiv 승인 판정이 아닙니다.', '']
            for f in saved['findings']:
                feedback.extend([f"## {f['id']} — {f['priority']}", '', f"위치: {f['location']}", '', f['explanation'], '', '수정 제안: '+f['suggested_correction'], ''])
            (directory/'feedback.ko.md').write_text('\n'.join(feedback), encoding='utf-8')
    except subprocess.TimeoutExpired:
        gate.update(status='reviewer_timeout', reason='Bounded review runtime exceeded; no automatic retry')
    state.setdefault('handled', []).append(digest)
    rounds[task] = rounds.get(task, 0) + 1
    write_json(package/'reviews/gate.json', gate)
    return state


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--package', required=True); parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    package = safe(ROOT, args.package); directory=package/'reviews'; directory.mkdir(exist_ok=True)
    state_file = directory/'watcher_state.json'
    with (directory/'watcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state=json.loads(state_file.read_text()) if state_file.exists() else {'handled': [], 'rounds': {}}
        if args.once:
            state=dispatch(args.package,state);write_json(state_file,state);deliver_feedback(args.package,directory);return
        previous=None; stable_since=time.monotonic()
        while True:
            current=signature(package)
            if current != previous: previous=current;stable_since=time.monotonic()
            elif time.monotonic()-stable_since>=30 and state.get('last_signature')!=current:
                try:
                    state=dispatch(args.package,state);write_json(state_file,state)
                    state['last_signature']=current;write_json(state_file,state)
                except Exception as error:
                    write_json(directory/'watcher_error.json', {'error': type(error).__name__, 'message': str(error), 'at': datetime.now(timezone.utc).isoformat()})
                stable_since=time.monotonic()
            deliver_feedback(args.package,directory)
            time.sleep(20)


if __name__=='__main__':main()
