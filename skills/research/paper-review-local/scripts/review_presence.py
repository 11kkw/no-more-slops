"""Maintain the review bot's Gateway presence without ingesting chat content."""
from datetime import datetime, timezone
import json
from pathlib import Path
import discord
from dotenv import dotenv_values

PROFILE=Path('/home/happy5700x/.hermes/profiles/paper-reviewer')
STATE=PROFILE/'review_gateway_state.json'
BOT='1558443383176044556'


def record(status, **details):
    data={'status':status,'at':datetime.now(timezone.utc).isoformat(),**details}
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(STATE)


client=discord.Client(intents=discord.Intents.none(),status=discord.Status.online,
                      activity=discord.Activity(type=discord.ActivityType.watching,name='논문 작성 · 피드백 검토'))


@client.event
async def on_ready():
    if str(client.user.id)!=BOT:
        record('wrong_bot_identity');await client.close();return
    record('connected',bot_id=str(client.user.id),bot_name=client.user.name,gateway_intents=0)


@client.event
async def on_resumed():
    record('connected',bot_id=str(client.user.id),resumed=True,gateway_intents=0)


@client.event
async def on_disconnect():
    record('reconnecting',bot_id=BOT)


if __name__=='__main__':
    token=dotenv_values(PROFILE/'.env').get('DISCORD_BOT_TOKEN')
    if not token:raise RuntimeError('Review bot credential missing')
    record('connecting',bot_id=BOT)
    client.run(token,log_handler=None,reconnect=True)
