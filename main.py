import json
import logging
import os
import sys
import openai  

from fastapi import FastAPI, HTTPException, Request
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ReplyMessageRequest,
    TextMessage,
    ApiClient,
    MessagingApi,
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from firebase import firebase
from utils import is_url_valid, shorten_url_by_reurl_api, create_gcal_url
import uvicorn

# Load environment variables from .env file
if os.getenv("API_ENV") != "production":
    load_dotenv()

# Logging
logging.basicConfig(level=os.getenv("LOG", "INFO"))
logger = logging.getLogger(__file__)

app = FastAPI()

# LINE configuration
channel_secret = os.getenv("LINE_CHANNEL_SECRET")
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
if not channel_secret or not channel_access_token:
    print("LINE credentials not set.")
    sys.exit(1)

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

# OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("OPENAI_ASSISTANT_ID")

# Firebase
firebase_url = os.getenv("FIREBASE_URL")
fdb = firebase.FirebaseApplication(firebase_url, None)

# 
@app.get("/health")
async def health():
    return "ok"


@app.post("/webhooks/line")
async def handle_callback(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    body = body.decode()

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    text = event.message.text  
    user_id = event.source.user_id

    # Fetch conversation history from Firebase
    user_chat_path = f"chat/{user_id}"
    conversation_data = fdb.get(user_chat_path, None) or []

    # Add user's message to conversation
    conversation_data.append({"role": "user", "content": text})

    # Call OpenAI Assistant API
    response = openai.ChatCompletion.create(
        assistant=assistant_id,
        messages=conversation_data
    )
    assistant_reply = response.choices[0].message["content"]

    # Store updated conversation in Firebase
    conversation_data.append({"role": "assistant", "content": assistant_reply})
    fdb.put_async(user_chat_path, None, conversation_data)

    # Send the assistant's reply to the user via LINE
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=assistant_reply)],
            )
        )

    return "OK"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    debug = os.getenv("API_ENV") == "develop"
    logging.info("Application will start...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
