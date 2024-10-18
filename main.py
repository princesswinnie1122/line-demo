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
from dotenv import load_dotenv
from firebase import firebase
import uvicorn

# Load environment variables from .env file
if os.getenv("API_ENV") != "production":
    load_dotenv()

# Logging setup
logging.basicConfig(level=os.getenv("LOG", "INFO"))
logger = logging.getLogger(__file__)

# FastAPI app initialization
app = FastAPI()

# LINE Bot configuration
channel_secret = os.getenv("LINE_CHANNEL_SECRET")
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

if not channel_secret or not channel_access_token:
    print("LINE credentials not set.")
    sys.exit(1)

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("OPENAI_ASSISTANT_ID")

# Firebase setup
firebase_url = os.getenv("FIREBASE_URL")
fdb = firebase.FirebaseApplication(firebase_url, None)

# Health check endpoint
@app.get("/health")
async def health():
    return "ok"

# LINE webhook endpoint
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

    # Fetch or create a thread ID from Firebase
    user_chat_path = f"chat/{user_id}"
    thread_id = fdb.get(user_chat_path, "thread_id")

    if not thread_id:
        logger.info(f"No thread_id found for user {user_id}. Creating a new thread.")
        thread = client.beta.threads.create()
        thread_id = thread.id
        fdb.put(user_chat_path, "thread_id", thread_id)

    # Add the user's message to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=text,
    )

    # Run the thread with the assistant to generate a response
    try:
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        assistant_reply = run.messages[-1].content

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        assistant_reply = "Sorry, I couldn't process your request."

    # Store the updated conversation in Firebase
    fdb.put_async(user_chat_path, None, {"assistant_reply": assistant_reply})

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
    

# Entry point to run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    debug = os.getenv("API_ENV") == "develop"
    logging.info("Application will start...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
