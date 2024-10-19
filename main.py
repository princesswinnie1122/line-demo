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
from typing_extensions import override
from openai import AssistantEventHandler

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

# OpenAI client initialization
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

# Assistant ID from environment variables
assistant_id = os.getenv("OPENAI_ASSISTANT_ID")

# Firebase setup
firebase_url = os.getenv("FIREBASE_URL")
fdb = firebase.FirebaseApplication(firebase_url, None)

# Define an EventHandler to handle streaming responses
class EventHandler(AssistantEventHandler):
    def __init__(self):
        self.final_response = ""  # Store the accumulated response

    @override
    def on_text_created(self, text) -> None:
        """Handle initial text creation."""
        print(f"\nassistant > {text}", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        """Accumulate partial responses."""
        self.final_response += delta.value
        print(delta.value, end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        """Log when a tool is called."""
        print(f"\nassistant > Tool call: {tool_call.type}\n", flush=True)

    @override
    def on_tool_call_delta(self, delta, snapshot):
        """Log intermediate tool results."""
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

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
    """Handle incoming LINE messages."""
    text = event.message.text
    user_id = event.source.user_id

    # Retrieve or create thread ID
    user_chat_path = f"chat/{user_id}"
    thread_id = fdb.get(user_chat_path, "thread_id")

    if not thread_id:
        logger.info(f"No thread_id found for user {user_id}. Creating a new thread.")
        thread = client.beta.threads.create()
        thread_id = thread.id
        fdb.put(user_chat_path, "thread_id", thread_id)

    # Add user message to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=text,
    )

    # Stream the assistant response
    stream_handler = EventHandler()

    try:
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            event_handler=stream_handler,
        ) as stream:
            stream.until_done()

        assistant_reply = stream_handler.final_response  # Accumulate response

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        assistant_reply = "Sorry, I couldn't process your request."

    # Store reply in Firebase and send to LINE
    fdb.put_async(user_chat_path, None, {"assistant_reply": assistant_reply})

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=assistant_reply)],
            )
        )

    return "OK"

'''
# Local test function to simulate interactions
def local_test():
    print("Starting local test...")
    thread = client.beta.threads.create()
    print(f"Created thread with ID: {thread.id}")

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="What courses are available in the JSON file?",
    )

    stream_handler = EventHandler()

    try:
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant_id,
            event_handler=stream_handler,
        ) as stream:
            stream.until_done()

        print("\nFinal Response:", stream_handler.final_response)

    except Exception as e:
        logger.error(f"OpenAI API error during local test: {e}")

'''


# Entry point to run the application
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        local_test()  # Run local test if 'test' argument is provided
    else:
        port = int(os.getenv("PORT", 8080))
        debug = os.getenv("API_ENV") == "develop"
        logging.info("Starting the application...")
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
