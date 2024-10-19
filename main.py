import json
import logging
import os
import sys
import openai
import re

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
from linebot.v3.webhooks import MessageEvent, TextMessageContent, FollowEvent
from dotenv import load_dotenv
from firebase import firebase
import uvicorn
from openai import AssistantEventHandler
from typing_extensions import override

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

# Define an EventHandler for streaming responses
class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.final_response = ""

    @override
    def on_text_created(self, text: str) -> None:
        """Handle the initial creation of text."""
        print(f"\nassistant > {text}", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        """Accumulate text deltas as they stream in."""
        self.final_response += delta.value
        print(delta.value, end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        """Log when a tool call is made."""
        print(f"\nassistant > Tool call: {tool_call.type}\n", flush=True)

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


@handler.add(FollowEvent)
def handle_follow_event(event):
    user_id = event.source.user_id

    # Get user's display name (optional)
    profile = None
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        try:
            profile = line_bot_api.get_profile(user_id)
            display_name = profile.display_name
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            display_name = "there"

    # Prepare the greeting message
    greeting_message = f"""Hello {display_name}!
Welcome to {os.getenv('LINE_BOT_NAME', 'Our Service')} 😊

To get started, please set up your identity by answering these questions below so we can assist you better! ✨

Let us know if you need any help along the way! We're here for you. 💬🫶

Please enter your Country/Language (e.g., Japan/Japanese)."""

    # Send the greeting message
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=greeting_message)],
            )
        )

    # Initialize the user's state in Firebase
    user_data_path = f"users/{user_id}"
    fdb.put(user_data_path, "state", "awaiting_country_language")


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    """Handle incoming messages."""
    text = event.message.text.strip()
    user_id = event.source.user_id

    # Paths for Firebase
    user_data_path = f"users/{user_id}"
    user_chat_path = f"chat/{user_id}"

    # Retrieve user state from Firebase
    user_state = fdb.get(user_data_path, "state")

    # If user is in the setup process
    if user_state == "awaiting_country_language":
        # Validate the format (Country/Language)
        if "/" in text:
            country_language = text.split("/", 1)
            country = country_language[0].strip()
            language = country_language[1].strip()

            # Save to Firebase
            fdb.put(user_data_path, "country", country)
            fdb.put(user_data_path, "language", language)
            fdb.put(user_data_path, "state", "awaiting_major_grade")

            # Ask for Major/Grade
            prompt_major_grade = "What's your major/grade? (e.g., Computer Science/26)"
            reply_messages = [TextMessage(text=prompt_major_grade)]

        else:
            # Invalid format, prompt again
            prompt_retry = "Please enter your Country/Language in the format 'Country/Language' (e.g., Japan/Japanese)."
            reply_messages = [TextMessage(text=prompt_retry)]

        # Send the reply
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=reply_messages,
                )
            )
        return "OK"

    elif user_state == "awaiting_major_grade":
        # Validate the format (Major/Grade)
        if "/" in text:
            major_grade = text.split("/", 1)
            major = major_grade[0].strip()
            grade = major_grade[1].strip()

            # Save to Firebase
            fdb.put(user_data_path, "major", major)
            fdb.put(user_data_path, "grade", grade)
            fdb.put(user_data_path, "state", "setup_complete")

            # Acknowledge completion
            completion_message = "Thank you! Your information has been saved. How can I assist you today?"
            reply_messages = [TextMessage(text=completion_message)]

        else:
            # Invalid format, prompt again
            prompt_retry = "Please enter your Major/Grade in the format 'Major/Grade' (e.g., Computer Science/26)."
            reply_messages = [TextMessage(text=prompt_retry)]

        # Send the reply
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=reply_messages,
                )
            )
        return "OK"

    else:
        # Regular message handling after setup
        # Retrieve or create thread ID
        thread_id = fdb.get(user_chat_path, "thread_id")

        if not thread_id:
            logger.info(f"Creating a new thread for user {user_id}.")
            thread = client.beta.threads.create()
            thread_id = thread.id
            fdb.put(user_chat_path, "thread_id", thread_id)

        # Retrieve user information for prompt customization
        country = fdb.get(user_data_path, "country") or "unknown country"
        language = fdb.get(user_data_path, "language") or "English"
        major = fdb.get(user_data_path, "major") or "your major"
        grade = fdb.get(user_data_path, "grade") or "your grade"

        # Prepare custom prompt or system message
        custom_system_message = f"Answer in {language}, and based on the student's major {major} and grade {grade}."

        # Add the user's message and system message to the thread
        # client.beta.threads.messages.create(
        #     thread_id=thread_id,
        #     role="system",
        #     content=custom_system_message,
        # )

        # Combine custom system message with user's message
        combined_message = f"{custom_system_message}\n\n{text}"

        # Add the combined message to the thread as a 'user' message
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=combined_message,
        )

        # Stream the assistant's response
        event_handler = EventHandler()

        try:
            with client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant_id,
                event_handler=event_handler,
            ) as stream:
                stream.until_done()

            assistant_reply = event_handler.final_response

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            assistant_reply = "Sorry, I couldn't process your request."

        # Remove content within 【】 from the assistant's reply
        assistant_reply_cleaned = re.sub(r'【.*?】', '', assistant_reply)

        # Store the assistant's reply in Firebase (optional)
        fdb.put_async(user_chat_path, None, {"assistant_reply": assistant_reply_cleaned})

        # Send the cleaned reply to the user via LINE
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=assistant_reply_cleaned.strip())],
                )
            )

        return "OK"



# Entry point to run the application
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        local_test()  # Run local test if 'test' argument is provided
    else:
        port = int(os.getenv("PORT", 8080))
        debug = os.getenv("API_ENV") == "develop"
        logging.info("Starting the application...")
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
