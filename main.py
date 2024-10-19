import json
import logging
import os
import sys
import openai
import re
import tempfile

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
from linebot.v3.webhooks import (
    MessageEvent, 
    FollowEvent, 
    TextMessageContent, 
    AudioMessageContent,
    ImageMessageContent
)

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

    # Prepare the greeting messages
    greeting_message_part1 = f"""Hello {display_name}!!
Welcome to UniHelp 😊

To get started, please set up your identity by answering these questions below so we can assist you better! ✨

Let us know if you need any help along the way! We're here for you. 💬🫶"""

    greeting_message_part2 = """【STEP 1】Please enter your country and native language (e.g., Japan, Japanese)."""

    # Send the greeting messages
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(text=greeting_message_part1),
                    TextMessage(text=greeting_message_part2),
                ],
            )
        )

    # Initialize the user's state in Firebase
    user_data_path = f"users/{user_id}"
    fdb.put(user_data_path, "state", "awaiting_country_language")


def handle_user_message(event: MessageEvent, text: str):
    """Unified message handling for text and transcribed audio."""
    user_id = event.source.user_id
    reply_token = event.reply_token

    # Paths for Firebase
    user_data_path = f"users/{user_id}"
    user_chat_path = f"chat/{user_id}"

    # Retrieve user state from Firebase
    user_state = fdb.get(user_data_path, "state")


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    """Handle incoming messages."""
    text = event.message.text.strip()
    user_id = event.source.user_id
    handle_user_message(event, text)


    # Paths for Firebase
    user_data_path = f"users/{user_id}"
    user_chat_path = f"chat/{user_id}"

    # Retrieve user state from Firebase
    user_state = fdb.get(user_data_path, "state")

    # If user is in the setup process
    if user_state == "awaiting_country_language":
        # Validate the format (Country, Language)
        if "," in text:
            country_language = text.split(",", 1)
            country = country_language[0].strip()
            language = country_language[1].strip()

            # Save to Firebase
            fdb.put(user_data_path, "country", country)
            fdb.put(user_data_path, "language", language)
            fdb.put(user_data_path, "state", "awaiting_major_grade")

            # Ask for Major/Grade
            prompt_major_grade = "【STEP 2】What's the major and grade you're in? (e.g., Computer Science, 3)"
            reply_messages = [TextMessage(text=prompt_major_grade)]

        else:
            # Invalid format, prompt again
            prompt_retry = "Please enter in the format 'Country, Language' (e.g., Japan, Japanese)."
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
        # Validate the format (Major, Grade)
        if "," in text:
            major_grade = text.split(",", 1)
            major = major_grade[0].strip()
            grade = major_grade[1].strip()

            # Save to Firebase
            fdb.put(user_data_path, "major", major)
            fdb.put(user_data_path, "grade", grade)
            fdb.put(user_data_path, "state", "awaiting_mode_selection")

            # Ask for mode preference
            completion_message = "Thank you! Your information has been saved. Would you prefer normal or bilingual mode (showing both your native language and Traditional Chinese)? Type 0 for normal and 1 for bilingual."
            reply_messages = [TextMessage(text=completion_message)]

        else:
            # Invalid format, prompt again
            prompt_retry = "Please enter in the format 'Major, Grade' (e.g., Computer Science, 3)."
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

    elif user_state == "awaiting_mode_selection":
        # Validate the user's input (0 or 1)
        if text in ["0", "1"]:
            fdb.put(user_data_path, "mode", text)
            fdb.put(user_data_path, "state", "setup_complete")

            # Acknowledge completion
            if text == "1":
                # User selected bilingual mode
                completion_message = (
                    "Thank you! Your preference has been saved. How can I assist you today?\n"
                    "謝謝！您的偏好已保存。請問今天有什麼可以幫助您的？"
                )
            else:
                # User selected normal mode
                completion_message = "Thank you! Your preference has been saved. How can I assist you today?"

            reply_messages = [TextMessage(text=completion_message)]
        else:
            # Invalid input, prompt again
            prompt_retry = "Please enter 0 for normal mode or 1 for bilingual mode."
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
        mode = fdb.get(user_data_path, "mode") or "0"  # default to normal mode

        # Prepare custom prompt or system message based on mode
        if mode == "0":
            custom_system_message = f"Answer in {language} based on the student's major {major} and grade {grade}."
        elif mode == "1":
            custom_system_message = f"Answer in both {language} based on the student's major {major} and grade {grade}. Then answer a translated version of Traditional Chinese again. (Do not mention words like 'Traditional Chinese' or 'Translate' in your answer.)"
        else:
            # Default to normal mode if mode is not recognized
            custom_system_message = f"Answer in {language}, and based on the student's major {major} and grade {grade}."

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
def reset_user_to_initial_state(user_id: str, reply_token: str):
    """Reset the user's data to the initial state and send the welcome message."""
    user_data_path = f"users/{user_id}"
    user_chat_path = f"chat/{user_id}"

    # 刪除該使用者的資料和聊天記錄
    fdb.delete(user_data_path, None)
    fdb.delete(user_chat_path, None)

    # 初始化狀態，回到「等待國家和語言」的階段
    fdb.put(user_data_path, "state", "awaiting_country_language")

    # 傳送歡迎訊息和初始化問題
    greeting_message_part1 = f"""Hello! 👋  
Welcome back to UniHelp 😊  

We’ve reset your information to start fresh. Let's set up your identity again to assist you better! ✨"""

    greeting_message_part2 = """【STEP 1】Please enter your country and native language (e.g., Japan, Japanese)."""

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[
                    TextMessage(text="Your data has been reset successfully."),
                    TextMessage(text=greeting_message_part1),
                    TextMessage(text=greeting_message_part2),
                ],
            )
        )

# 處理 TextMessage 事件，偵測 reset 指令
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    text = event.message.text.strip().lower()
    user_id = event.source.user_id

    if text == "reset" or text == "user setup":
        # 執行使用者重設
        reset_user_to_initial_state(user_id, event.reply_token)
    else:
        # 處理其他訊息
        handle_user_message(event, text)


@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio_message(event: MessageEvent):
    user_id = event.source.user_id
    message_id = event.message.id

    # Get the audio content from LINE's server using MessageContentApi
    with ApiClient(configuration) as api_client:
        message_content_api = MessageContentApi(api_client)
        message_content_response = message_content_api.get_message_content(message_id)
        audio_content = message_content_response.read()

    # Save the audio content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_audio_file:
        temp_audio_file.write(audio_content)
        temp_audio_file_path = temp_audio_file.name

    try:
        # Transcribe the audio using OpenAI's Whisper API
        audio_file = open(temp_audio_file_path, 'rb')
        transcript = openai.Audio.transcribe('whisper-1', audio_file)
        transcribed_text = transcript['text']
        audio_file.close()

        # Use the transcribed text as input to your assistant
        if transcribed_text:
            handle_user_message(event, transcribed_text)
        else:
            # Send an error message to the user
            error_message = "Sorry, I couldn't understand your audio message. Please try again."
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=error_message)],
                    )
                )
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        error_message = "An error occurred while processing your audio message. Please try again later."
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=error_message)],
                )
            )
    finally:
        # Delete the temporary file
        if os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path)

            
# Entry point to run the application
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        local_test()  # Run local test if 'test' argument is provided
    else:
        port = int(os.getenv("PORT", 8080))
        debug = os.getenv("API_ENV") == "develop"
        logging.info("Starting the application...")
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
