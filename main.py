import json
import logging
import os
import sys
import openai
import re
import tempfile
import requests
from PIL import Image
from io import BytesIO
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ReplyMessageRequest,
    TextMessage,
    AudioMessage,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
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
import google.generativeai as genai
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

Let us know if you need any help along the way! We're here for you. 🫶"""

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
            completion_message = """Thank you! Your information has been saved. 

Would you prefer normal or bilingual mode (showing both your native language and Traditional Chinese)?

Type "0" for normal and "1" for bilingual.💬"""

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

'''
def reset_user_to_initial_state(user_id: str, reply_token: str):
    """重設使用者資料，並模擬追蹤事件（FollowEvent）。"""
    user_data_path = f"users/{user_id}"
    user_chat_path = f"chat/{user_id}"

    # 刪除該使用者的所有資料和聊天記錄
    fdb.delete(user_data_path, None)
    fdb.delete(user_chat_path, None)

    # 初始化 Firebase 狀態為等待輸入國家和語言
    fdb.put(user_data_path, "state", "awaiting_country_language")

    # 模擬一個 FollowEvent 事件並傳遞給 handle_follow_event
    mock_event = FollowEvent(
        source=event.source,  # 包含 user_id 等相關資料
        reply_token=reply_token,
        timestamp=None  # 可選，設為 None 或 datetime.now().timestamp()
    )

    # 呼叫 handle_follow_event，模擬使用者剛加入時的情況
    handle_follow_event(mock_event)


# 處理 TextMessage 事件，偵測 reset 指令
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    text = event.message.text.strip().lower()
    user_id = event.source.user_id

    if text == "User Setup":
        # 執行使用者重設
        reset_user_to_initial_state(user_id, event.reply_token)
    else:
        # 處理其他訊息
        handle_user_message(event, text)
'''



@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio_message(event):

    user_id = event.source.user_id
    reply_token = event.reply_token
    message_id = event.message.id

    user_data_path = f"users/{user_id}"
    user_chat_path = f"chat/{user_id}"
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
            completion_message = """Thank you! Your information has been saved. Would you prefer normal or bilingual mode (showing both your native language and Traditional Chinese)? Type 0 for normal and 1 for bilingual.💬"""

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
        with ApiClient(configuration) as api_client:
            line_bot_blob_api = MessagingApiBlob(api_client)
            audio_content = line_bot_blob_api.get_message_content(message_id)

        # Save the audio content to a temporary file with .m4a extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio_file:
            temp_audio_file.write(audio_content)
            temp_audio_path = temp_audio_file.name

        # Save the temporary audio file to Firebase storage or database
        with open(temp_audio_path, "rb") as audio_file:
            # Convert the file to binary data and save it in Firebase
            fdb.put(f"audio/{user_id}", message_id, audio_file.read())

        # Convert the audio to text using Whisper API
        with open(temp_audio_path, "rb") as audio_file:
            whisper_response = openai.Audio.transcribe("whisper-1", audio_file)
            transcribed_text = whisper_response["text"]

        # Delete the temporary audio file after conversion
        os.remove(temp_audio_path)
        logger.info(f"{transcribed_text}")

        thread_id = fdb.get(user_chat_path, "thread_id")
        if not thread_id:
            logger.info(f"Creating a new thread for user {user_id}.")
            thread = client.beta.threads.create()
            thread_id = thread.id
            fdb.put(user_chat_path, "thread_id", thread_id)


        custom_system_message = f"Organize content of the audio into notes. Do not use markdown bold formatting (**). Do not start with 'The image shows' or 'This image depicts'. Here's the description of the image:"
        combined_message = f"{custom_system_message}\n\n{transcribed_text}"

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=combined_message,
        )

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


        # Store the assistant's reply in Firebase (optional)
        fdb.put_async(user_chat_path, None, {"assistant_reply_to_audio": assistant_reply})

        # Send the cleaned reply to the user via LINE
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=assistant_reply.strip())],
                )
            )

        return "OK"


# Image processing
def check_image(url=None, b_image=None):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    if url is not None:
        response = requests.get(url)
        if response.status_code == 200:
            image_data = response.content
    elif b_image is not None:
        image_data = b_image
    else:
        return "None"
    logger.info(f"URL: {url} \n Image: {b_image}")
    image = Image.open(BytesIO(image_data))

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [
            "Extracted the words and describe the whole image.",
            image,
        ]
    )
    return response.text

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):

    user_id = event.source.user_id
    user_data_path = f"users/{user_id}"
    user_chat_path = f"chat/{user_id}"
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
            completion_message = """Thank you! Your information has been saved. Would you prefer normal or bilingual mode (showing both your native language and Traditional Chinese)? Type 0 for normal and 1 for bilingual.💬"""

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
        image_content = b""
        with ApiClient(configuration) as api_client:
            line_bot_blob_api = MessagingApiBlob(api_client)
            image_content = line_bot_blob_api.get_message_content(event.message.id)
        image_data = check_image(b_image=image_content)
        logger.info(f"{image_data}")

        thread_id = fdb.get(user_chat_path, "thread_id")
        if not thread_id:
            logger.info(f"Creating a new thread for user {user_id}.")
            thread = client.beta.threads.create()
            thread_id = thread.id
            fdb.put(user_chat_path, "thread_id", thread_id)


        custom_system_message = f"Organize content of the image into notes. Do not use markdown bold formatting (**). Do not start with 'The image shows' or 'This image depicts'. Here's the description of the image:"
        combined_message = f"{custom_system_message}\n\n{image_data}"

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=combined_message,
        )

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


        # Store the assistant's reply in Firebase (optional)
        fdb.put_async(user_chat_path, None, {"assistant_reply_to_image": assistant_reply})

        # Send the cleaned reply to the user via LINE
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=assistant_reply.strip())],
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
