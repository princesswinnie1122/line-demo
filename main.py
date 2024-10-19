@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    """Handle incoming messages."""
    text = event.message.text.strip()
    user_id = event.source.user_id

    # Paths for user data in Firebase
    user_chat_path = f"chat/{user_id}"
    user_data_path = f"users/{user_id}"

    # Retrieve or create thread ID
    thread_id = fdb.get(user_chat_path, "thread_id")
    if not thread_id:
        logger.info(f"Creating a new thread for user {user_id}.")
        thread = client.beta.threads.create()
        thread_id = thread.id
        fdb.put(user_chat_path, 'thread_id', thread_id)

    # Retrieve user data
    user_data = fdb.get(user_data_path, None) or {}
    user_state = user_data.get('state', 'new')

    # Initialize the response message
    reply_message = ""

    # State machine logic
    if user_state == 'new':
        # User is new, fetch nickname and send greeting message
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            try:
                profile = line_bot_api.get_profile(user_id)
                nickname = profile.display_name
            except Exception as e:
                logger.error(f"Failed to get user profile: {e}")
                nickname = 'there'  # Default to 'there' if nickname is unavailable

        account_name = os.getenv('ACCOUNT_NAME', 'Our Service')

        reply_message = f"""Hello {nickname}!
Welcome to {account_name}😊

To get started, please set up your identity by answering these questions below so we can assist you better! ✨

Let us know if you need any help along the way! We're here for you. 💬🫶

Please enter your Country/Language (e.g., Japan/Japanese)."""
        user_data['state'] = 'waiting_for_country_language'
        fdb.put('users', user_id, user_data)

    elif user_state == 'waiting_for_country_language':
        # Expecting Country/Language input
        if re.match(r"^\w+/\w+$", text):
            country, language = text.split('/')
            user_data['country'] = country
            user_data['language'] = language
            user_data['state'] = 'waiting_for_major_grade'
            fdb.put('users', user_id, user_data)
            reply_message = "Thank you! What's your major/grade? Please enter in the format Major/Grade (e.g., Computer Science/26)."
        else:
            reply_message = "Please enter your Country/Language in the correct format (e.g., Japan/Japanese)."

    elif user_state == 'waiting_for_major_grade':
        # Expecting Major/Grade input
        if re.match(r"^\w+/\d+$", text):
            major, grade = text.split('/')
            user_data['major'] = major
            user_data['grade'] = grade
            user_data['state'] = 'complete'
            fdb.put('users', user_id, user_data)
            reply_message = "Thank you! You can now start asking questions."
        else:
            reply_message = "Please enter your Major/Grade in the correct format (e.g., Computer Science/26)."

    else:
        # User has completed onboarding, proceed with assistant interaction
        # Prepare assistant prompt with user info
        country = user_data.get('country', '')
        language = user_data.get('language', '')
        major = user_data.get('major', '')
        grade = user_data.get('grade', '')

        # Add the user's message to the thread with additional context
        assistant_prompt = f"Answer the following question in {language}, based on a {grade}-year-old {major} student from {country}."
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=f"{assistant_prompt}\n\nUser: {text}",
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
        fdb.put_async(user_chat_path, 'assistant_reply', assistant_reply_cleaned)

        reply_message = assistant_reply_cleaned.strip()

    # Send the reply to the user via LINE
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_message)],
            )
        )

    return "OK"
