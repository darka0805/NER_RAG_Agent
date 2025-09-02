# streamlit_app.py

import streamlit as st
import requests
import logging
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API URL (default: local FastAPI service)
API_URL = os.getenv("AGENT_API_URL", "http://localhost:8001")

# Streamlit page config
st.set_page_config(page_title="Agent API Chatbot", page_icon="🤖")
st.header("Agent API Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the student or try NER..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # Call FastAPI backend
            payload = {"text": prompt}
            response = requests.post(f"{API_URL}/process", json=payload)
            response.raise_for_status()

            # Extract response
            response_data = response.json().get("response", "⚠️ No response received.")
            message_placeholder.markdown(response_data)

            # Save to session state
            st.session_state.messages.append({"role": "assistant", "content": response_data})

        except requests.exceptions.ConnectionError:
            st.error("Помилка: Не вдалося підключитися до API. Переконайтеся, що ваш FastAPI-сервіс запущено.")
            logger.error("Помилка підключення до API за адресою %s", API_URL)

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Помилка: {e}. Можливо, ендпоінт '/process' не знайдено.")
            logger.error("HTTP помилка: %s", e)

        except Exception as e:
            st.error("Виникла непередбачена помилка.")
            logger.error("Непередбачена помилка: %s", e)
