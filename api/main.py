from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.callbacks import AsyncIteratorCallbackHandler

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage
)

from dotenv import load_dotenv
from icecream import ic
from typing import AsyncIterable

import asyncio
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


app = FastAPI()

class StreamRequest(BaseModel):
    """Request body for streaming."""
    message: str


### Send messages streaming
async def send_message(message: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )

    # Begin a task that runs in the background.
    task = asyncio.create_task(
        model.agenerate(messages=[[HumanMessage(content=message)]])
        )
    
    try:
        async for token in callback.aiter():
            # Use server-sent-events to stream the response
            yield token
    except Exception as e_async:
        ic(e_async)
    finally:
        callback.done.set()

    await task



@app.get("/")
def read_root():
    return {"message": "API Test Open AI"}



@app.post("/stream")
def stream(body: StreamRequest):
    return StreamingResponse(send_message(body.message), media_type="text/event-stream")

