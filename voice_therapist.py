import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

# Load environment variables for API keys
load_dotenv()

# Define a simple user data class for therapy context
@dataclass
class UserData:
    user_name: Optional[str] = None
    session_notes: list[str] = field(default_factory=list)

    def summarize(self) -> str:
        return f"User: {self.user_name or 'unknown'}, Notes: {self.session_notes}"

RunContext_T = RunContext[UserData]

class TherapyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                """
                You are a compassionate AI therapist.
                - Listen carefully to the user's concerns and respond empathetically.
                - Ask open-ended questions when appropriate to encourage reflection.
                - Keep your responses concise: respond in 200 characters or less, but ensure your answer is a complete, helpful thought.
                - If you need to summarize, do so in a way that preserves the user's intent and emotional content.
                - Format your response as a single, clear sentence.
                - Example:
                User: I feel overwhelmed at work and don't know how to cope.
                Assistant: It sounds like work has been stressful; can you share what feels most overwhelming right now?
                """
            ),
            tts=cartesia.TTS(),
            llm=openai.LLM.with_ollama(
                model="llava:13b",
                base_url="http://localhost:11434/v1",
            ),  # Use Ollama via OpenAI plugin
        )

    async def on_enter(self) -> None:
        chat_ctx = self.chat_ctx.copy()
        userdata: UserData = self.session.userdata
        chat_ctx.add_message(
            role="system",
            content=f"Therapy session started. {userdata.summarize()}"
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    userdata = UserData()
    agent = TherapyAgent()
    session = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM.with_ollama(
            model="llava:13b",
            base_url="http://localhost:11434/v1",
        ),  # Use Ollama via OpenAI plugin
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
