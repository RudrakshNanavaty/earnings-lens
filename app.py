import os
import sys
import asyncio
import json
import gradio as gr
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from earnings_analyst.server.earnings_analyst_environment import EarningsAnalystEnvironment
    from earnings_analyst.models import EarningsAnalystAction, EarningsAnalystObservation
    from earnings_analyst.tasks.registry import TASK_IDS, TASKS
except (ImportError, ModuleNotFoundError):
    from server.earnings_analyst_environment import EarningsAnalystEnvironment
    from models import EarningsAnalystAction, EarningsAnalystObservation
    from tasks.registry import TASK_IDS, TASKS

load_dotenv()

class State:
    def __init__(self):
        self.env: Optional[EarningsAnalystEnvironment] = None
        self.obs: Optional[EarningsAnalystObservation] = None
        self.task_id: str = "sentiment_label"

state = State()

async def reset_env(task_id: str):
    state.task_id = task_id
    state.env = EarningsAnalystEnvironment(task_id=task_id)
    state.obs = state.env.reset()
    
    # Format observation for display
    text_context = ""
    if state.obs.text_context:
        for name, text in sorted(state.obs.text_context.items()):
            text_context += f"### {name}\n{text}\n\n"
            
    numerical_context = json.dumps(state.obs.numerical_context, indent=2) if state.obs.numerical_context else "No numerical data."
    
    return [
        state.obs.task_instruction,
        text_context,
        numerical_context,
        gr.update(visible=True), # Prediction row
        gr.update(visible=False), # Result row
        "", # Prediction input
        "", # Log/Message
    ]

async def step_env(prediction: str):
    if not state.env or not state.obs:
        return [gr.update(), "Error: Environment not initialized. Click Reset."]
    
    action = EarningsAnalystAction(prediction=prediction)
    state.obs = state.env.step(action)
    
    reward = state.obs.reward
    ground_truth = getattr(state.obs, "ground_truth", "N/A")
    
    result_text = f"**Reward:** {reward:.4f} \n\n**Ground Truth:** {ground_truth}"
    
    return [
        gr.update(visible=False), # Hide prediction row
        gr.update(visible=True, value=result_text), # Show result row
    ]

async def run_agent(task_id: str, api_key: str, model: str, base_url: str):
    if not api_key:
        return [gr.update()] * 6 + ["Please provide an API Key."]
    
    # 1. Reset
    out = await reset_env(task_id)
    
    # 2. Predict with LLM
    client_params = {"api_key": api_key}
    if base_url:
        client_params["base_url"] = base_url
    
    client = AsyncOpenAI(**client_params)
    
    user_content = f"{state.obs.task_instruction}\n\n"
    if state.obs.text_context:
        user_content += "## Text context\n"
        for name, text in sorted(state.obs.text_context.items()):
            user_content += f"### {name}\n{text}\n"
    if state.obs.numerical_context:
        user_content += f"\n## Numerical context\n{json.dumps(state.obs.numerical_context)}\n"
        
    system_prompt = (
        "You are a financial analyst assistant. "
        "Analyze the data and respond EXACTLY as instructed. "
        "Reply with a single JSON object containing 'prediction' key."
    )
    
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        response_text = completion.choices[0].message.content or "{}"
        parsed = json.loads(response_text)
        prediction = str(parsed.get("prediction", response_text))
        
        # 3. Step
        step_out = await step_env(prediction)
        
        # Update UI components
        # out: [instr, text, num, pred_row, res_row, pred_input, msg]
        # step_out: [pred_row, res_row]
        
        return [
            out[0], out[1], out[2], 
            step_out[0], step_out[1], 
            prediction, 
            f"Agent used {model}. Raw response: {response_text}"
        ]
        
    except Exception as e:
        return [out[0], out[1], out[2], out[3], out[4], "", f"Error: {str(e)}"]

# Custom CSS for a premium look
custom_css = """
footer {visibility: hidden}
.container { 
    max-width: 1100px; 
    margin: auto; 
    padding-top: 2rem; 
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.header { text-align: center; margin-bottom: 2rem; }
.header h1 { font-weight: 800; font-size: 2.5rem; background: linear-gradient(90deg, #ff4d94, #ff85b3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header p { color: #666; font-size: 1.1rem; }
.card { border-radius: 12px; border: 1px solid #eee; background: white; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
.context-box { height: 400px; overflow-y: auto; border: 1px solid #f0f0f0; border-radius: 8px; padding: 1rem; background: #fafafa; }
"""

with gr.Blocks(css=custom_css, title="Earnings Analyst - OpenEnv") as demo:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("# 🏑 Earnings Analyst")
            gr.Markdown("Interactive environment for financial analysis tasks using [OpenEnv](https://github.com/meta-pytorch/OpenEnv). Evaluate agents or your own analysis on earnings call data.")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    task_select = gr.Dropdown(choices=TASK_IDS, value="sentiment_label", label="Active Task")
                    reset_btn = gr.Button("🔄 New Episode", variant="primary")
                    
                gr.Markdown("### 🤖 Auto-Agent Settings")
                with gr.Group():
                    api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...", value=os.environ.get("OPENAI_API_KEY", ""))
                    model_name = gr.Textbox(label="Model", value="gpt-4o-mini")
                    base_url = gr.Textbox(label="Base URL (optional)", placeholder="https://api.openai.com/v1", value=os.environ.get("OPENAI_BASE_URL", ""))
                    agent_btn = gr.Button("🚀 Run LLM Agent", variant="secondary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Observation"):
                        instr_view = gr.Markdown("### Instruction\n*N/A*")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### Text Context")
                                text_view = gr.Markdown(elem_classes="context-box")
                            with gr.Column():
                                gr.Markdown("#### Numerical Context")
                                num_view = gr.Code(label="JSON", language="json", elem_classes="context-box")
                    
                    with gr.TabItem("Analysis"):
                        with gr.Column(visible=False) as prediction_row:
                            pred_input = gr.Textbox(label="Your Prediction / Analysis Output", placeholder="e.g. bullish, or 0.05")
                            submit_btn = gr.Button("Submit Analysis", variant="primary")
                        
                        result_view = gr.Markdown(visible=False, elem_classes="card")
                        message_view = gr.Textbox(label="Agent Log / Error Messages", interactive=False)

    # Event handlers
    reset_btn.click(
        fn=reset_env,
        inputs=[task_select],
        outputs=[instr_view, text_view, num_view, prediction_row, result_view, pred_input, message_view]
    )

    submit_btn.click(
        fn=step_env,
        inputs=[pred_input],
        outputs=[prediction_row, result_view]
    )

    agent_btn.click(
        fn=run_agent,
        inputs=[task_select, api_key, model_name, base_url],
        outputs=[instr_view, text_view, num_view, prediction_row, result_view, pred_input, message_view]
    )

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create the main FastAPI app
api_app = FastAPI(title="Earnings Analyst API")

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import the environment app
try:
    from earnings_analyst.server.app import app as env_app
except (ImportError, ModuleNotFoundError):
    from server.app import app as env_app

# Mount the environment API at /api
api_app.mount("/api", env_app)

# Wrap Gradio with FastAPI
app = gr.mount_gradio_app(api_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
