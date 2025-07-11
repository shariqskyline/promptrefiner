from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Initialize FastAPI
app = FastAPI()

# Model name (light version)
model_name = "alibaba-pai/Qwen2-1.5B-Instruct-Refine"

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Request model
class PromptRequest(BaseModel):
    prompt: str

# Endpoint to refine prompt
@app.post("/refine-prompt/")
async def refine_prompt(request: PromptRequest):
    user_prompt = request.prompt

    # Prompt formatting for the model
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional prompt refiner. Your task is to take a user's prompt and improve it by correcting "
                "grammar, spelling, and sentence structure. Enhance fluency, clarity, and natural tone without changing "
                "the original intent. Add slight descriptive detail only if it improves understanding. Do not over-extend, "
                "repeat, or remove any important information. Return the refined prompt as a single, clean sentence or paragraph."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    # Convert messages to a plain prompt string
    combined_prompt = ""
    for msg in messages:
        role = msg["role"].capitalize()
        combined_prompt += f"{role}: {msg['content']}\n"

    # Tokenize and run inference
    try:
        inputs = tokenizer(combined_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=128, num_return_sequences=1)
        refined = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        refined = refined.split("Refined Prompt:")[-1].strip()  # Adjust if needed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")

    return {"refined_prompt": refined}

# Uvicorn run block
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))