from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Initialize FastAPI
app = FastAPI()

# Corrected model name for the lighter version
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

# Define request model
class PromptRequest(BaseModel):
    prompt: str

@app.post("/refine-prompt/")
async def refine_prompt(request: PromptRequest):
    # User input prompt
    user_prompt = request.prompt

    # Refiner instruction
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

    # Tokenize the messages
    inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate output
    try:
        output = model.generate(**inputs, max_new_tokens=128, num_return_sequences=1)
        refined = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        refined = refined.split("Refined Prompt:")[-1].strip()  # Adjust according to your output format
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")

    return {"refined_prompt": refined}

# Uvicorn run block for deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))