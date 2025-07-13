from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import contextlib
from fastapi.middleware.cors import CORSMiddleware

HF_MODEL_NAME = "ogaa12/skripsi-llama2-vocaboost-lora-model"
tokenizer = None
model = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        raise RuntimeError(f"âŒ Gagal memuat model: {e}")
    yield
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.7

@app.post("/generate")
async def generate_text(request: PromptRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat.")

    try:
        prompt = request.prompt.strip()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Hapus prompt dari hasil (biar ga ngulang prompt awal)
        cleaned_output = generated_text.replace(prompt, "").strip()
        return {"generated_text": cleaned_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"âŒ Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok" if model and tokenizer else "not ready"}
