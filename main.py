from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
import google.generativeai as genai
import base64
import json
import os

# 1. Initialize FastAPI Application
app = FastAPI(title="IITM Combined API Service")

# 2. Enable CORS (Required for Cloudflare Worker grader validation)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Configure Gemini API (Safely reading from Environment Variables only)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("CRITICAL ERROR: GEMINI_API_KEY environment variable is not set!")

genai.configure(api_key=GEMINI_API_KEY)


# ==========================================
# TASK 1: FIXED SCHEMA INVOICE EXTRACTION
# ==========================================

class InvoiceRequest(BaseModel):
    invoice_text: Optional[str] = None
    text: Optional[str] = None

class InvoiceResponse(BaseModel):
    contact_email: Optional[str] = None
    currency: Optional[str] = "INR"
    due_in_days: Optional[int] = None
    invoice_date: Optional[str] = None
    is_paid: Optional[bool] = False
    item_count: Optional[int] = 0
    line_items: List[Any] = []
    priority: Optional[str] = "medium"
    total_amount: Optional[float] = None
    vendor: Optional[str] = None

@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(payload: InvoiceRequest):
    try:
        text = payload.invoice_text or payload.text
        if not text:
            raise HTTPException(status_code=422, detail="Content missing.")
            
        # Updated to explicitly use the models/ prefix to solve the 404 issue
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        
        prompt = (
            "You are a precise financial data extraction assistant.\n"
            f"Analyze this raw invoice text:\n{text}\n\n"
            "Extract and return a strict JSON object with these exact keys:\n"
            "- contact_email (string, lowercase email found in text, null if missing)\n"
            "- currency (string, standard 3-letter code like INR, USD, EUR. Default to INR)\n"
            "- due_in_days (integer, extract payment terms/due days sequence, default to 21)\n"
            "- invoice_date (string, normalize strictly to YYYY-MM-DD format, null if missing)\n"
            "- is_paid (boolean, true if clearly marked as paid, otherwise false)\n"
            "- item_count (integer, number of items listed, default to 1)\n"
            "- line_items (array, leave empty [])\n"
            "- priority (string, default to 'medium')\n"
            "- total_amount (number, raw floating-point total/payable value, null if missing)\n"
            "- vendor (string, vendor/issuer company name, null if missing)\n\n"
            "Return ONLY the valid JSON object. No conversation, no markdown ticks."
        )

        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        extracted_data = json.loads(response.text.strip())
        return InvoiceResponse(**extracted_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failure: {str(e)}")


# ==========================================
# TASK 2: MULTIMODAL IMAGE QUESTION-ANSWERING
# ==========================================

class QAInput(BaseModel):
    image_base64: Optional[str] = None
    image: Optional[str] = None
    question: str

class QAOutput(BaseModel):
    answer: str

@app.post("/answer-image", response_model=QAOutput)
async def answer_image(payload: QAInput):
    try:
        b64_str = payload.image_base64 or payload.image
        if not b64_str:
            raise HTTPException(status_code=422, detail="Missing base64 data.")

        if "," in b64_str:
            b64_str = b64_str.split(",")[1]

        image_bytes = base64.b64decode(b64_str)
        image_part = {"mime_type": "image/png", "data": image_bytes}

        system_instruction = (
            "You are a precise document analytics assistant.\n"
            "Analyze the image and answer the user's question explicitly based on the content.\n"
            "Strict Output Format Rules:\n"
            "1. Return ONLY the final direct answer as a clean string data value.\n"
            "2. If the answer is a numeric value, return ONLY the raw digit string (e.g., '4089.35').\n"
            "3. Do NOT include currency symbols, units, commas, or full sentences."
        )

        # Updated to explicitly use the models/ prefix to solve the 404 issue
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content([
            system_instruction,
            image_part,
            f"Question: {payload.question}"
        ])

        return QAOutput(answer=response.text.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failure: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
