from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import google.generativeai as genai
import json
import os

# 1. Initialize FastAPI Application
app = FastAPI(title="IITM Invoice Extraction API")

# 2. Enable CORS (Required for the external Cloudflare Worker grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 3. Configure Gemini API
# Securely set your key as an environment variable or paste it directly below
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_ACTUAL_GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 4. Define Request and Response Schemas using Pydantic
class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = Field(None, description="The unique invoice number or reference.")
    date: Optional[str] = Field(None, description="The date formatted strictly as ISO format YYYY-MM-DD.")
    vendor: Optional[str] = Field(None, description="The name of the vendor or supplier.")
    amount: Optional[float] = Field(None, description="The subtotal before tax as a raw float number.")
    tax: Optional[float] = Field(None, description="The isolated tax amount as a raw float number.")
    currency: Optional[str] = Field(None, description="The currency code flag (e.g., INR, USD).")

# 5. Define the POST Endpoint matching the specification exactly
@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(payload: InvoiceRequest):
    try:
        # Utilize gemini-1.5-flash for rapid schema-constrained textual extraction
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = (
            "You are a precise financial data extraction assistant.\n"
            "Analyze the following raw invoice text and extract the target fields.\n\n"
            f"Invoice text:\n{payload.invoice_text}\n\n"
            "Strict Instructions:\n"
            "1. Extract 'invoice_no' (look for fields like Ref, Invoice No, Receipt #).\n"
            "2. Extract 'date' and normalize it strictly into ISO format YYYY-MM-DD. If a text format like '15 March 2026' is provided, resolve it to '2026-03-15'.\n"
            "3. Extract 'vendor' company name.\n"
            "4. Extract 'amount' representing the subtotal BEFORE taxes. Return as a raw floating-point number without commas or text symbols.\n"
            "5. Extract 'tax' isolated tax value only. Return as a raw floating-point number.\n"
            "6. Extract the standard 3-letter currency abbreviation code flag (e.g., INR, USD).\n"
            "7. If any key cannot be found in the text, map its value to null.\n"
            "8. Return ONLY a valid JSON object matching the requested schema. No conversation or markdown ticks."
        )

        # Enforce structural output via standard model structure params
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Parse output string mapping natively back to output object
        extracted_data = json.loads(response.text.strip())
        return InvoiceResponse(**extracted_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Dynamically bind to the port assigned by Render, or fallback to 8000 for local execution
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)