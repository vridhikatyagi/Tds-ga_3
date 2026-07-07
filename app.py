from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re
import os

# 1. Initialize FastAPI Application
app = FastAPI(title="IITM Fixed Schema Invoice Extraction API")

# 2. Enable CORS (Required for the external Cloudflare Worker grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# 3. Define Request and Response Schemas
class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None

# Helper function to clean and parse numbers
def parse_float(text: str) -> Optional[float]:
    if not text:
        return None
    try:
        # Remove currency symbols, spaces, and commas (handles formats like 1,40,000.00)
        cleaned = re.sub(r'[^\d.]', '', text)
        return float(cleaned) if cleaned else None
    except ValueError:
        return None

# Helper function to normalize various date formats to YYYY-MM-DD
def normalize_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    date_str = date_str.strip()
    
    # Check if already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
        
    # Handle human-readable formats like "15 March 2026"
    months = {
        "january": "01", "jan": "01", "february": "02", "feb": "02",
        "march": "03", "mar": "03", "april": "04", "apr": "04",
        "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
        "august": "08", "aug": "08", "september": "09", "sep": "09",
        "october": "10", "oct": "10", "november": "11", "nov": "11",
        "december": "12", "dec": "12"
    }
    
    parts = re.split(r'[\s,/-]+', date_str)
    if len(parts) >= 3:
        # Assuming format: DD Month YYYY or Month DD YYYY
        day, month, year = None, None, None
        for part in parts:
            part_lower = part.lower()
            if part_lower in months:
                month = months[part_lower]
            elif part.isdigit():
                if len(part) == 4:
                    year = part
                elif len(part) <= 2:
                    if not day:
                        day = part.zfill(2)
                    else:
                        year = part  # fallback for YY
        if year and month and day:
            return f"{year}-{month}-{day}"
            
    return None

# 4. Define the POST Endpoint matching the specification exactly
@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(payload: InvoiceRequest):
    try:
        text = payload.invoice_text
        lines = text.split('\n')
        
        # Heuristics & Regex Mapping
        invoice_no = None
        inv_match = re.search(r'(?i)(?:invoice\s*no|ref|reference|inv\s*#)[:\s\-]+([A-Za-z0-9\/]+)', text)
        if inv_match:
            invoice_no = inv_match.group(1)

        date_val = None
        date_match = re.search(r'(?i)(?:date|issued)[:\s\-]+([0-9a-zA-Z\s,./-]+)', text)
        if date_match:
            date_val = normalize_date(date_match.group(1))

        vendor = None
        vendor_match = re.search(r'(?i)vendor[:\s\-]+([^\n]+)', text)
        if vendor_match:
            vendor = vendor_match.group(1).strip()
        else:
            # Fallback: Check if the first line contains a company name indication
            if lines and ("solutions" in lines[0].lower() or "solutions" in lines[0].lower() or "pvt" in lines[0].lower() or "ltd" in lines[0].lower()):
                vendor = lines[0].split('—')[0].strip()

        amount = None
        amt_match = re.search(r'(?i)subtotal[:\s\-]+([^\n]+)', text)
        if amt_match:
            amount = parse_float(amt_match.group(1))

        tax = None
        tax_match = re.search(r'(?i)(?:gst|tax|igst|cgst|sgst|vat)[^:]*[:\s\-]+([^\n]+)', text)
        if tax_match:
            tax = parse_float(tax_match.group(1))

        currency = "INR" # Default fallback
        curr_match = re.search(r'(?i)currency[:\s\-]+([A-Z]{3})', text)
        if curr_match:
            currency = curr_match.group(1).upper()
        elif "$" in text or "USD" in text:
            currency = "USD"

        return InvoiceResponse(
            invoice_no=invoice_no,
            date=date_val,
            vendor=vendor,
            amount=amount,
            tax=tax,
            currency=currency
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal extraction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Dynamically bind to the port assigned by Render, or fallback to 8000 for local execution
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
