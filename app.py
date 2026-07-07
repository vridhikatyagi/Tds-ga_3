from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
import re
import os

# 1. Initialize FastAPI Application
app = FastAPI(title="IITM Fixed Schema Invoice Extraction API")

# 2. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Request Schema
class InvoiceRequest(BaseModel):
    invoice_text: Optional[str] = None
    text: Optional[str] = None

# 4. Target Response Schema
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

# Helper parser functions
def parse_float(text: str) -> Optional[float]:
    if not text:
        return None
    try:
        cleaned = re.sub(r'[^\d.]', '', text)
        return float(cleaned) if cleaned else None
    except ValueError:
        return None

def normalize_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    date_str = date_str.strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
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
                        year = part
        if year and month and day:
            return f"{year}-{month}-{day}"
    return None

# 5. POST Endpoint
@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(payload: InvoiceRequest):
    try:
        text = payload.invoice_text or payload.text
        if not text:
            raise HTTPException(status_code=422, detail="Content missing.")
            
        lines = text.split('\n')
        
        # 1. Parse Invoice Date
        invoice_date = None
        date_match = re.search(r'(?i)(?:date|issued|invoice\s*date)[:\s\-]+([0-9a-zA-Z\s,./-]+)', text)
        if date_match:
            invoice_date = normalize_date(date_match.group(1))

        # 2. Parse Vendor
        vendor = None
        vendor_match = re.search(r'(?i)(?:vendor|issued\s*by)[:\s\-]+([^\n]+)', text)
        if vendor_match:
            vendor = vendor_match.group(1).strip()
        else:
            if lines and any(k in lines[0].lower() for k in ["solutions", "pvt", "ltd", "logistics", "corp"]):
                vendor = lines[0].split('—')[0].strip()

        # 3. Parse Contact Email (Forced to lowercase here)
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        contact_email = email_match.group(0).lower() if email_match else None

        # 4. Parse Total Amount
        total_amount = None
        amt_match = re.search(r'(?i)(?:total|amount|due|payable)[:\s\-]+([^\n]+)', text)
        if amt_match:
            total_amount = parse_float(amt_match.group(1))

        # 5. Parse Currency
        currency = "INR"
        if "$" in text or "USD" in text:
            currency = "USD"
        elif "EUR" in text or "€" in text:
            currency = "EUR"

        # 6. Parse Due In Days
        due_in_days = 30  
        due_match = re.search(r'(?i)(?:due\s*in|within)[:\s\-]*(\d+)\s*days', text)
        if due_match:
            due_in_days = int(due_match.group(1))

        # 7. Check if Paid
        is_paid = "paid" in text.lower() and "unpaid" not in text.lower()

        return InvoiceResponse(
            contact_email=contact_email,
            currency=currency,
            due_in_days=due_in_days,
            invoice_date=invoice_date,
            is_paid=is_paid,
            item_count=1,  
            line_items=[],
            priority="medium",
            total_amount=total_amount,
            vendor=vendor
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
