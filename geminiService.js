/**
 * Gemini AI OCR Service
 * Sends preprocessed images to Google Gemini for payment schedule extraction.
 * Includes production-grade prompt engineering, retry logic, and validation.
 */

const { GoogleGenerativeAI } = require('@google/generative-ai');

let model = null;

/**
 * Initialize the Gemini model
 * @param {string} apiKey
 */
function initGemini(apiKey) {
  if (!apiKey) {
    console.warn('[Gemini] No API key provided — OCR will not work');
    return false;
  }
  try {
    const genAI = new GoogleGenerativeAI(apiKey);
    model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });
    console.log('[Gemini] Initialized with gemini-2.5-flash-lite');
    return true;
  } catch (err) {
    console.error('[Gemini] Init failed:', err.message);
    return false;
  }
}

/**
 * Check if Gemini is ready
 */
function isReady() {
  return model !== null;
}

// The core extraction prompt
const EXTRACTION_PROMPT = `You are an expert document reader specializing in Egyptian real estate contracts and payment schedules.

TASK: Extract the complete payment schedule from this document image. The document is an Egyptian property contract written in Arabic and/or English.

INSTRUCTIONS:
1. Find ALL payment rows in the table (down payments, installments, delivery payments, maintenance fees).
2. Extract each payment's label, amount in EGP, and due date.
3. Convert ALL Arabic numerals (٠١٢٣٤٥٦٧٨٩) to English numerals (0123456789).
4. Normalize ALL dates to ISO format YYYY-MM-DD.
5. Clean amounts: remove commas, currency symbols, spaces. Return as plain numbers.
6. Categorize each payment as one of: "downPayment", "installment", "delivery", "maintenance".
7. If the document contains property/unit info (name, project, delivery date), extract that too.

DATE FORMAT HANDLING:
- "21-09-2025" or "21/09/2025" → "2025-09-21"
- "2025-09-21" → "2025-09-21" (already ISO)
- "٢١/٠٩/٢٠٢٥" → "2025-09-21"
- If only month/year, use first of month: "03/2026" → "2026-03-01"
- If date is unreadable, use null

CATEGORY RULES:
- Contains "مقدم" or "down" or "دفعة أولى" or "booking" or "تعاقد" → "downPayment"
- Contains "تسليم" or "delivery" or "استلام" → "delivery"
- Contains "صيانة" or "maintenance" → "maintenance"
- Everything else (قسط, installment, numbered payments) → "installment"

RESPOND WITH ONLY THIS JSON (no markdown, no backticks, no explanation):
{
  "unit": {
    "name": "unit name or number if visible, empty string if not",
    "project": "project/compound name if visible, empty string if not",
    "delivery_date": "YYYY-MM-DD if visible, null if not"
  },
  "installments": [
    {
      "label": "Down Payment",
      "amount": 1760983,
      "date": "2025-09-21",
      "category": "downPayment",
      "confidence": 0.95
    }
  ],
  "totals": {
    "scheduled_total": 19670813
  }
}

CRITICAL RULES:
- Return ONLY valid JSON. No markdown code fences. No explanation text.
- Every amount must be a NUMBER (not a string).
- Every date must be "YYYY-MM-DD" string or null.
- Include ALL rows from the table, not just the first few.
- If you cannot read a value, set amount to 0 and confidence to 0.3.
- scheduled_total should be the sum of all amounts if visible, or 0 if not shown.`;

// Simplified retry prompt if first attempt fails
const RETRY_PROMPT = `Look at this document image carefully. It contains a payment schedule table.
Extract EVERY row from the payment table.
Return ONLY a JSON object with this exact structure (no markdown, no backticks):
{"installments":[{"label":"Payment name","amount":0,"date":"YYYY-MM-DD","category":"installment","confidence":0.8}],"totals":{"scheduled_total":0},"unit":{"name":"","project":"","delivery_date":null}}
Include ALL rows. Amounts as numbers. Dates as YYYY-MM-DD or null.`;

/**
 * Extract payment data from an image using Gemini
 *
 * @param {Buffer} imageBuffer - Processed image buffer
 * @param {string} mimeType - Image MIME type
 * @returns {Promise<Object>} Extracted data
 */
async function extractPayments(imageBuffer, mimeType) {
  if (!model) throw new Error('Gemini not initialized');

  const imagePart = {
    inlineData: {
      data: imageBuffer.toString('base64'),
      mimeType: mimeType,
    },
  };

  // First attempt (will throw if quota exceeded)
  let data = await attemptExtraction(imagePart, EXTRACTION_PROMPT);

  // If first attempt returned no installments, retry with simpler prompt
  if (!data || !data.installments || data.installments.length === 0) {
    console.log('[Gemini] First attempt returned no data, retrying with simpler prompt...');
    data = await attemptExtraction(imagePart, RETRY_PROMPT);
  }

  if (!data || !data.installments || data.installments.length === 0) {
    return null;
  }

  // Validate and clean the results
  const beforeCount = data.installments.length;
  data.installments = data.installments
    .map(cleanInstallment)
    .filter(inst => inst.amount > 0);
  console.log(`[Gemini] After cleaning: ${data.installments.length}/${beforeCount} installments kept`);

  // Recalculate total if not provided
  if (!data.totals || !data.totals.scheduled_total) {
    data.totals = {
      scheduled_total: data.installments.reduce((sum, i) => sum + (i.amount || 0), 0),
    };
  }

  // Ensure unit exists
  if (!data.unit) {
    data.unit = { name: '', project: '', delivery_date: null };
  }

  return data;
}

/**
 * Attempt a single extraction with a given prompt
 */
async function attemptExtraction(imagePart, prompt) {
  try {
    const result = await model.generateContent([prompt, imagePart]);
    const response = await result.response;
    const text = response.text();

    console.log('[Gemini] Raw response (first 2000 chars):');
    console.log(text.slice(0, 2000));

    const parsed = parseGeminiJSON(text);
    if (parsed) {
      const count = parsed.installments ? parsed.installments.length : 0;
      console.log(`[Gemini] Parsed OK: ${count} installments found`);
    } else {
      console.log('[Gemini] Parse returned null');
    }
    return parsed;
  } catch (err) {
    const msg = err.message || '';
    console.error('[Gemini] Extraction error:', msg);

    // Detect quota/rate limit errors and throw so caller can show proper message
    if (msg.includes('429') || msg.includes('quota') || msg.includes('Too Many Requests') || msg.includes('rate')) {
      const quotaErr = new Error('QUOTA_EXCEEDED');
      quotaErr.isQuota = true;
      throw quotaErr;
    }

    return null;
  }
}

/**
 * Parse JSON from Gemini response, handling markdown fences and other quirks
 */
function parseGeminiJSON(text) {
  // Strip markdown code fences
  let cleaned = text
    .replace(/```json\s*/gi, '')
    .replace(/```\s*/g, '')
    .trim();

  // Try direct parse
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    // Try to find JSON object in the text
    const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        return JSON.parse(jsonMatch[0]);
      } catch (e2) {
        console.error('[Gemini] JSON parse failed. Raw text:', text.slice(0, 500));
        return null;
      }
    }
    console.error('[Gemini] No JSON found in response:', text.slice(0, 500));
    return null;
  }
}

/**
 * Clean and validate a single installment entry
 */
function cleanInstallment(inst) {
  return {
    label: String(inst.label || 'Payment').trim(),
    amount: cleanAmount(inst.amount),
    date: cleanDate(inst.date),
    category: cleanCategory(inst.category, inst.label),
    confidence: Math.min(1, Math.max(0, Number(inst.confidence) || 0.8)),
    frequency: 'once',
  };
}

/**
 * Clean amount value
 */
function cleanAmount(val) {
  if (typeof val === 'number') return Math.max(0, val);
  if (typeof val === 'string') {
    // Remove commas, currency, spaces, Arabic chars
    const num = parseFloat(val.replace(/[,\s٬]/g, '').replace(/[^\d.]/g, ''));
    return isNaN(num) ? 0 : Math.max(0, num);
  }
  return 0;
}

/**
 * Clean and normalize date to YYYY-MM-DD
 */
function cleanDate(val) {
  if (!val || val === 'null' || val === 'N/A') return null;

  const str = String(val).trim();

  // Already ISO format
  if (/^\d{4}-\d{2}-\d{2}$/.test(str)) return str;

  // DD/MM/YYYY or DD-MM-YYYY
  const dmyMatch = str.match(/^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})$/);
  if (dmyMatch) {
    const [, d, m, y] = dmyMatch;
    return `${y}-${m.padStart(2, '0')}-${d.padStart(2, '0')}`;
  }

  // MM/YYYY
  const myMatch = str.match(/^(\d{1,2})[\/\-.](\d{4})$/);
  if (myMatch) {
    const [, m, y] = myMatch;
    return `${y}-${m.padStart(2, '0')}-01`;
  }

  // Try native Date parse as last resort
  const parsed = new Date(str);
  if (!isNaN(parsed.getTime())) {
    return parsed.toISOString().split('T')[0];
  }

  return null;
}

/**
 * Normalize category string
 */
function cleanCategory(category, label) {
  const cat = (category || '').toLowerCase();
  const lbl = (label || '').toLowerCase();

  if (cat === 'downpayment' || cat === 'down_payment' || cat === 'down') return 'downPayment';
  if (cat === 'delivery') return 'delivery';
  if (cat === 'maintenance') return 'maintenance';
  if (cat === 'installment' || cat === 'regular') return 'installment';

  // Infer from label if category is missing
  if (/down|مقدم|تعاقد|booking|دفعة أولى/i.test(lbl)) return 'downPayment';
  if (/deliver|تسليم|استلام/i.test(lbl)) return 'delivery';
  if (/maint|صيانة/i.test(lbl)) return 'maintenance';

  return 'installment';
}

module.exports = { initGemini, isReady, extractPayments };
