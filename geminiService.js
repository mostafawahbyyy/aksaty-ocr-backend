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
    // Switched from gemini-2.5-flash-lite → gemini-2.5-flash because Flash-Lite
    // had reliability issues processing multi-page Arabic property-contract
    // PDFs (silent extraction failures with no error surfaced). Regular Flash
    // is the documented model for PDF document understanding and works well
    // for both photos and PDFs at similar latency. Cost is slightly higher
    // but worth it for the OCR success rate. If this ever needs to revert
    // for cost reasons, swap to gemini-2.5-flash-lite-002 once Google ships
    // that variant with improved document support.
    model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    console.log('[Gemini] Initialized with gemini-2.5-flash');
    return true;
  } catch (err) {
    console.error('[Gemini] Init failed:', err.message);
    return false;
  }
}

// Optimistic: assume key works after init succeeds. The real check happens
// on the first OCR request (which throws API_KEY_INVALID with proper error
// code if the key is bad). validateKey() is a separate optional warmup
// used in local dev; on Vercel serverless we skip it to avoid cold-start
// latency and rely on per-request error handling instead.
let keyValid = true;

/**
 * Check if Gemini is ready (model initialized and key not known-bad)
 */
function isReady() {
  return model !== null && keyValid;
}

/**
 * Validate API key with a lightweight test call. Optional warmup — only
 * called from local dev startup. If it fails, we mark the key invalid so
 * /health reflects the truth and requests fast-fail with a proper code.
 */
async function validateKey() {
  if (!model) { keyValid = false; return false; }
  try {
    await model.countTokens('test');
    keyValid = true;
    console.log('[Gemini] API key validated successfully');
    return true;
  } catch (err) {
    keyValid = false;
    console.error('[Gemini] API key validation FAILED:', err.message);
    return false;
  }
}

function getKeyStatus() {
  if (!model) return 'no_key';
  if (!keyValid) return 'invalid_key';
  return 'valid';
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
- "Q1 2025" → "2025-01-01", "Q2 2025" → "2025-04-01", "Q3 2025" → "2025-07-01", "Q4 2025" → "2025-10-01"
- If date is unreadable, use null

CATEGORY RULES:
- Contains "مقدم" or "down" or "دفعة أولى" or "booking" or "تعاقد" or "حجز" or "عربون" → "downPayment"
- Contains "تسليم" or "delivery" or "استلام" → "delivery"
- Contains "صيانة" or "maintenance" → "maintenance"
- Everything else (قسط, أقساط, installment, numbered payments) → "installment"

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

  // Retry if average confidence is very low (likely bad OCR read)
  if (data.installments.length > 0) {
    const avgConf = data.installments.reduce((s, i) => s + (i.confidence || 0), 0) / data.installments.length;
    if (avgConf < 0.5) {
      console.log(`[Gemini] Low avg confidence (${avgConf.toFixed(2)}), retrying with simpler prompt...`);
      const retryData = await attemptExtraction(imagePart, RETRY_PROMPT);
      if (retryData && retryData.installments) {
        const retryClean = retryData.installments.map(cleanInstallment).filter(i => i.amount > 0);
        const retryAvg = retryClean.length > 0
          ? retryClean.reduce((s, i) => s + (i.confidence || 0), 0) / retryClean.length
          : 0;
        if (retryAvg > avgConf && retryClean.length > 0) {
          console.log(`[Gemini] Retry improved confidence: ${avgConf.toFixed(2)} → ${retryAvg.toFixed(2)}`);
          data.installments = retryClean;
        }
      }
    }
  }

  // Deduplicate by amount+date (Gemini may read the same row twice)
  data.installments = deduplicateInstallments(data.installments);

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
    const msg = (err.message || '').toLowerCase();
    console.error('[Gemini] Extraction error:', err.message);

    // Classify provider errors into distinct, honest error codes
    const ocrErr = (code, message) => {
      const e = new Error(message || code);
      e.errorCode = code;
      return e;
    };

    if (msg.includes('api key') || msg.includes('api_key_invalid') || (msg.includes('invalid') && msg.includes('key')) || msg.includes('permission denied') || msg.includes('forbidden')) {
      throw ocrErr('API_KEY_INVALID', 'Gemini API key is invalid or expired');
    }

    if (msg.includes('quota') || msg.includes('resource exhausted')) {
      throw ocrErr('QUOTA_EXCEEDED', 'Gemini quota exceeded');
    }

    if (msg.includes('429') || msg.includes('too many requests') || msg.includes('rate limit')) {
      throw ocrErr('RATE_LIMITED', 'Gemini rate limited');
    }

    if (msg.includes('safety') || msg.includes('blocked')) {
      throw ocrErr('SAFETY_BLOCKED', 'Content blocked by safety filter');
    }

    if (msg.includes('timeout') || msg.includes('deadline') || msg.includes('timed out')) {
      throw ocrErr('PROVIDER_TIMEOUT', 'Gemini request timed out');
    }

    return null;
  }
}

/**
 * Parse JSON from Gemini response, handling markdown fences and other quirks
 */
function parseGeminiJSON(text) {
  let cleaned = text
    .replace(/```json\s*/gi, '')
    .replace(/```\s*/g, '')
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch (e) {
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
 * Remove duplicate installments by amount+date key
 */
function deduplicateInstallments(installments) {
  const seen = new Set();
  return installments.filter(inst => {
    const key = `${inst.amount}|${inst.date || 'nodate'}`;
    if (seen.has(key)) {
      console.log(`[Gemini] Dropping duplicate: ${key} (${inst.label})`);
      return false;
    }
    seen.add(key);
    return true;
  });
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

function cleanAmount(val) {
  if (typeof val === 'number') return Math.max(0, val);
  if (typeof val === 'string') {
    const num = parseFloat(val.replace(/[,\s٬]/g, '').replace(/[^\d.]/g, ''));
    return isNaN(num) ? 0 : Math.max(0, num);
  }
  return 0;
}

function cleanDate(val) {
  if (!val || val === 'null' || val === 'N/A') return null;

  const str = String(val).trim();

  if (/^\d{4}-\d{2}-\d{2}$/.test(str)) return str;

  const dmyMatch = str.match(/^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})$/);
  if (dmyMatch) {
    const [, d, m, y] = dmyMatch;
    return `${y}-${m.padStart(2, '0')}-${d.padStart(2, '0')}`;
  }

  const myMatch = str.match(/^(\d{1,2})[\/\-.](\d{4})$/);
  if (myMatch) {
    const [, m, y] = myMatch;
    return `${y}-${m.padStart(2, '0')}-01`;
  }

  const qMatch = str.match(/Q(\d)\s*[\/\-.]?\s*(\d{4})/i);
  if (qMatch) {
    const quarter = parseInt(qMatch[1]);
    const year = qMatch[2];
    const month = ((quarter - 1) * 3 + 1).toString().padStart(2, '0');
    return `${year}-${month}-01`;
  }

  const parsed = new Date(str);
  if (!isNaN(parsed.getTime())) {
    return parsed.toISOString().split('T')[0];
  }

  return null;
}

function cleanCategory(category, label) {
  const cat = (category || '').toLowerCase();
  const lbl = (label || '').toLowerCase();

  if (cat === 'downpayment' || cat === 'down_payment' || cat === 'down') return 'downPayment';
  if (cat === 'delivery') return 'delivery';
  if (cat === 'maintenance') return 'maintenance';
  if (cat === 'installment' || cat === 'regular') return 'installment';

  if (/down|مقدم|تعاقد|booking|دفعة أولى|حجز|عربون/i.test(lbl)) return 'downPayment';
  if (/deliver|تسليم|استلام/i.test(lbl)) return 'delivery';
  if (/maint|صيانة/i.test(lbl)) return 'maintenance';

  return 'installment';
}

module.exports = {
  initGemini, isReady, extractPayments, validateKey, getKeyStatus,
  _test: { cleanDate, cleanAmount, cleanCategory, cleanInstallment, deduplicateInstallments },
};
