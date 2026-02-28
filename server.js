/**
 * AKSATY OCR Backend Server
 *
 * Gemini-only OCR with sharp image preprocessing.
 * Accepts contract photos/PDFs, extracts payment schedules.
 *
 * Environment Variables:
 *   GEMINI_API_KEY - Google Gemini API key (required)
 *   PORT           - Server port (default: 5001)
 *   NODE_ENV       - "production" enables auth enforcement
 */

require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const rateLimit = require('express-rate-limit');

const { preprocessImage } = require('./imagePreprocessor');
const { initGemini, isReady, extractPayments } = require('./geminiService');
const { verifyFirebaseToken } = require('./authMiddleware');

const app = express();
const PORT = process.env.PORT || 5001;
const IS_PRODUCTION = process.env.NODE_ENV === 'production';

// Multer config — memory storage (required for Vercel serverless; no disk access)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB
});

// ─── Rate Limiting ───────────────────────────────────────

// General API rate limit: 100 requests per 15 minutes per IP
const generalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
  message: { ok: false, error: 'Too many requests. Please try again later.' },
});

// OCR rate limit: 10 requests per minute per IP (OCR is expensive)
const ocrLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { ok: false, error: 'Too many scan requests. Please wait a minute and try again.' },
});

// ─── Middleware ──────────────────────────────────────────

// CORS: restrict in production, allow all in development
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',')
  : ['*'];

app.use(cors({
  origin: IS_PRODUCTION ? allowedOrigins : '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept', 'X-Request-ID', 'Authorization'],
}));
app.use(express.json());
app.use(generalLimiter);

// Request logging
app.use((req, res, next) => {
  const id = req.headers['x-request-id'] || `srv_${Date.now()}`;
  req.requestId = id;
  console.log(`[${id}] ${req.method} ${req.path}`);
  next();
});

// ─── Auth Middleware (conditional) ───────────────────────
// In production, all OCR endpoints require Firebase auth.
// In development, auth is optional (for easier local testing).

const authMiddleware = IS_PRODUCTION
  ? verifyFirebaseToken
  : (req, res, next) => {
      // In dev, try to verify token if provided, but don't require it
      const authHeader = req.headers.authorization;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        verifyFirebaseToken(req, res, next);
      } else {
        req.user = { uid: 'dev-user' };
        next();
      }
    };

// ─── Health ───────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'aksaty-ocr',
    version: '3.1.0',
    gemini: isReady(),
    auth: IS_PRODUCTION ? 'required' : 'optional',
    timestamp: new Date().toISOString(),
  });
});

// ─── OCR Endpoint ─────────────────────────────────────────
// Serves both paths so mobile doesn't need changes

const handleScan = async (req, res) => {
  const rid = req.requestId;
  const userId = req.user?.uid || 'unknown';

  if (!isReady()) {
    return res.status(503).json({
      ok: false,
      error: 'OCR service not configured. Set GEMINI_API_KEY in .env',
    });
  }

  if (!req.file) {
    return res.status(400).json({
      ok: false,
      error: 'No file uploaded. Please select an image or PDF.',
    });
  }

  const file = req.file;
  console.log(`[${rid}] User: ${userId} | File: ${file.originalname || 'unknown'} (${(file.size / 1024).toFixed(0)}KB, ${file.mimetype})`);

  try {
    // Use in-memory buffer directly (no disk I/O on Vercel)
    const rawBuffer = req.file.buffer;

    // Preprocess image
    console.log(`[${rid}] Preprocessing image...`);
    const { buffer: processedBuffer, mimeType } = await preprocessImage(rawBuffer);

    // Send to Gemini
    console.log(`[${rid}] Sending to Gemini...`);
    const startTime = Date.now();
    const data = await extractPayments(processedBuffer, mimeType);
    const elapsed = Date.now() - startTime;
    console.log(`[${rid}] Gemini responded in ${elapsed}ms`);

    // No data extracted
    if (!data || !data.installments || data.installments.length === 0) {
      console.log(`[${rid}] No payments found`);
      return res.json({
        ok: false,
        error: 'No payment table found in the document. Make sure the entire payment schedule is visible and the photo is clear.',
      });
    }

    console.log(`[${rid}] Success: ${data.installments.length} payments extracted`);

    // Return in the format the mobile app expects
    return res.json({
      ok: true,
      raw_text: '',
      data: {
        unit: data.unit || { name: '', project: '', delivery_date: null },
        add_ons: { maintenance: 0, garage: 0 },
        installments: data.installments,
        totals: data.totals || { scheduled_total: 0 },
      },
    });

  } catch (err) {
    console.error(`[${rid}] Error:`, err.message);

    // User-friendly error messages
    let userMessage = 'Could not read the document. Try taking a clearer photo with better lighting.';
    if (err.isQuota || err.message === 'QUOTA_EXCEEDED') {
      userMessage = 'AI service is temporarily busy (rate limit). Please wait 1 minute and try again.';
    } else if (err.message.includes('Could not process')) {
      userMessage = 'Image format not supported. Please use JPEG, PNG, or PDF.';
    } else if (err.message.includes('quota') || err.message.includes('rate') || err.message.includes('429')) {
      userMessage = 'AI service is temporarily busy (rate limit). Please wait 1 minute and try again.';
    } else if (err.message.includes('safety')) {
      userMessage = 'Could not process this image. Please try a different photo.';
    }

    return res.status(500).json({
      ok: false,
      error: userMessage,
    });
  }
};

// Both paths: rate limit + auth + file upload + handler
app.post('/api/ocr/local', ocrLimiter, authMiddleware, upload.single('contract'), handleScan);
app.post('/api/scan-contract', ocrLimiter, authMiddleware, upload.single('contract'), handleScan);

// ─── Startup ──────────────────────────────────────────────

// Initialize Gemini
const geminiKey = process.env.GEMINI_API_KEY || process.env.GEN_AI_KEY;
const geminiOk = initGemini(geminiKey);

// Start server only when running locally (not on Vercel)
if (process.env.VERCEL !== '1') {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n=== AKSATY OCR Server ===`);
    console.log(`Port:   ${PORT}`);
    console.log(`Mode:   ${IS_PRODUCTION ? 'PRODUCTION' : 'DEVELOPMENT'}`);
    console.log(`Auth:   ${IS_PRODUCTION ? 'REQUIRED' : 'OPTIONAL'}`);
    console.log(`Gemini: ${geminiOk ? 'Enabled' : 'DISABLED (no API key)'}`);
    console.log(`Limits: 100 req/15min (general), 10 req/min (OCR)`);
    console.log(`Routes: POST /api/ocr/local, POST /api/scan-contract`);
    console.log(`Health: GET /health`);
    console.log(`========================\n`);
  });
}

// Export for Vercel serverless
module.exports = app;
