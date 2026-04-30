/**
 * Image Preprocessor
 * Uses sharp to optimize images for Gemini OCR processing.
 * Handles EXIF rotation, resizing, contrast enhancement, and sharpening.
 * Detects PDFs and passes them through directly (Gemini supports application/pdf).
 */

const sharp = require('sharp');

const MAX_WIDTH = 2000;
const MAX_HEIGHT = 2000;
const JPEG_QUALITY = 85;
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB

/**
 * Check if a buffer is a PDF (starts with %PDF)
 */
function isPDF(buffer) {
  return buffer.length >= 4 &&
    buffer[0] === 0x25 && buffer[1] === 0x50 &&
    buffer[2] === 0x44 && buffer[3] === 0x46;
}

/**
 * Preprocess a file buffer for OCR.
 * - PDFs: passed through directly with application/pdf mime type (Gemini handles them natively)
 * - Images: preprocessed with sharp (rotation, resize, contrast, sharpen)
 *
 * @param {Buffer} inputBuffer - Raw file buffer
 * @returns {Promise<{ buffer: Buffer, mimeType: string, isPdf: boolean }>}
 */
async function preprocessImage(inputBuffer) {
  // Check file size
  if (inputBuffer.length > MAX_FILE_SIZE) {
    const err = new Error(`File too large (${(inputBuffer.length / 1024 / 1024).toFixed(1)}MB). Max ${MAX_FILE_SIZE / 1024 / 1024}MB.`);
    err.errorCode = 'FILE_TOO_LARGE';
    throw err;
  }

  // PDF: pass through directly — Gemini supports application/pdf natively
  if (isPDF(inputBuffer)) {
    console.log(`[Preprocessor] PDF detected (${(inputBuffer.length / 1024).toFixed(0)}KB) — passing to Gemini directly`);
    return { buffer: inputBuffer, mimeType: 'application/pdf', isPdf: true };
  }

  // Image: preprocess with sharp
  try {
    const metadata = await sharp(inputBuffer).metadata();
    console.log(`[Preprocessor] Input: ${metadata.width}x${metadata.height}, format=${metadata.format}`);

    const processed = await sharp(inputBuffer)
      .rotate()
      .resize({
        width: MAX_WIDTH,
        height: MAX_HEIGHT,
        fit: 'inside',
        withoutEnlargement: true,
      })
      .normalize()
      .sharpen({ sigma: 1.0, m1: 0.5, m2: 0.5 })
      .jpeg({ quality: JPEG_QUALITY })
      .toBuffer();

    const outMeta = await sharp(processed).metadata();
    console.log(`[Preprocessor] Output: ${outMeta.width}x${outMeta.height}, size=${(processed.length / 1024).toFixed(0)}KB`);

    return { buffer: processed, mimeType: 'image/jpeg', isPdf: false };
  } catch (err) {
    console.error('[Preprocessor] Failed:', err.message);
    // If it's not a recognized image format, throw a clear error
    if (err.message.includes('unsupported') || err.message.includes('Input file is missing')) {
      const procErr = new Error('Unsupported file format. Please use JPEG, PNG, or PDF.');
      procErr.errorCode = 'UNSUPPORTED_FORMAT';
      throw procErr;
    }
    // Don't send unprocessable data to Gemini — fail clearly
    const fallbackErr = new Error('Could not process this image. Please use a clear JPEG, PNG, or PDF.');
    fallbackErr.errorCode = 'UNSUPPORTED_FORMAT';
    throw fallbackErr;
  }
}

module.exports = { preprocessImage, _test: { isPDF } };
