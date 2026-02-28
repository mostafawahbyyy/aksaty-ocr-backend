/**
 * Image Preprocessor
 * Uses sharp to optimize images for Gemini OCR processing.
 * Handles EXIF rotation, resizing, contrast enhancement, and sharpening.
 */

const sharp = require('sharp');

const MAX_WIDTH = 2000;
const MAX_HEIGHT = 2000;
const JPEG_QUALITY = 85;

/**
 * Preprocess an image buffer for OCR.
 * Returns the processed JPEG buffer and its mime type.
 *
 * @param {Buffer} inputBuffer - Raw image file buffer
 * @returns {Promise<{ buffer: Buffer, mimeType: string }>}
 */
async function preprocessImage(inputBuffer) {
  try {
    const metadata = await sharp(inputBuffer).metadata();
    console.log(`[Preprocessor] Input: ${metadata.width}x${metadata.height}, format=${metadata.format}`);

    const processed = await sharp(inputBuffer)
      // Fix orientation from EXIF data (handles rotated phone photos)
      .rotate()
      // Resize if too large, preserving aspect ratio
      .resize({
        width: MAX_WIDTH,
        height: MAX_HEIGHT,
        fit: 'inside',
        withoutEnlargement: true,
      })
      // Normalize contrast (stretches histogram to full range)
      .normalize()
      // Mild sharpen to improve text edges
      .sharpen({ sigma: 1.0, m1: 0.5, m2: 0.5 })
      // Output as JPEG
      .jpeg({ quality: JPEG_QUALITY })
      .toBuffer();

    const outMeta = await sharp(processed).metadata();
    console.log(`[Preprocessor] Output: ${outMeta.width}x${outMeta.height}, size=${(processed.length / 1024).toFixed(0)}KB`);

    return { buffer: processed, mimeType: 'image/jpeg' };
  } catch (err) {
    console.error('[Preprocessor] Failed, using original:', err.message);
    // If preprocessing fails, return original buffer as-is
    return { buffer: inputBuffer, mimeType: 'image/jpeg' };
  }
}

module.exports = { preprocessImage };
