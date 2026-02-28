/**
 * Firebase Authentication Middleware
 * ===================================
 * Verifies Firebase ID tokens on incoming requests.
 * Attaches decoded user info to req.user.
 */

const admin = require('firebase-admin');

// Initialize Firebase Admin SDK (uses default credentials or service account)
if (!admin.apps.length) {
  admin.initializeApp({
    projectId: process.env.FIREBASE_PROJECT_ID || 'aksaty-c0bbc',
  });
}

/**
 * Middleware that verifies Firebase ID token from Authorization header.
 * Expected header format: "Bearer <idToken>"
 *
 * On success: sets req.user = { uid, email, ... } and calls next()
 * On failure: returns 401 Unauthorized
 */
async function verifyFirebaseToken(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({
      ok: false,
      error: 'Unauthorized. Missing or invalid Authorization header.',
    });
  }

  const idToken = authHeader.split('Bearer ')[1];

  try {
    const decodedToken = await admin.auth().verifyIdToken(idToken);
    req.user = decodedToken;
    next();
  } catch (error) {
    console.error('[Auth] Token verification failed:', error.message);
    return res.status(401).json({
      ok: false,
      error: 'Unauthorized. Invalid or expired token.',
    });
  }
}

module.exports = { verifyFirebaseToken };
