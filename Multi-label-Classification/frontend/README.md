# Frontend (React) for Movie Tags Predictor

This folder contains a minimal Vite + React frontend that calls a backend `/predict` endpoint.

Quick start (from this folder):

```powershell
# Install dependencies
npm install
# Start dev server
npm run dev
```

The app expects a local backend endpoint POST /predict that accepts JSON { title: string, overview: string } and returns JSON { tags: ["Action", "Drama"], scores?: { ... } }.

If your backend runs on a different host/port, either configure a proxy (see Vite docs) or change the fetch URL in `src/App.jsx`.

Notes
- This is a minimal app to test the model. Enhance with routing, auth, or advanced UI as needed.
