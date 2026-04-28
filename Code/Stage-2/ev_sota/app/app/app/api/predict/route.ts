/**
 * /api/predict — thin proxy to the FastAPI backend.
 *
 * Set NEXT_PUBLIC_API_URL (or API_URL) in .env.local to your Render URL:
 *   API_URL=https://ev-energy-prediction-api.onrender.com
 *
 * Falls back to http://localhost:8000 for local development.
 */
import { NextResponse } from 'next/server';

const BACKEND = process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const mode = searchParams.get('mode');
  const idx  = searchParams.get('idx');

  let upstreamUrl: string;

  if (mode === 'fetch') {
    upstreamUrl = `${BACKEND}/predict/fetch`;
  } else if (mode === 'predict' && idx !== null) {
    upstreamUrl = `${BACKEND}/predict/run?idx=${idx}`;
  } else {
    return NextResponse.json(
      { error: 'Invalid mode or missing parameters' },
      { status: 400 }
    );
  }

  try {
    const res = await fetch(upstreamUrl, { cache: 'no-store' });

    if (!res.ok) {
      const text = await res.text();
      console.error(`Backend error (${res.status}):`, text);
      return NextResponse.json(
        { error: 'Backend returned an error', details: text },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);

  } catch (err: any) {
    console.error('Failed to reach FastAPI backend:', err);
    return NextResponse.json(
      { error: 'Could not reach the prediction backend', details: err.message },
      { status: 502 }
    );
  }
}
