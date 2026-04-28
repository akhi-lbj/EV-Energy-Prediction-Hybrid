import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import util from 'util';
import path from 'path';

const execAsync = util.promisify(exec);

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const mode = searchParams.get('mode');
  const idx = searchParams.get('idx');

  try {
    const pythonScriptDir = path.join(process.cwd(), '..', '..');
    
    let command = '';
    if (mode === 'fetch') {
      command = `uv run python predict_api.py --mode fetch`;
    } else if (mode === 'predict' && idx !== null) {
      command = `uv run python predict_api.py --mode predict --idx ${idx}`;
    } else {
      return NextResponse.json({ error: 'Invalid mode or missing parameters' }, { status: 400 });
    }

    const { stdout, stderr } = await execAsync(command, { cwd: pythonScriptDir, timeout: 60000 });
    
    try {
      const jsonStart = stdout.indexOf('{');
      const jsonEnd = stdout.lastIndexOf('}') + 1;
      const jsonStr = stdout.substring(jsonStart, jsonEnd);
      
      const predictionData = JSON.parse(jsonStr);
      return NextResponse.json(predictionData);
    } catch (parseError) {
      console.error("Failed to parse Python output:", stdout);
      return NextResponse.json({ error: 'Failed to parse model output' }, { status: 500 });
    }

  } catch (error: any) {
    console.error("API Error executing Python:", error);
    return NextResponse.json({ error: 'Failed to run machine learning model', details: error.message }, { status: 500 });
  }
}
