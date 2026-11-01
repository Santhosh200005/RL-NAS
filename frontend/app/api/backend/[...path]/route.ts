import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = (process.env.RLNAS_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const API_KEY = process.env.RLNAS_API_KEY;

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const target = `${BACKEND_URL}/${path.join('/')}${request.nextUrl.search}`;
  const headers = new Headers();
  const contentType = request.headers.get('content-type');
  if (contentType) headers.set('content-type', contentType);
  if (API_KEY) headers.set('X-API-Key', API_KEY);

  const body = ['GET', 'HEAD'].includes(request.method) ? undefined : await request.arrayBuffer();
  const response = await fetch(target, { method: request.method, headers, body, cache: 'no-store' });
  const responseHeaders = new Headers();
  const responseType = response.headers.get('content-type');
  if (responseType) responseHeaders.set('content-type', responseType);
  return new NextResponse(response.body, { status: response.status, headers: responseHeaders });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
