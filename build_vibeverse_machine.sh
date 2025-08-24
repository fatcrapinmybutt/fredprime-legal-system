#!/usr/bin/env bash
set -euo pipefail

# =============================
# VIBEVERSE — FULL MACHINE BUILD
# =============================
# One-command scaffold + (optional) deploy.
# Requirements (install/log in before running for full automation):
# - git, node, npm, jq, curl
# - gh (GitHub CLI, authenticated)  -> https://cli.github.com/
# - vercel (Vercel CLI, logged in)  -> npm i -g vercel
# - railway (Railway CLI, optional) -> npm i -g @railway/cli
# - python3, pip (for FastAPI local/dev)
#
# Usage (examples):
#   chmod +x build_vibeverse_machine.sh
#   ./build_vibeverse_machine.sh \
#     --repo vibeverse \
#     --owner YOUR_GITHUB_USER_OR_ORG \
#     --vercel-project vibeverse-storefront \
#     --stripe-pk pk_test_xxx --stripe-sk sk_test_xxx \
#     --price-001 price_123 --price-002 price_456 --price-003 price_789 --price-bundle price_bundle \
#     --status-webhook https://hooks.slack.com/services/XXX/YYY/ZZZ \
#     --api-deploy railway
#
# Notes:
# - If a tool is missing or not logged in, the script will still generate the repo and print next steps.
# - Windows: run via WSL or Git Bash.

REPO_NAME="vibeverse"
GITHUB_OWNER=""
VERCEL_PROJECT=""
STRIPE_PK=""
STRIPE_SK=""
PRICE_001=""
PRICE_002=""
PRICE_003=""
PRICE_BUNDLE=""
STATUS_WEBHOOK_URL=""
API_DEPLOY="none"   # railway|none

# --- Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_NAME="$2"; shift 2;;
    --owner) GITHUB_OWNER="$2"; shift 2;;
    --vercel-project) VERCEL_PROJECT="$2"; shift 2;;
    --stripe-pk) STRIPE_PK="$2"; shift 2;;
    --stripe-sk) STRIPE_SK="$2"; shift 2;;
    --price-001) PRICE_001="$2"; shift 2;;
    --price-002) PRICE_002="$2"; shift 2;;
    --price-003) PRICE_003="$2"; shift 2;;
    --price-bundle) PRICE_BUNDLE="$2"; shift 2;;
    --status-webhook) STATUS_WEBHOOK_URL="$2"; shift 2;;
    --api-deploy) API_DEPLOY="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "==> BUILD START"
echo "Repo: $REPO_NAME"
[[ -n "$GITHUB_OWNER" ]] && echo "GitHub owner: $GITHUB_OWNER" || echo "GitHub owner: (none, local only)"
[[ -n "$VERCEL_PROJECT" ]] && echo "Vercel project: $VERCEL_PROJECT" || echo "Vercel project: (link later)"
echo "API deploy: $API_DEPLOY"

# --- Create repo root (fresh)
if [[ -d "$REPO_NAME" ]]; then
  echo "Removing existing dir $REPO_NAME"
  rm -rf "$REPO_NAME"
fi
mkdir -p "$REPO_NAME"
cd "$REPO_NAME"

# --- Write README
cat > README.md << 'EOF'
# VIBEVERSE Machine

Monorepo with **Next.js storefront** (Vercel-ready), **FastAPI API** (Render/Railway/local), **Stripe Checkout**, and **hourly notifier** (GitHub Actions).

## Quick Start
1) Push to GitHub (or run locally)  
2) Import repo to **Vercel** for the frontend  
3) Deploy **api/** to Railway/Render/Fly (or run locally)  
4) Add Stripe test keys to Vercel env (`pk_test`, `sk_test`)  
5) Add `STATUS_WEBHOOK_URL` secret in GitHub for hourly pings

## Structure
- `frontend/` — Next.js 14 app (pages router) with Stripe checkout
- `api/` — FastAPI app (endpoints for capsules, affiliates, predict, vibeDNA)
- `.github/workflows/hourly_status.yml` — Hourly status webhook pings
- `infra/` — Railway/Render templates
- `scripts/` — helper docs
EOF

# --- Frontend files
mkdir -p frontend/pages/api
mkdir -p frontend
cat > frontend/package.json << 'EOF'
{
  "name": "vibeverse-frontend",
  "private": true,
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.2.5",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "stripe": "15.12.0"
  },
  "devDependencies": {
    "typescript": "5.5.4",
    "@types/react": "18.2.66",
    "@types/node": "20.14.10",
    "eslint": "8.57.0",
    "eslint-config-next": "14.2.5"
  }
}
EOF

cat > frontend/.env.example << EOF
NEXT_PUBLIC_API_BASE=
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=${STRIPE_PK}
STRIPE_SECRET_KEY=${STRIPE_SK}
PRICE_CAPSULE_001=${PRICE_001}
PRICE_CAPSULE_002=${PRICE_002}
PRICE_CAPSULE_003=${PRICE_003}
NEXT_PUBLIC_PRICE_BUNDLE=${PRICE_BUNDLE}
EOF

cat > frontend/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": false,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
EOF

cat > frontend/next-env.d.ts << 'EOF'
/// <reference types="next" />
/// <reference types="next/image-types/global" />
EOF

cat > frontend/next.config.mjs << 'EOF'
export default { reactStrictMode: true };
EOF

cat > frontend/styles.css << 'EOF'
:root{--bg:#0b0b10;--fg:#e8e8f0;--muted:#9aa0a6;--accent:#7cf1c8}
html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}
a{color:var(--accent);text-decoration:none}
.container{max-width:980px;margin:0 auto;padding:24px}
.card{border:1px solid #222;border-radius:12px;padding:16px;background:#101018}
.btn{display:inline-block;padding:10px 16px;border-radius:8px;background:var(--accent);color:#111;font-weight:600}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
h1,h2{margin:8px 0 12px 0}
small{color:var(--muted)}
EOF

mkdir -p frontend/pages
cat > frontend/pages/_app.tsx << 'EOF'
import type { AppProps } from 'next/app';
import '../styles.css';
export default function MyApp({ Component, pageProps }: AppProps) { return <Component {...pageProps} />; }
EOF

cat > frontend/pages/index.tsx << 'EOF'
import Link from 'next/link';
export default function Home() {
  return (
    <div className="container">
      <h1>VIBEVERSE</h1>
      <p>Capsule storefront starter. Choose a section:</p>
      <div className="grid">
        <div className="card"><h2>Capsules</h2><p>Browse & buy.</p><Link className="btn" href="/capsules">Open</Link></div>
        <div className="card"><h2>Bundle</h2><p>All-in-one value pack.</p><Link className="btn" href="/bundle">Open</Link></div>
        <div className="card"><h2>VIBE DNA</h2><p>Quiz + funnel.</p><Link className="btn" href="/dna">Open</Link></div>
        <div className="card"><h2>Affiliate</h2><p>Earn by sharing.</p><Link className="btn" href="/affiliate">Open</Link></div>
        <div className="card"><h2>Drops</h2><p>Upcoming launches.</p><Link className="btn" href="/drops">Open</Link></div>
      </div>
    </div>
  )
}
EOF

cat > frontend/pages/capsules.tsx << 'EOF'
import { useEffect, useState } from 'react';
type Capsule = { id:string; name:string; priceId?:string; priceLabel?:string; };
export default function Capsules() {
  const [data,setData] = useState<Capsule[]>([]);
  useEffect(()=>{
    const base = process.env.NEXT_PUBLIC_API_BASE;
    const url = base ? `${base}/capsules/latest` : '/api/stub-capsules';
    fetch(url).then(r=>r.json()).then(setData).catch(()=>setData([]));
  },[]);
  async function checkout(priceId:string){
    const r = await fetch('/api/checkout', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ priceId })});
    const j = await r.json();
    if(j?.url) window.location.href = j.url;
  }
  return (
    <div className="container">
      <h1>Capsules</h1>
      <div className="grid">
        {data.map(c => (
          <div className="card" key={c.id}>
            <h3>{c.name}</h3>
            <small>{c.id}</small><br/><br/>
            {c.priceId ? <button className="btn" onClick={()=>checkout(c.priceId!)}>Buy {c.priceLabel || ''}</button> : <small>Coming soon</small>}
          </div>
        ))}
      </div>
    </div>
  )
}
EOF

cat > frontend/pages/bundle.tsx << 'EOF'
import { useState } from 'react';
export default function Bundle(){
  const [loading,setLoading]=useState(false);
  async function checkout(){
    setLoading(true);
    const r = await fetch('/api/checkout', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ priceId: process.env.NEXT_PUBLIC_PRICE_BUNDLE || 'price_bundle' })
    });
    const j = await r.json();
    if(j?.url) window.location.href = j.url;
    setLoading(false);
  }
  return (
    <div className="container">
      <h1>Super Bundle</h1>
      <p>All starter capsules at a discount.</p>
      <button className="btn" onClick={checkout} disabled={loading}>{loading?'Preparing...':'Buy Bundle'}</button>
    </div>
  )
}
EOF

cat > frontend/pages/affiliate.tsx << 'EOF'
export default function Affiliate(){
  return (
    <div className="container">
      <h1>Affiliate Program</h1>
      <p>Get your link, share the vibes, earn commission.</p>
      <a className="btn" href="https://coastalmonkvibes.lemonsqueezy.com/affiliates" target="_blank" rel="noreferrer">Join Affiliates</a>
    </div>
  )
}
EOF

cat > frontend/pages/dna.tsx << 'EOF'
import { useState } from 'react';
export default function DNA(){
  const [email,setEmail]=useState('');
  const [res,setRes]=useState('');
  async function submit(e:any){
    e.preventDefault();
    const score = Math.random();
    const vibe = score>0.66?'Post-Digital Oracle': score>0.33?'Techno Forest':'Coastal Monk';
    setRes(`Your vibe match: ${vibe}. Check your email for a discount (demo).`);
  }
  return (
    <div className="container">
      <h1>VIBE DNA</h1>
      <form onSubmit={submit}>
        <input value={email} onChange={e=>setEmail(e.target.value)} style={{padding:8, borderRadius:8, border:'1px solid #333', width:'60%'}} />
        <button className="btn" style={{marginLeft:12}}>Get My Vibe</button>
      </form>
      <p>{res}</p>
    </div>
  )
}
EOF

cat > frontend/pages/drops.tsx << 'EOF'
export default function Drops(){
  return (
    <div className="container">
      <h1>Drops</h1>
      <ul>
        <li>Capsule 004: Viral Vortex — TBA</li>
        <li>Capsule 005: Analog Gods — TBA</li>
      </ul>
    </div>
  )
}
EOF

cat > frontend/pages/api/stub-capsules.ts << 'EOF'
import type { NextApiRequest, NextApiResponse } from 'next';
export default function handler(req:NextApiRequest,res:NextApiResponse){
  res.status(200).json([
    { id:'capsule-001', name:'VIBE INITIATION', priceId: process.env.PRICE_CAPSULE_001, priceLabel:'$9' },
    { id:'capsule-002', name:'INNER SUN', priceId: process.env.PRICE_CAPSULE_002, priceLabel:'$19' },
    { id:'capsule-003', name:'EMPIRE DROP', priceId: process.env.PRICE_CAPSULE_003, priceLabel:'$29' },
  ]);
}
EOF

cat > frontend/pages/api/checkout.ts << 'EOF'
import type { NextApiRequest, NextApiResponse } from 'next';
import Stripe from 'stripe';
const secret = process.env.STRIPE_SECRET_KEY || '';
const stripe = secret ? new Stripe(secret, { apiVersion: '2024-06-20' }) : null;
export default async function handler(req:NextApiRequest,res:NextApiResponse){
  if(req.method!=='POST') return res.status(405).json({error:'Method not allowed'});
  const { priceId } = req.body || {};
  if(!priceId) return res.status(400).json({error:'Missing priceId'});
  if(!stripe) return res.status(500).json({error:'Stripe not configured'});
  const session = await stripe.checkout.sessions.create({
    mode:'payment',
    line_items:[{ price: String(priceId), quantity: 1 }],
    success_url: `${req.headers.origin}/?success=1`,
    cancel_url: `${req.headers.origin}/?canceled=1`
  });
  return res.status(200).json({ url: session.url });
}
EOF

# --- API (FastAPI)
mkdir -p api
cat > api/requirements.txt << 'EOF'
fastapi==0.112.0
uvicorn==0.30.3
pydantic==2.8.2
EOF

cat > api/main.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
app = FastAPI(title="Vibe API", version="1.0.0")
class Capsule(BaseModel):
    id: str
    name: str
    priceId: Optional[str] = None
    priceLabel: Optional[str] = None
@app.get("/capsules/latest", response_model=List[Capsule])
def latest_capsules():
    return [
        {"id":"capsule-001","name":"VIBE INITIATION","priceId":"price_123","priceLabel":"$9"},
        {"id":"capsule-002","name":"INNER SUN","priceId":"price_456","priceLabel":"$19"},
        {"id":"capsule-003","name":"EMPIRE DROP","priceId":"price_789","priceLabel":"$29"},
    ]
class PredictIn(BaseModel):
    prompt: str
@app.post("/capsule/predict")
def predict_capsule(inp: PredictIn):
    name = f"Predicted Capsule • {inp.prompt[:24]}"
    return {"name": name, "suggestedPrice":"$14.99"}
@app.post("/affiliates/create")
def affiliates_create():
    return {"status":"ok","url":"https://coastalmonkvibes.lemonsqueezy.com/affiliates"}
@app.get("/vibeDNA/{id}")
def vibe_dna(id: str):
    return {"id": id, "match":"Post-Digital Oracle", "score":0.82}
@app.get("/health")
def health():
    return {"ok":True,"time":datetime.utcnow().isoformat()+"Z"}
EOF

cat > api/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]
EOF

# --- Infra
mkdir -p infra
cat > infra/render.yaml << 'EOF'
services:
  - type: web
    name: vibe-api
    env: docker
    plan: free
    autoDeploy: true
    dockerfilePath: api/Dockerfile
EOF

cat > infra/railway.json << 'EOF'
{
  "services":[{"name":"vibe-api","path":"api"}]
}
EOF

# --- GitHub Actions
mkdir -p .github/workflows
cat > .github/workflows/hourly_status.yml << 'EOF'
name: hourly-status
on:
  schedule:
    - cron: '0 * * * *'
  workflow_dispatch:
jobs:
  send-status:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Send status ping
        env:
          STATUS_WEBHOOK_URL: ${{ secrets.STATUS_WEBHOOK_URL }}
        run: |
          if [ -z "$STATUS_WEBHOOK_URL" ]; then
            echo "No STATUS_WEBHOOK_URL secret set. Skipping."
            exit 0
          fi
          NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
          payload=$(jq -n --arg t "$NOW" '{text: ("VIBEVERSE hourly status ping: " + $t)}')
          curl -s -X POST -H "Content-Type: application/json" -d "$payload" "$STATUS_WEBHOOK_URL"
EOF

# --- Scripts
mkdir -p scripts
cat > scripts/local_dev.md << 'EOF'
# Local Dev
Frontend:
  cd frontend && npm i && npm run dev  -> http://localhost:3000
Backend:
  cd api && pip install -r requirements.txt && uvicorn main:app --reload  -> http://localhost:8080
Set NEXT_PUBLIC_API_BASE=http://localhost:8080 in frontend env for local testing.
EOF

# --- Git init & optional GH create
git init -q
git add .
git commit -q -m "VIBEVERSE machine: initial scaffold"

if command -v gh >/dev/null 2>&1 && [[ -n "$GITHUB_OWNER" ]]; then
  echo "==> Creating GitHub repo via gh..."
  gh repo create "${GITHUB_OWNER}/${REPO_NAME}" --private --source=. --push || echo "gh repo create failed or repo exists. Skipping."
else
  echo "==> Skipping GitHub create (missing gh or owner)."
fi

# --- Set GH secret for hourly webhook (if gh + provided)
if command -v gh >/dev/null 2>&1 && [[ -n "$STATUS_WEBHOOK_URL" ]]; then
  echo "==> Setting GitHub Actions secret STATUS_WEBHOOK_URL"
  gh secret set STATUS_WEBHOOK_URL -b "$STATUS_WEBHOOK_URL" || echo "Failed to set GH secret."
fi

# --- Vercel env & deploy (if CLI available)
if command -v vercel >/dev/null 2>&1; then
  echo "==> Configuring Vercel (env + deploy)"
  pushd frontend >/dev/null
    if [[ -n "$VERCEL_PROJECT" ]]; then
      vercel link --project "$VERCEL_PROJECT" --yes || true
    else
      vercel link --yes || true
    fi
    # Set envs for all environments (production, preview, development)
    if [[ -n "$STRIPE_PK" ]]; then
      vercel env add NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY production <<< "$STRIPE_PK" || true
      vercel env add NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY preview <<< "$STRIPE_PK" || true
      vercel env add NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY development <<< "$STRIPE_PK" || true
    fi
    if [[ -n "$STRIPE_SK" ]]; then
      vercel env add STRIPE_SECRET_KEY production <<< "$STRIPE_SK" || true
      vercel env add STRIPE_SECRET_KEY preview <<< "$STRIPE_SK" || true
      vercel env add STRIPE_SECRET_KEY development <<< "$STRIPE_SK" || true
    fi
    if [[ -n "$PRICE_001" ]]; then
      vercel env add PRICE_CAPSULE_001 production <<< "$PRICE_001" || true
      vercel env add PRICE_CAPSULE_001 preview <<< "$PRICE_001" || true
      vercel env add PRICE_CAPSULE_001 development <<< "$PRICE_001" || true
    fi
    if [[ -n "$PRICE_002" ]]; then
      vercel env add PRICE_CAPSULE_002 production <<< "$PRICE_002" || true
      vercel env add PRICE_CAPSULE_002 preview <<< "$PRICE_002" || true
      vercel env add PRICE_CAPSULE_002 development <<< "$PRICE_002" || true
    fi
    if [[ -n "$PRICE_003" ]]; then
      vercel env add PRICE_CAPSULE_003 production <<< "$PRICE_003" || true
      vercel env add PRICE_CAPSULE_003 preview <<< "$PRICE_003" || true
      vercel env add PRICE_CAPSULE_003 development <<< "$PRICE_003" || true
    fi
    if [[ -n "$PRICE_BUNDLE" ]]; then
      vercel env add NEXT_PUBLIC_PRICE_BUNDLE production <<< "$PRICE_BUNDLE" || true
      vercel env add NEXT_PUBLIC_PRICE_BUNDLE preview <<< "$PRICE_BUNDLE" || true
      vercel env add NEXT_PUBLIC_PRICE_BUNDLE development <<< "$PRICE_BUNDLE" || true
    fi
    # Deploy
    vercel --prod --confirm || echo "Vercel deploy skipped/failed; ensure login & project selection."
  popd >/dev/null
else
  echo "==> Skipping Vercel deploy (vercel CLI not found)."
fi

# --- API deploy via Railway (optional)
if [[ "$API_DEPLOY" == "railway" ]]; then
  if command -v railway >/dev/null 2>&1; then
    echo "==> Deploying API via Railway"
    pushd api >/dev/null
      railway up || echo "Railway deploy skipped/failed. Ensure railway CLI login."
    popd >/dev/null
  else
    echo "==> Railway CLI not found; skipping API deploy."
  fi
else
  echo "==> API cloud deploy skipped (API_DEPLOY=$API_DEPLOY)."
fi

echo "============================"
echo "BUILD COMPLETE."
echo "Next steps:"
echo "  - In Vercel dashboard, confirm env vars and production URL."
echo "  - For backend: deploy api/ to Railway/Render or run locally."
echo "  - Test checkout with Stripe test card: 4242 4242 4242 4242"
echo "============================"
