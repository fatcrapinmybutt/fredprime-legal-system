#!/bin/bash
set -e
mkdir -p frontend/pages
cat <<'TSX' > frontend/pages/dna.tsx
import React from 'react';

export default function DNA() {
  return (
    <input type="email" placeholder="Email" />
  );
}
TSX
