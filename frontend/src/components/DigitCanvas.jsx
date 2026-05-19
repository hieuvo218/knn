import React, { useEffect, useRef, useState } from 'react';

export default function DigitCanvas({ onPixelsChange }) {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const debounceRef = useRef(null);
  const hasDrawnRef = useRef(false);

  useEffect(() => {
    clearCanvas();
  }, []);

  function getPos(event) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const touch = event.touches?.[0];
    const x = (touch ? touch.clientX : event.clientX) - rect.left;
    const y = (touch ? touch.clientY : event.clientY) - rect.top;
    return { x, y };
  }

  function begin(event) {
    event.preventDefault();
    setDrawing(true);
    const ctx = canvasRef.current.getContext('2d');
    const { x, y } = getPos(event);
    ctx.beginPath();
    ctx.moveTo(x, y);
    hasDrawnRef.current = false;
  }

  function move(event) {
    if (!drawing) return;
    event.preventDefault();
    const ctx = canvasRef.current.getContext('2d');
    const { x, y } = getPos(event);
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';
    ctx.lineTo(x, y);
    ctx.stroke();
    hasDrawnRef.current = true;
    scheduleEmit();
  }

  function end(event) {
    if (event) event.preventDefault();
    if (!drawing) return;
    setDrawing(false);
    if (hasDrawnRef.current) {
      scheduleEmit();
      hasDrawnRef.current = false;
    }
  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    if (onPixelsChange) onPixelsChange(null);
  }

  function extractPixels() {
    const canvas = canvasRef.current;
    const off = document.createElement('canvas');
    off.width = 28;
    off.height = 28;
    const offCtx = off.getContext('2d');
    offCtx.fillStyle = '#ffffff';
    offCtx.fillRect(0, 0, 28, 28);
    offCtx.drawImage(canvas, 0, 0, 28, 28);
    const data = offCtx.getImageData(0, 0, 28, 28).data;
    const pixels = [];
    for (let i = 0; i < 784; i++) {
      const r = data[i * 4 + 0];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const gray = (r + g + b) / 3;
      pixels.push(Math.round(255 - gray));
    }
    return pixels;
  }

  function scheduleEmit() {
    if (!hasDrawnRef.current) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      onPixelsChange(extractPixels());
      debounceRef.current = null;
    }, 1000);
  }

  return (
    <div className="canvas-box">
      <canvas
        ref={canvasRef}
        width="280"
        height="280"
        className="draw-canvas"
        onMouseDown={begin}
        onMouseMove={move}
        onMouseUp={end}
        onMouseLeave={end}
        onTouchStart={begin}
        onTouchMove={move}
        onTouchEnd={end}
      />
      <button onClick={clearCanvas}>Clear</button>
    </div>
  );
}
