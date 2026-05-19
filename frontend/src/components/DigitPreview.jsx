import React from 'react';

export default function DigitPreview({ pixels, size = 84 }) {
  const canvasRef = React.useRef(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pixels || pixels.length !== 784) return;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.max(0, Math.min(255, Number(pixels[i] || 0)));
      image.data[i * 4 + 0] = 255 - v;
      image.data[i * 4 + 1] = 255 - v;
      image.data[i * 4 + 2] = 255 - v;
      image.data[i * 4 + 3] = 255;
    }
    const off = document.createElement('canvas');
    off.width = 28;
    off.height = 28;
    off.getContext('2d').putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(off, 0, 0, size, size);
  }, [pixels, size]);

  return <canvas ref={canvasRef} width={size} height={size} className="digit-preview" />;
}
