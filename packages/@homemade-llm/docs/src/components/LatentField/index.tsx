import { type ReactNode, useEffect, useRef } from 'react';

import styles from './styles.module.css';

type Node = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  live: number; // 0..1 "活性度"。たまにライムに光る
  phase: number;
};

/**
 * 潜在空間（latent space）を漂う埋め込みベクトルの星座。
 * 近いノード同士を attention のように線で結ぶ。マウスで視差。
 * SSR されないよう描画は useEffect 内（canvas 要素のみ初期出力）。
 */
export default function LatentField(): ReactNode {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const reduce = window.matchMedia(
      '(prefers-reduced-motion: reduce)',
    ).matches;
    let width = 0;
    let height = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let nodes: Node[] = [];
    const mouse = { x: 0, y: 0, tx: 0, ty: 0 };

    const rand = (a: number, b: number) => a + Math.random() * (b - a);

    const build = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      width = rect?.width ?? window.innerWidth;
      height = rect?.height ?? 520;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.min(120, Math.round((width * height) / 11000));
      nodes = Array.from({ length: count }, () => ({
        x: rand(0, width),
        y: rand(0, height),
        vx: rand(-0.18, 0.18),
        vy: rand(-0.18, 0.18),
        live: Math.random() < 0.12 ? rand(0.4, 1) : 0,
        phase: rand(0, Math.PI * 2),
      }));
    };

    const LINK = 132;
    let raf = 0;
    let t = 0;

    const draw = () => {
      t += 0.016;
      mouse.x += (mouse.tx - mouse.x) * 0.05;
      mouse.y += (mouse.ty - mouse.y) * 0.05;
      const ox = (mouse.x - width / 2) * 0.02;
      const oy = (mouse.y - height / 2) * 0.02;

      ctx.clearRect(0, 0, width, height);

      for (const n of nodes) {
        if (!reduce) {
          n.x += n.vx;
          n.y += n.vy;
          if (n.x < -20) n.x = width + 20;
          if (n.x > width + 20) n.x = -20;
          if (n.y < -20) n.y = height + 20;
          if (n.y > height + 20) n.y = -20;
        }
      }

      // links
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        if (!a) continue;
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          if (!b) continue;
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < LINK * LINK) {
            const alpha = (1 - Math.sqrt(d2) / LINK) * 0.5;
            const hot = a.live > 0.3 || b.live > 0.3;
            ctx.strokeStyle = hot
              ? `rgba(182, 244, 0, ${alpha * 0.5})`
              : `rgba(130, 150, 220, ${alpha * 0.32})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x + ox, a.y + oy);
            ctx.lineTo(b.x + ox, b.y + oy);
            ctx.stroke();
          }
        }
      }

      // nodes
      for (const n of nodes) {
        const pulse = reduce ? 0.6 : 0.5 + 0.5 * Math.sin(t * 1.6 + n.phase);
        const px = n.x + ox;
        const py = n.y + oy;
        if (n.live > 0.3) {
          const r = 1.6 + n.live * 1.8 * pulse;
          ctx.fillStyle = `rgba(182, 244, 0, ${0.5 + 0.4 * pulse})`;
          ctx.shadowColor = 'rgba(182, 244, 0, 0.8)';
          ctx.shadowBlur = 10;
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else {
          ctx.fillStyle = `rgba(150, 170, 235, ${0.32 + 0.25 * pulse})`;
          ctx.beginPath();
          ctx.arc(px, py, 1.4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (!reduce) raf = requestAnimationFrame(draw);
    };

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.tx = e.clientX - rect.left;
      mouse.ty = e.clientY - rect.top;
    };

    build();
    mouse.x = mouse.tx = width / 2;
    mouse.y = mouse.ty = height / 2;
    draw();

    const onResize = () => {
      build();
      if (reduce) draw();
    };
    window.addEventListener('resize', onResize);
    window.addEventListener('mousemove', onMove);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('mousemove', onMove);
    };
  }, []);

  return <canvas ref={canvasRef} className={styles.canvas} />;
}
