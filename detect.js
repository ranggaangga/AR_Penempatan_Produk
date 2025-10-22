import * as tf from '@tensorflow/tfjs';
import cocoSsd from '@tensorflow-models/coco-ssd';

const video = document.getElementById('cam');
const wrap  = document.getElementById('wrap');

async function setupCam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
  video.srcObject = stream;
  await video.play();
  return new Promise(resolve => video.onloadedmetadata = resolve);
}

function drawBoxes(preds) {
  // Bersihkan box lama
  [...wrap.querySelectorAll('.box')].forEach(n => n.remove());
  const { videoWidth: vw, videoHeight: vh } = video;
  const rw = wrap.clientWidth, rh = wrap.clientHeight;
  // hitung skala (object-fit: cover)
  const scale = Math.max(rw / vw, rh / vh);
  const offX = (rw - vw * scale) / 2;
  const offY = (rh - vh * scale) / 2;

  preds.forEach(p => {
    const [x, y, w, h] = p.bbox; // dalam pixel video
    const left = offX + x * scale;
    const top  = offY + y * scale;
    const bw   = w * scale;
    const bh   = h * scale;

    const box = document.createElement('div');
    box.className = 'box';
    box.style.left = `${left}px`;
    box.style.top  = `${top}px`;
    box.style.width  = `${bw}px`;
    box.style.height = `${bh}px`;

    const lbl = document.createElement('div');
    lbl.className = 'lbl';
    lbl.textContent = `${p.class} ${(p.score*100).toFixed(0)}%`;
    box.appendChild(lbl);
    wrap.appendChild(box);
  });
}

async function main() {
  await setupCam();
  const model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });

  async function tick() {
    const preds = await model.detect(video, 6); // max 6 objek
    drawBoxes(preds);
    requestAnimationFrame(tick);
  }
  tick();
}

main().catch(console.error);