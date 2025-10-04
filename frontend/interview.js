// ✅ Auto-fill candidate ID + auto-start webcam if redirected
const urlParams = new URLSearchParams(window.location.search);
const candidateIdParam = urlParams.get('candidate_id') || localStorage.getItem("candidate_id");
const autoStart = urlParams.get('autoStart') === "true";

window.addEventListener("DOMContentLoaded", async () => {
  if (candidateIdParam) {
    document.getElementById('candidateId').value = candidateIdParam;
  }

  // ✅ Auto-start after redirect
  if (autoStart) {
    await startWebcam();
    if (!intervalId) intervalId = setInterval(captureAndSend, captureInterval);
  }
});

// =============== Interview Logic ===============
const video = document.getElementById('webcam');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const candidateInput = document.getElementById('candidateId');
const latest = document.getElementById('latest');

let stream = null;
let intervalId = null;
const captureInterval = 1000; // ms

// Charts
const confCtx = document.getElementById('confChart').getContext('2d');
const engCtx = document.getElementById('engChart').getContext('2d');

const confChart = new Chart(confCtx, {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'Identity Confidence', data: [], borderColor: '#00aaff', fill: false }] },
  options: { responsive: false, animation: false }
});

const engChart = new Chart(engCtx, {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'Engagement', data: [], borderColor: '#00ffcc', fill: false }] },
  options: { responsive: false, animation: false }
});

async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    alert('Webcam error: ' + e.message);
  }
}

function stopWebcam() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
}

function dataURLtoBlob(dataurl) {
  const arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]); let n = bstr.length; const u8arr = new Uint8Array(n);
  while (n--) u8arr[n] = bstr.charCodeAt(n);
  return new Blob([u8arr], { type: mime });
}

async function captureAndSend() {
  if (!video.videoWidth) return;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
  const blob = dataURLtoBlob(dataUrl);

  const fd = new FormData();
  fd.append('file', blob, 'frame.jpg');
  const candidate_id = candidateInput.value.trim();
  if (candidate_id) fd.append('candidate_id', candidate_id);

  try {
    const resp = await fetch('http://127.0.0.1:8000/interview/analyze', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) {
      latest.innerText = 'Error: ' + data.error;
      return;
    }
    const t = new Date().toLocaleTimeString();
    addPoint(confChart, t, data.identity_confidence ?? null);
    addPoint(engChart, t, data.engagement_score);
    latest.innerHTML = `
      <b>Time:</b> ${t} <br>
      <b>Identity:</b> ${data.identity_confidence ?? 'N/A'} <br>
      <b>Engagement:</b> ${data.engagement_score} <br>
      <small>eye: ${data.eye_score} | mouth: ${data.mouth_score}</small>
    `;
  } catch (err) {
    latest.innerText = 'Network error: ' + err.message;
  }
}

function addPoint(chart, label, value) {
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > 30) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update();
}

btnStart.onclick = async () => {
  await startWebcam();
  if (!intervalId) intervalId = setInterval(captureAndSend, captureInterval);
};

btnStop.onclick = () => {
  stopWebcam();
};
