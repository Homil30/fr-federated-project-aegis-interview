// ✅ Robust Interview.js — Final Fixed Version (Fully Functional Live Dashboard)
document.addEventListener('DOMContentLoaded', async () => {
  console.log("🧠 Initializing Interview.js...");

  // ---------- DOM References ----------
  const video = document.getElementById('webcam');
  const candidateInput = document.getElementById('candidateId');
  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');
  const btnSubmit = document.getElementById('btnSubmit');
  const summaryBox = document.getElementById('summaryBox');
  const insightMsgEl = document.getElementById('insightMsg');
  const confCanvas = document.getElementById('confChart');
  const engCanvas = document.getElementById('engChart');

  // ✅ New live elements
  const idVal = document.getElementById("identityVal");
  const engVal = document.getElementById("engagementVal");
  const eyeVal = document.getElementById("eyeVal");
  const mouthVal = document.getElementById("mouthVal");
  const emoVal = document.getElementById("emotionVal");
  const logBox = document.getElementById("logBox");

  // ---------- State Variables ----------
  let intervalId = null;
  const captureInterval = 2000; // every 2 seconds
  let isProcessing = false;

  // ---------- Data Storage ----------
  let identityScores = [];
  let engagementScores = [];
  let eyeScores = [];
  let mouthScores = [];

  // ---------- Chart Instances ----------
  let confChart = null;
  let engChart = null;

  // ---------- Chart Initialization ----------
  async function initCharts() {
    if (!window.Chart) {
      console.error("❌ Chart.js not found. Please check script import in HTML.");
      return;
    }

    console.log("📊 Initializing charts...");

    confChart = new Chart(confCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Identity Confidence',
          data: [],
          borderColor: 'rgb(0, 191, 255)',
          borderWidth: 2,
          tension: 0.25,
          fill: false
        }]
      },
      options: {
        responsive: true,
        animation: false,
        scales: { y: { beginAtZero: true, max: 1 } }
      }
    });

    engChart = new Chart(engCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Engagement Score',
          data: [],
          borderColor: 'rgb(255, 99, 132)',
          borderWidth: 2,
          tension: 0.25,
          fill: false
        }]
      },
      options: {
        responsive: true,
        animation: false,
        scales: { y: { beginAtZero: true, max: 1 } }
      }
    });

    console.log("✅ Charts initialized successfully");
  }

  // Add new point to a chart (maintain 20 samples)
  function addPoint(chart, label, value) {
    if (!chart) return;
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);
    if (chart.data.labels.length > 20) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }
    chart.update();
  }

  // ---------- Helper: Convert Data URL to Blob ----------
  function dataURLtoBlob(dataurl) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new Blob([u8arr], { type: mime });
  }

  // ---------- Webcam Control ----------
  async function startWebcam() {
    try {
      console.log("🎥 Requesting webcam access...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      video.srcObject = stream;

      await new Promise((resolve, reject) => {
        video.onloadedmetadata = () => {
          video.play().then(resolve).catch(reject);
        };
        setTimeout(() => reject(new Error("Webcam timeout")), 5000);
      });

      console.log("✅ Webcam started successfully");
    } catch (err) {
      console.error("❌ Webcam error:", err);
      insightMsgEl.innerText = "❌ Camera access denied: " + err.message;
      throw err;
    }
  }

  function stopWebcam() {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
    }
    btnStart.disabled = false;
    btnStop.disabled = true;
    console.log("🛑 Webcam stopped");
  }

  // ---------- Frame Capture + API Call ----------
  async function captureAndSend() {
    if (isProcessing) return;
    if (!video.videoWidth || !video.videoHeight) return;

    isProcessing = true;
    const t = new Date().toLocaleTimeString();

    try {
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

      const resp = await fetch('http://127.0.0.1:8000/interview/analyze', {
        method: 'POST',
        body: fd
      });

      const data = await resp.json();
      console.log("📥 Response:", data);

      if (data.status === "no_face") {
        insightMsgEl.innerText = "❌ No face detected";
        addPoint(confChart, t, 0);
        addPoint(engChart, t, 0);
        return;
      }

      // ✅ Extract values
      const id = Number(data.identity_confidence || 0);
      const eng = Number(data.engagement_score || 0);
      const eye = Number(data.eye_score || 0);
      const mouth = Number(data.mouth_score || 0);
      const emo = data.emotion || "neutral";

      console.log(`✅ Parsed: ID=${id}, ENG=${eng}, EYE=${eye}, MOUTH=${mouth}, EMO=${emo}`);

      addPoint(confChart, t, id);
      addPoint(engChart, t, eng);

      // ✅ Update live UI
      if (idVal) idVal.innerText = id.toFixed(2);
      if (engVal) engVal.innerText = eng.toFixed(2);
      if (eyeVal) eyeVal.innerText = eye.toFixed(2);
      if (mouthVal) mouthVal.innerText = mouth.toFixed(2);
      if (emoVal) {
        emoVal.innerText = emo;
        if (emo === "happy") emoVal.style.color = "lime";
        else if (emo === "neutral") emoVal.style.color = "cyan";
        else if (emo === "angry") emoVal.style.color = "red";
        else emoVal.style.color = "white";
      }

      if (logBox)
        logBox.innerHTML += `<div>⏰ ${t} → ENG=${eng.toFixed(2)}, EMO=${emo}</div>`;

      // Insight summary
      let insight = `Emotion: ${emo}\n`;
      if (id > 0.8) insight += "✅ Verified | ";
      else if (id > 0.4) insight += "⚠️ Partial match | ";
      else insight += "❌ Not recognized | ";

      if (eng > 0.7) insight += "👀 Excellent engagement!";
      else if (eng > 0.4) insight += "🙂 Moderate engagement.";
      else insight += "😐 Low engagement.";

      insightMsgEl.innerText = insight;

    } catch (err) {
      console.error("❌ Capture error:", err);
      insightMsgEl.innerText = "⚠️ Error: " + err.message;
    } finally {
      isProcessing = false;
    }
  }

  // ---------- Button Handlers ----------
  btnStart.onclick = async () => {
    btnStart.disabled = true;
    btnStop.disabled = false;
    try {
      if (!confChart || !engChart) await initCharts();
      await startWebcam();
      insightMsgEl.innerText = "🎥 Analyzing...";
      if (intervalId) clearInterval(intervalId);
      setTimeout(() => {
        captureAndSend();
        intervalId = setInterval(captureAndSend, captureInterval);
      }, 1000);
    } catch (err) {
      console.error("❌ Start failed:", err);
      btnStart.disabled = false;
      btnStop.disabled = true;
      insightMsgEl.innerText = "❌ Failed to start: " + err.message;
    }
  };

  btnStop.onclick = () => {
    stopWebcam();
    insightMsgEl.innerText = "🛑 Paused";
  };

  btnSubmit.onclick = () => {
    stopWebcam();
    if (identityScores.length === 0) {
      alert("No data collected. Please start the interview first.");
      return;
    }

    const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    const avgIdentity = avg(identityScores).toFixed(3);
    const avgEngagement = avg(engagementScores).toFixed(3);
    const avgEye = avg(eyeScores).toFixed(3);
    const avgMouth = avg(mouthScores).toFixed(3);

    let verdict = "";
    if (avgIdentity > 0.8 && avgEngagement > 0.7) verdict = "🌟 Excellent performance!";
    else if (avgIdentity > 0.6 && avgEngagement > 0.5) verdict = "🙂 Good effort!";
    else verdict = "⚠️ Needs improvement";

    document.getElementById('avgIdentity').innerText = avgIdentity;
    document.getElementById('avgEngagement').innerText = avgEngagement;
    document.getElementById('avgEye').innerText = avgEye;
    document.getElementById('avgMouth').innerText = avgMouth;
    document.getElementById('verdict').innerText = verdict;

    summaryBox.style.display = 'block';
    summaryBox.scrollIntoView({ behavior: 'smooth' });
  };

  // ---------- Initialize on Load ----------
  btnStop.disabled = true;
  await initCharts();
  console.log("📝 Frontend ready ✅");
});
