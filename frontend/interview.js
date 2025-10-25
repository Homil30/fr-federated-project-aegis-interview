// ==========================================
// Interview Analysis - Minimal fixes to prevent runtime errors
// ==========================================

document.addEventListener('DOMContentLoaded', async () => {
  console.log("🧠 Initializing Interview.js...");

  // ---------- DOM References ----------
  const video = document.getElementById('webcam');
  const candidateInput = document.getElementById('candidateId');
  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');
  const btnSubmit = document.getElementById('btnSubmit');
  const summaryBox = document.getElementById('summaryBox');
  const confCanvas = document.getElementById('confChart');
  const engCanvas = document.getElementById('engChart');

  // --- Minimal DOM validation to avoid "null.getContext" and similar errors ---
  const required = { video, btnStart, btnStop, btnSubmit, confCanvas, engCanvas };
  for (const [name, el] of Object.entries(required)) {
    if (!el) {
      console.error(`Required element "${name}" not found. Make sure element with id="${name.replace(/([A-Z])/g, m => m.toLowerCase())}" exists.`);
    }
  }
  if (!video || !btnStart || !btnStop || !btnSubmit || !confCanvas || !engCanvas) {
    alert('Some required UI elements are missing. Check console for details.');
    return; // stop initialization early to avoid runtime errors
  }

  // ---------- Configuration ----------
  const API_BASE = 'http://127.0.0.1:8000';
  const captureInterval = 2000; // Capture every 2 seconds

  // ---------- State Variables ----------
  let intervalId = null;
  let isProcessing = false;
  let stream = null;

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
      alert("Chart.js library not loaded. Please refresh the page.");
      return;
    }

    try {
      console.log("📊 Initializing charts...");

      // Identity Confidence Chart
      confChart = new Chart(confCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            label: 'Identity Confidence',
            data: [],
            borderColor: '#00bfff',
            backgroundColor: 'rgba(0, 191, 255, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            pointRadius: 3,
            pointHoverRadius: 5
          }]
        },
        options: chartOptions('#b8dfff', 'rgba(0, 191, 255, 0.1)')
      });

      // Engagement Score Chart
      engChart = new Chart(engCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            label: 'Engagement Score',
            data: [],
            borderColor: '#ff5a5f',
            backgroundColor: 'rgba(255, 90, 95, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            pointRadius: 3,
            pointHoverRadius: 5
          }]
        },
        options: chartOptions('#b8dfff', 'rgba(255, 90, 95, 0.1)')
      });

      console.log("✅ Charts initialized successfully");
    } catch (err) {
      console.error("❌ Error initializing charts:", err);
      // leave confChart/engChart null so caller can abort start
    }
  }

  // ---------- Chart Options helper ----------
  function chartOptions(tickColor, gridColor) {
    return {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 0 },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          ticks: { color: tickColor, stepSize: 0.2 },
          grid: { color: gridColor }
        },
        x: {
          ticks: { color: tickColor, maxRotation: 45, minRotation: 0 },
          grid: { color: gridColor }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#dfefff', font: { size: 12 } }
        }
      }
    };
  }

  // ---------- Add Data Point to Chart ----------
  function addPoint(chart, label, value) {
    if (!chart) {
      console.warn("⚠ Chart not initialized");
      return;
    }
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);
    // Keep only last 20 points
    if (chart.data.labels.length > 20) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }
    chart.update('none');
  }

  // ---------- Helper: Convert Data URL to Blob ----------
  function dataURLtoBlob(dataurl) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  }

  // ---------- Webcam Control ----------
  async function startWebcam() {
    try {
      console.log("🎥 Requesting webcam access...");
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });
      video.srcObject = stream;
      await video.play();
      console.log("✅ Webcam started successfully");
    } catch (err) {
      console.error("❌ Webcam error:", err);
      alert(`Camera access denied: ${err.message}\n\nPlease check browser permissions.`);
      throw err;
    }
  }

  function stopWebcam() {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
      console.log("⏹ Analysis interval stopped");
    }

    if (stream) {
      stream.getTracks().forEach(track => {
        track.stop();
        console.log(`🛑 Stopped track: ${track.kind}`);
      });
      stream = null;
    }

    if (video.srcObject) {
      video.srcObject = null;
    }

    if (btnStart) btnStart.disabled = false;
    if (btnStop) btnStop.disabled = true;
    console.log("🛑 Webcam stopped");
  }

  // ---------- Frame Capture + API Call ----------
  async function captureAndSend() {
    if (isProcessing) {
      console.log("⏳ Already processing frame, skipping...");
      return;
    }

    if (!video.videoWidth || !video.videoHeight) {
      console.warn("⚠ Video not ready yet");
      return;
    }

    isProcessing = true;
    const timestamp = new Date().toLocaleTimeString();

    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      const blob = dataURLtoBlob(dataUrl);

      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      const candidate_id = candidateInput.value.trim();
      if (candidate_id) {
        formData.append('candidate_id', candidate_id);
      }

      console.log(`📤 Sending frame at ${timestamp}...`);
      const response = await fetch(`${API_BASE}/interview/analyze`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("📥 API Response:", data);

      if (data.status === "no_face") {
        console.warn("⚠ No face detected in frame");
        addPoint(confChart, timestamp, 0);
        addPoint(engChart, timestamp, 0);
        return;
      }

      const identityConf = Number(data.identity_confidence) || 0;
      const engagementScore = Number(data.engagement_score) || 0;
      const eyeScore = Number(data.eye_score) || 0;
      const mouthScore = Number(data.mouth_score) || 0;
      const emotion = data.emotion || "neutral";

      console.log(`✅ Parsed: ID=${identityConf.toFixed(3)}, ENG=${engagementScore.toFixed(3)}, EYE=${eyeScore.toFixed(3)}, MOUTH=${mouthScore.toFixed(3)}, EMO=${emotion}`);

      identityScores.push(identityConf);
      engagementScores.push(engagementScore);
      eyeScores.push(eyeScore);
      mouthScores.push(mouthScore);

      addPoint(confChart, timestamp, identityConf);
      addPoint(engChart, timestamp, engagementScore);

      // Update live metrics display (if elements exist)
      const setTextIfExist = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
      setTextIfExist('dataTime', timestamp);
      setTextIfExist('dataIdentity', identityConf.toFixed(2));
      setTextIfExist('dataEngagement', engagementScore.toFixed(2));
      setTextIfExist('dataEye', eyeScore.toFixed(2));
      setTextIfExist('dataMouth', mouthScore.toFixed(2));

      const emotionEl = document.getElementById('emotionVal');
      if (emotionEl) {
        emotionEl.textContent = emotion;
        // Color code emotions
        if (emotion === "happy") emotionEl.style.color = "lime";
        else if (emotion === "neutral") emotionEl.style.color = "cyan";
        else if (emotion === "sad") emotionEl.style.color = "orange";
        else if (emotion === "angry") emotionEl.style.color = "red";
        else if (emotion === "surprise") emotionEl.style.color = "yellow";
        else if (emotion === "fear") emotionEl.style.color = "purple";
        else emotionEl.style.color = "white";
      }

      // Update insight message
      const insightEl = document.getElementById('insightMsg');
      if (insightEl) {
        let insight = `Emotion: ${emotion}`;
        if (identityConf > 0.8) insight += " | ✅ Verified";
        else if (identityConf > 0.4) insight += " | ⚠ Partial match";
        else insight += " | ❌ Not recognized";

        if (engagementScore > 0.7) insight += " | 👀 Excellent engagement!";
        else if (engagementScore > 0.4) insight += " | 🙂 Moderate engagement";
        else insight += " | 😐 Low engagement";

        insightEl.textContent = insight;
      }

      console.log(`📊 Charts updated. Total points: ${identityScores.length}`);

    } catch (err) {
      console.error("❌ Frame capture/analysis error:", err);
      console.error("Error details:", err.message || err);
    } finally {
      isProcessing = false;
    }
  }

  // ---------- Button Event Handlers ----------
  btnStart.onclick = async () => {
    console.log("▶ Start button clicked");
    btnStart.disabled = true;
    btnStop.disabled = false;

    try {
      if (!confChart || !engChart) {
        await initCharts();
      }

      // If charts didn't initialize (e.g., Chart.js missing), abort gracefully
      if (!confChart || !engChart) {
        alert('Charts not initialized. Check console for details.');
        btnStart.disabled = false;
        btnStop.disabled = true;
        return;
      }

      await startWebcam();

      identityScores = [];
      engagementScores = [];
      eyeScores = [];
      mouthScores = [];

      if (confChart) {
        confChart.data.labels = [];
        confChart.data.datasets[0].data = [];
        confChart.update();
      }
      if (engChart) {
        engChart.data.labels = [];
        engChart.data.datasets[0].data = [];
        engChart.update();
      }

      if (summaryBox) summaryBox.style.display = 'none';

      console.log("🎬 Starting analysis loop...");

      setTimeout(() => {
        captureAndSend();
        intervalId = setInterval(captureAndSend, captureInterval);
        console.log(`✅ Analysis loop started (interval: ${captureInterval}ms)`);
      }, 1000);

    } catch (err) {
      console.error("❌ Start failed:", err);
      if (btnStart) btnStart.disabled = false;
      if (btnStop) btnStop.disabled = true;
      alert(`Failed to start: ${err.message || err}`);
    }
  };

  btnStop.onclick = () => {
    console.log("⏹ Stop button clicked");
    stopWebcam();
  };

  btnSubmit.onclick = () => {
    console.log("✓ Finish Interview button clicked");

    stopWebcam();

    if (identityScores.length === 0) {
      alert("No data collected. Please start the interview first.");
      console.warn("⚠ No data to summarize");
      return;
    }

    const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    const avgIdentity = avg(identityScores);
    const avgEngagement = avg(engagementScores);
    const avgEye = avg(eyeScores);
    const avgMouth = avg(mouthScores);

    console.log(`📊 Summary calculated from ${identityScores.length} samples`);
    console.log(`   Avg Identity: ${avgIdentity.toFixed(3)}`);
    console.log(`   Avg Engagement: ${avgEngagement.toFixed(3)}`);
    console.log(`   Avg Eye: ${avgEye.toFixed(3)}`);
    console.log(`   Avg Mouth: ${avgMouth.toFixed(3)}`);

    let verdict = "";
    if (avgIdentity > 0.8 && avgEngagement > 0.7) {
      verdict = "🌟 Excellent performance! High recognition and engagement.";
    } else if (avgIdentity > 0.6 && avgEngagement > 0.5) {
      verdict = "👍 Good effort! Decent recognition and engagement.";
    } else if (avgIdentity > 0.4 || avgEngagement > 0.3) {
      verdict = "⚠ Needs improvement. Low recognition or engagement.";
    } else {
      verdict = "❌ Poor performance. Very low metrics detected.";
    }

    // Safe DOM writes (only if elements exist)
    const safeSet = (id, txt) => { const el = document.getElementById(id); if (el) el.innerText = txt; };
    safeSet('avgIdentity', avgIdentity.toFixed(3));
    safeSet('avgEngagement', avgEngagement.toFixed(3));
    safeSet('avgEye', avgEye.toFixed(3));
    safeSet('avgMouth', avgMouth.toFixed(3));
    safeSet('verdict', verdict);

    if (summaryBox) {
      summaryBox.style.display = 'block';
      summaryBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    console.log("✅ Summary displayed");
  };

  // ---------- Initialize on Page Load ----------
  if (btnStop) btnStop.disabled = true;
  if (btnSubmit) btnSubmit.disabled = false;

  await initCharts();

  console.log("✅ Interview.js fully initialized and ready");
});
