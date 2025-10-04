// register.js
const video = document.createElement('video');
const canvas = document.createElement('canvas');
let stream = null;
let capturedBlob = null;

const startBtn = document.getElementById('startBtn');
const captureBtn = document.getElementById('captureBtn');
const submitBtn = document.getElementById('submitBtn');
const statusDiv = document.getElementById('status');
const candidateInput = document.getElementById('candidateId');
const preview = document.getElementById('preview');

// ✅ Step 1 — Check if candidate already registered
window.addEventListener("DOMContentLoaded", async () => {
  const storedId = localStorage.getItem("candidate_id");
  if (storedId) {
    try {
      const resp = await fetch(`http://127.0.0.1:8000/interview/check?candidate_id=${storedId}`);
      const data = await resp.json();
      if (data.exists) {
        // ✅ Candidate exists
        statusDiv.innerHTML = `👋 Welcome back, <b>${storedId}</b>! Redirecting you to Interview Analysis...`;
        setTimeout(() => {
          window.location.href = `interview.html?candidate_id=${encodeURIComponent(storedId)}&autoStart=true`;
        }, 2500);
        return;
      } else {
        statusDiv.innerHTML = `🆕 No record found for <b>${storedId}</b>. Please register below.`;
        candidateInput.value = storedId;
      }
    } catch (err) {
      console.warn("Check failed:", err.message);
    }
  }
});

// ✅ Start Camera
startBtn.onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();
    video.style.transform = "scaleX(-1)";
    preview.innerHTML = "";
    preview.appendChild(video);
    statusDiv.innerText = "Camera started...";
  } catch (err) {
    alert("Camera error: " + err.message);
  }
};

// ✅ Capture Face
captureBtn.onclick = () => {
  if (!stream) return alert("Please start the camera first.");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    capturedBlob = blob;
    preview.innerHTML = "";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(blob);
    img.width = 480;
    img.height = 360;
    img.style.borderRadius = "10px";
    preview.appendChild(img);
    statusDiv.innerHTML = "✅ Face captured!";
  }, 'image/jpeg', 0.9);
};

// ✅ Submit Face + ID → Register → Redirect
submitBtn.onclick = async () => {
  const candidateId = candidateInput.value.trim();
  if (!candidateId) return alert("Please enter Candidate ID.");
  if (!capturedBlob) return alert("Please capture your face first.");

  const fd = new FormData();
  fd.append('candidate_id', candidateId);
  fd.append('file', capturedBlob, 'face.jpg');

  statusDiv.innerText = "Submitting...";

  try {
    const resp = await fetch('http://127.0.0.1:8000/interview/register', {
      method: 'POST',
      body: fd
    });

    const data = await resp.json();
    console.log("Server response:", data);

    if (data.status === "ok" || data.candidate_id) {
      statusDiv.innerHTML = "✅ Registration successful! Redirecting...";
      localStorage.setItem("candidate_id", candidateId);

      // Redirect to Interview Page
      setTimeout(() => {
        window.location.href = `interview.html?candidate_id=${encodeURIComponent(candidateId)}&autoStart=true`;
      }, 2000);
    } else {
      statusDiv.innerHTML = "❌ Registration failed. Please try again.";
    }
  } catch (err) {
    statusDiv.innerText = "❌ Network error: " + err.message;
  }
};
