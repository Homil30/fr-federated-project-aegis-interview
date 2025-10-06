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

// ✅ Toast Notification Utility
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.textContent = message;
  toast.style.position = "fixed";
  toast.style.bottom = "20px";
  toast.style.right = "20px";
  toast.style.padding = "12px 18px";
  toast.style.borderRadius = "8px";
  toast.style.fontFamily = "Inter, sans-serif";
  toast.style.fontSize = "0.9em";
  toast.style.zIndex = "9999";
  toast.style.color = "#fff";
  toast.style.boxShadow = "0 2px 10px rgba(0,0,0,0.3)";
  toast.style.transition = "opacity 0.4s ease";
  toast.style.opacity = "0";
  toast.style.background =
    type === "success"
      ? "linear-gradient(90deg,#00c851,#007e33)"
      : type === "error"
      ? "linear-gradient(90deg,#ff4444,#cc0000)"
      : type === "warning"
      ? "linear-gradient(90deg,#ffbb33,#ff8800)"
      : "linear-gradient(90deg,#33b5e5,#0099cc)";
  document.body.appendChild(toast);
  setTimeout(() => (toast.style.opacity = "1"), 100);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 500);
  }, 3000);
}

// ✅ Step 1 — Check if candidate already registered
window.addEventListener("DOMContentLoaded", async () => {
  const storedId = localStorage.getItem("candidate_id");
  if (storedId && storedId.trim() !== "") {
    try {
      const resp = await fetch(`http://127.0.0.1:8000/interview/check?candidate_id=${storedId}`);
      const data = await resp.json();

      if (data.exists) {
        statusDiv.innerHTML = `👋 Welcome back, <b>${storedId}</b>! Please verify your face to continue.`;
        const verifyBtn = document.createElement("button");
        verifyBtn.textContent = "Verify My Face";
        verifyBtn.classList.add("button", "button-primary");

        // ✅ Add verified logic
        verifyBtn.onclick = async () => {
          if (!storedId || storedId.trim() === "") {
            showToast("Please enter a valid Candidate ID before verification!", "warning");
            return;
          }

          try {
            const checkResp = await fetch(`http://127.0.0.1:8000/interview/check?candidate_id=${storedId}`);
            const checkData = await checkResp.json();

            if (checkData.exists) {
              localStorage.setItem("candidate_id", storedId);
              showToast("✅ Face verified! Redirecting to Interview...", "success");
              setTimeout(() => {
                window.location.href = `interview.html?candidate_id=${encodeURIComponent(storedId)}&verify=true`;
              }, 1500);
            } else {
              showToast("❌ Candidate ID not registered. Please register your face first!", "error");
            }
          } catch (err) {
            showToast("⚠️ Unable to verify your ID. Please try again later.", "warning");
          }
        };

        statusDiv.appendChild(verifyBtn);
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
    statusDiv.innerText = "📸 Camera started. Click 'Capture' when ready!";
  } catch (err) {
    showToast("Camera error: " + err.message, "error");
  }
};

// ✅ Capture Face
captureBtn.onclick = () => {
  if (!stream) return showToast("Please start the camera first.", "warning");

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
    statusDiv.innerHTML = "✅ Face captured successfully!";
    showToast("Face captured successfully!", "success");
  }, 'image/jpeg', 0.9);
};

// ✅ Submit Face + ID → Register → Redirect
submitBtn.onclick = async () => {
  const candidateId = candidateInput.value.trim();
  if (!candidateId || !capturedBlob) {
    showToast("Please enter your Candidate ID and capture your face before submitting!", "warning");
    return;
  }

  const fd = new FormData();
  fd.append('candidate_id', candidateId);
  fd.append('file', capturedBlob, 'face.jpg');

  statusDiv.innerText = "⏳ Submitting registration...";

  try {
    const resp = await fetch('http://127.0.0.1:8000/interview/register', {
      method: 'POST',
      body: fd
    });

    const data = await resp.json();
    console.log("Server response:", data);

    if (data.error === "no_face_detected") {
      statusDiv.innerHTML = "❌ No face detected. Please capture your face clearly.";
      showToast("No face detected. Please try again.", "error");
      return;
    }

    if (data.status === "ok" || data.candidate_id) {
      statusDiv.innerHTML = "✅ Registration successful! Redirecting...";
      localStorage.setItem("candidate_id", candidateId);
      showToast("✅ Registration successful! Redirecting...", "success");
      setTimeout(() => {
        window.location.href = `interview.html?candidate_id=${encodeURIComponent(candidateId)}&autoStart=true`;
      }, 2000);
    } else {
      statusDiv.innerHTML = "❌ Registration failed. Please try again.";
      showToast("Registration failed. Please try again.", "error");
    }
  } catch (err) {
    statusDiv.innerText = "❌ Network error: " + err.message;
    showToast("Network error: " + err.message, "error");
  }
};
