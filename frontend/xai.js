// ✅ XAI (Grad-CAM) Explainability Function
async function uploadForExplain() {
  const fileInput = document.getElementById("xaiFile");
  if (!fileInput || !fileInput.files.length) {
    alert("Please select an image first!");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  const resultBox = document.getElementById("xaiResult");
  resultBox.innerHTML = "⏳ Generating Grad-CAM heatmap...";

  try {
    const response = await fetch("http://127.0.0.1:8000/explain/", {
      method: "POST",
      body: formData
    });

    if (!response.ok) throw new Error("Server error: " + response.status);

    const result = await response.json();
    console.log("XAI response:", result);

    if (result.error) {
      resultBox.innerHTML = `❌ Error: ${result.error}`;
      return;
    }

    // 🧠 Normalize image source (works for both prefixed and raw base64)
    let imgSrc = result.heatmap.startsWith("data:image")
      ? result.heatmap
      : `data:image/jpeg;base64,${result.heatmap}`;

    resultBox.innerHTML = `
      <p><strong>Prediction:</strong> Class ${result.prediction}</p>
      <img src="${imgSrc}" 
           alt="Grad-CAM Heatmap"
           style="max-width:500px; border-radius:12px; margin-top:10px; box-shadow:0 0 20px #00ffff;">
    `;
  } catch (error) {
    console.error("XAI Error:", error);
    resultBox.innerHTML = `❌ Request failed: ${error.message}`;
  }
}
