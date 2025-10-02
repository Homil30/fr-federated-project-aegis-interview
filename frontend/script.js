// 🔮 Image Upload & Prediction
async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    if (!fileInput || !fileInput.files.length) {
        alert("Please select an image first!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server error: " + response.status);
        }

        const result = await response.json();
        console.log("Prediction result:", result);

        // Labels for prediction (adjust for your dataset)
        const labels = ["Class 0: Unknown", "Class 1: Known"];
        const predictionText = labels[result.prediction] || `Class ${result.prediction}`;

        // --- Format nicely ---
        let html = `<div class="prediction-box">
            <strong>🔮 Prediction:</strong> ${predictionText}<br>
        `;

        if (result.confidence) {
            html += `<strong>Confidence:</strong> ${result.confidence}<br>`;
        }

        if (result.probabilities) {
            html += `<strong>Probabilities:</strong><br><ul>`;
            for (const [cls, prob] of Object.entries(result.probabilities)) {
                html += `<li>${cls}: ${prob}</li>`;
            }
            html += `</ul>`;
        }

        html += `</div>`;
        document.getElementById("result").innerHTML = html;

    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error: " + error.message;
    }
}

// 📊 Fetch evaluation results from backend and render charts
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch("http://127.0.0.1:8000/evaluation");
        if (!response.ok) {
            throw new Error("Server error: " + response.status);
        }

        const results = await response.json();
        console.log("Evaluation results:", results);

        if (results.error) {
            console.error("Error fetching evaluation results:", results.error);
            return;
        }

        // --- Fairness Chart ---
        const fairnessCanvas = document.getElementById("fairnessChart");
        if (fairnessCanvas && results.fairness) {
            const fairnessCtx = fairnessCanvas.getContext("2d");
            new Chart(fairnessCtx, {
                type: "bar",
                data: {
                    labels: Object.keys(results.fairness),
                    datasets: [{
                        label: "Accuracy (%)",
                        data: Object.values(results.fairness),
                        backgroundColor: "rgba(0, 200, 255, 0.7)",
                        borderColor: "#0ff",
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true, max: 100 } }
                }
            });
        }

        // --- Robustness Chart ---
        const robustnessCanvas = document.getElementById("robustnessChart");
        if (robustnessCanvas && results.robustness) {
            const robustnessCtx = robustnessCanvas.getContext("2d");
            new Chart(robustnessCtx, {
                type: "line",
                data: {
                    labels: Object.keys(results.robustness),
                    datasets: [{
                        label: "Accuracy (%)",
                        data: Object.values(results.robustness),
                        borderColor: "#0ff",
                        backgroundColor: "rgba(0, 200, 255, 0.2)",
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true, max: 100 } }
                }
            });
        }

        // --- Performance Chart ---
        const performanceCanvas = document.getElementById("performanceChart");
        if (performanceCanvas && results.performance) {
            const performanceCtx = performanceCanvas.getContext("2d");
            new Chart(performanceCtx, {
                type: "doughnut",
                data: {
                    labels: Object.keys(results.performance),
                    datasets: [{
                        data: Object.values(results.performance),
                        backgroundColor: ["#0ff", "#222"],
                        borderColor: ["#0ff", "#444"],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { position: "bottom" } }
                }
            });
        }

    } catch (error) {
        console.error("Failed to load evaluation results:", error);
    }
});
// ✅ XAI Upload & Explain
async function uploadForExplain() {
    const fileInput = document.getElementById("xaiFile");
    if (!fileInput || !fileInput.files.length) {
        alert("Please select an image first!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/explain/", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            document.getElementById("xaiResult").innerText = "Error: " + result.error;
            return;
        }

        document.getElementById("xaiResult").innerHTML = `
            <p><strong>Prediction:</strong> Class ${result.prediction}</p>
            <img src="data:image/png;base64,${result.heatmap}" 
                 alt="XAI Heatmap" style="max-width:400px; border-radius:10px; margin-top:10px;">
        `;
    } catch (error) {
        document.getElementById("xaiResult").innerText = "Error: " + error.message;
    }
}
