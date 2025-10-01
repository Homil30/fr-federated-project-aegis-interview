// 🔮 Image Upload & Prediction
async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
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
        const labels = ["Class 0: Unknown", "Class 1: Known"]; // Adjust as per your model
        const predictionText = labels[result.prediction] || `Class ${result.prediction}`;

        document.getElementById("result").innerHTML = `
            <div class="prediction-box">
                <strong>🔮 Prediction:</strong> ${predictionText}
            </div>
        `;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error: " + error.message;
    }
}

// 📊 Fetch evaluation results from backend and render charts
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch("http://127.0.0.1:8000/evaluation");
        const results = await response.json();

        if (results.error) {
            console.error("Error fetching evaluation results:", results.error);
            return;
        }

        // --- Fairness Chart ---
        const fairnessCtx = document.getElementById("fairnessChart").getContext("2d");
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

        // --- Robustness Chart ---
        const robustnessCtx = document.getElementById("robustnessChart").getContext("2d");
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

        // --- Performance Chart ---
        const performanceCtx = document.getElementById("performanceChart").getContext("2d");
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

    } catch (error) {
        console.error("Failed to load evaluation results:", error);
    }
});
