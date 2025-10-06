// login.js

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value.trim();

    if (!email || !password) {
      alert("Please enter both email and password.");
      return;
    }

    // ✅ Temporary authentication (you can connect backend later)
    if (email === "admin@aegis.com" && password === "1234") {
      alert("✅ Login successful!");
      localStorage.setItem("user_email", email);

      // Redirect to homepage
      window.location.href = "index.html";
    } else {
      alert("❌ Invalid credentials. Try admin@aegis.com / 1234 for testing.");
    }
  });
});
