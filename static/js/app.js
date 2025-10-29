const queryInput = document.getElementById("qInput");
const queryButton = document.getElementById("queryButton");
const hint = document.getElementById("hint");

async function checkUploadReady() {
  try {
    const res = await fetch("/upload_status");
    const data = await res.json();
    if (data.ready) {
      queryInput.disabled = false;
      queryButton.disabled = false;
      hint.textContent = "✅ Indexed successfully. You can now ask a question.";
      hint.style.color = "green";
    }
  } catch (e) {
    // ignore
  }
}
setInterval(checkUploadReady, 1500);
