const API_BASE = "http://127.0.0.1:8000/api/v1";

let activeMode = "image"; // "image" | "video" | "audio" | "text"
let selectedFile = null;

const tabImage = document.getElementById("tab-image");
const tabVideo = document.getElementById("tab-video");
const tabAudio = document.getElementById("tab-audio");
const tabText = document.getElementById("tab-text");

const mediaPanel = document.getElementById("media-panel");
const textPanel = document.getElementById("text-panel");
const dropZone = document.getElementById("drop-zone");
const mediaInput = document.getElementById("media-file-input");
const dropPrompt = document.getElementById("drop-prompt");
const mediaIcon = document.getElementById("media-icon");
const fileName = document.getElementById("file-name");
const analyzeMediaBtn = document.getElementById("analyze-media-btn");

const claimText = document.getElementById("claim-text");
const analyzeTextBtn = document.getElementById("analyze-text-btn");

const resultBox = document.getElementById("result-box");
const verdictTitle = document.getElementById("verdict-title");
const verdictConfidence = document.getElementById("verdict-confidence");
const plainExplanation = document.getElementById("plain-explanation");

function setTab(mode, btn, acceptType, icon, text) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  activeMode = mode;
  selectedFile = null;
  fileName.innerText = "";
  analyzeMediaBtn.disabled = true;
  resultBox.classList.add("hidden");

  if (mode === "text") {
    mediaPanel.classList.remove("active");
    textPanel.classList.add("active");
  } else {
    mediaPanel.classList.add("active");
    textPanel.classList.remove("active");
    mediaInput.accept = acceptType;
    mediaIcon.innerText = icon;
    dropPrompt.innerText = text;
  }
}

tabImage.addEventListener("click", () => setTab("image", tabImage, "image/*", "🖼️", "Upload or drop an image"));
tabVideo.addEventListener("click", () => setTab("video", tabVideo, "video/mp4,video/mkv,video/webm", "🎥", "Upload or drop an MP4 video"));
tabAudio.addEventListener("click", () => setTab("audio", tabAudio, "audio/*,.wav,.mp3,.ogg", "🎙️", "Upload a voice note or audio file"));
tabText.addEventListener("click", () => setTab("text", tabText, "", "", ""));

dropZone.addEventListener("click", () => mediaInput.click());
mediaInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    selectedFile = e.target.files[0];
    fileName.innerText = selectedFile.name;
    analyzeMediaBtn.disabled = false;
  }
});

analyzeMediaBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  showLoading(`Analyzing ${activeMode}...`);
  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/detect/${activeMode}`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    renderVerdict(data.verdict, data.confidence, data.plain_explanation);
  } catch (err) {
    renderVerdict("ERROR", null, `Failed to reach backend for ${activeMode} analysis.`);
  }
});

analyzeTextBtn.addEventListener("click", async () => {
  const query = claimText.value.trim();
  if (!query) return;

  showLoading("Cross-referencing claim databases...");
  const formData = new FormData();
  formData.append("text", query);

  try {
    const res = await fetch(`${API_BASE}/detect/text`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    renderVerdict(data.verdict, null, data.plain_explanation);
  } catch (err) {
    renderVerdict("ERROR", null, "Failed to verify text claim.");
  }
});

function showLoading(msg) {
  resultBox.classList.remove("hidden");
  verdictTitle.innerText = "PROCESSING";
  verdictTitle.style.color = "#f59e0b";
  verdictConfidence.innerText = "";
  plainExplanation.innerText = msg;
}

function renderVerdict(verdict, conf, text) {
  resultBox.classList.remove("hidden");
  verdictTitle.innerText = verdict;
  const isBad = verdict === "MANIPULATED" || verdict === "SYNTHETIC" || verdict === "DEBUNKED";
  verdictTitle.style.color = isBad ? "#ef4444" : "#10b981";
  verdictConfidence.innerText = conf !== null && conf !== undefined ? `${(conf * 100).toFixed(0)}%` : "";
  verdictConfidence.style.color = verdictTitle.style.color;
  plainExplanation.innerText = text;
}