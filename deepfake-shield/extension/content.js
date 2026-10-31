chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const targetImg = Array.from(document.images).find(img => img.src === request.srcUrl);

  if (request.action === "START_LOADING" && targetImg) {
    targetImg.style.outline = "3px dashed #f59e0b";
  }

  if (request.action === "SHOW_ERROR" && targetImg) {
    targetImg.style.outline = "3px solid #ef4444";
    alert(`Deepfake Shield error: ${request.error}`);
  }

  if (request.action === "SHOW_RESULT" && targetImg) {
    const data = request.data;
    const isTampered = data.verdict === "MANIPULATED" || data.verdict === "SYNTHETIC";

    // Set border outline
    targetImg.style.outline = isTampered ? "4px solid #ef4444" : "4px solid #10b981";

    // Render banner badge
    const badge = document.createElement("div");
    const rect = targetImg.getBoundingClientRect();
    badge.style.position = "absolute";
    badge.style.top = `${rect.top + window.scrollY}px`;
    badge.style.left = `${rect.left + window.scrollX}px`;
    badge.style.background = isTampered ? "#ef4444" : "#10b981";
    badge.style.color = "#ffffff";
    badge.style.padding = "6px 12px";
    badge.style.borderRadius = "4px";
    badge.style.fontWeight = "bold";
    badge.style.fontSize = "12px";
    badge.style.fontFamily = "sans-serif";
    badge.style.zIndex = "999999";
    badge.style.boxShadow = "0 4px 6px rgba(0,0,0,0.4)";
    badge.innerText = `${data.verdict} (${(data.confidence * 100).toFixed(0)}%) - ${data.plain_explanation}`;
    document.body.appendChild(badge);

    // Render suspicious bounding boxes
    if (data.suspicious_regions && data.suspicious_regions.length > 0) {
      data.suspicious_regions.forEach(region => {
        const [y1, x1, y2, x2] = region.box_2d;
        const overlayBox = document.createElement("div");
        overlayBox.style.position = "absolute";

        const scaleX = rect.width / (targetImg.naturalWidth || rect.width);
        const scaleY = rect.height / (targetImg.naturalHeight || rect.height);

        overlayBox.style.top = `${rect.top + window.scrollY + (y1 * scaleY)}px`;
        overlayBox.style.left = `${rect.left + window.scrollX + (x1 * scaleX)}px`;
        overlayBox.style.width = `${(x2 - x1) * scaleX}px`;
        overlayBox.style.height = `${(y2 - y1) * scaleY}px`;
        overlayBox.style.border = "3px solid #ef4444";
        overlayBox.style.backgroundColor = "rgba(239, 68, 68, 0.25)";
        overlayBox.style.zIndex = "999998";
        overlayBox.title = `${region.label} (${region.score})`;
        document.body.appendChild(overlayBox);
      });
    }
  }
});