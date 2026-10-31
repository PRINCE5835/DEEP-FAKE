chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const mediaType = request.mediaType || "image";
  const srcUrl = request.srcUrl;

  // Find the target element based on media type
  let targetEl = null;
  if (mediaType === "image") {
    targetEl = Array.from(document.images).find(img => img.src === srcUrl || img.currentSrc === srcUrl);
  } else if (mediaType === "video") {
    targetEl = Array.from(document.querySelectorAll("video")).find(
      v => v.src === srcUrl || v.currentSrc === srcUrl
    );
  } else if (mediaType === "audio") {
    targetEl = Array.from(document.querySelectorAll("audio")).find(
      a => a.src === srcUrl || a.currentSrc === srcUrl
    );
  }

  if (!targetEl) return;

  if (request.action === "START_LOADING" && targetEl) {
    targetEl.style.outline = "3px dashed #f59e0b";
  }

  if (request.action === "SHOW_ERROR" && targetEl) {
    targetEl.style.outline = "3px solid #ef4444";
    alert(`DEEP-SHIELD error: ${request.error}`);
  }

  if (request.action === "SHOW_RESULT" && targetEl) {
    const data = request.data;
    const isTampered =
      data.verdict === "MANIPULATED" ||
      data.verdict === "SYNTHETIC" ||
      data.verdict === "DEBUNKED" ||
      data.verdict === "SUSPECT_AI";

    // Set border outline
    targetEl.style.outline = isTampered ? "4px solid #ef4444" : "4px solid #10b981";

    // Render banner badge
    const badge = document.createElement("div");
    const rect = targetEl.getBoundingClientRect();
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
    badge.style.maxWidth = "300px";
    badge.style.overflow = "hidden";
    badge.style.textOverflow = "ellipsis";
    badge.innerText = `${data.verdict} (${(data.confidence * 100).toFixed(0)}%) - ${data.plain_explanation}`;
    document.body.appendChild(badge);

    // Render suspicious bounding boxes (image only)
    if (
      mediaType === "image" &&
      data.suspicious_regions &&
      data.suspicious_regions.length > 0
    ) {
      data.suspicious_regions.forEach(region => {
        const [y1, x1, y2, x2] = region.box_2d;
        const overlayBox = document.createElement("div");
        overlayBox.style.position = "absolute";

        const scaleX = rect.width / (targetEl.naturalWidth || rect.width);
        const scaleY = rect.height / (targetEl.naturalHeight || rect.height);

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
