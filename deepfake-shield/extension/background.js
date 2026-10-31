chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "shield-inspect-image",
    title: "🛡️ Verify with Deepfake Shield",
    contexts: ["image"]
  });
});

function dataURItoBlob(dataURI) {
  try {
    const parts = dataURI.split(',');
    const byteString = atob(parts[1]);
    const mimeString = parts[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
  } catch (e) {
    console.error("Base64 conversion failed:", e);
    return null;
  }
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "shield-inspect-image" && tab?.id) {
    if (tab.url.startsWith("chrome://") || tab.url.startsWith("edge://") || tab.url.startsWith("about:")) {
      return;
    }

    try {
      // 1. Ensure content.js is injected
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
      });

      // 2. Notify DOM to show loading outline
      chrome.tabs.sendMessage(tab.id, {
        action: "START_LOADING",
        srcUrl: info.srcUrl
      });

      // 3. Resolve Image Blob
      let blob;
      if (info.srcUrl.startsWith("data:")) {
        blob = dataURItoBlob(info.srcUrl);
      } else {
        const resp = await fetch(info.srcUrl);
        blob = await resp.blob();
      }

      if (!blob || blob.size === 0) {
        throw new Error("Unable to capture image bytes from element.");
      }

      // 4. Construct FormData
      const formData = new FormData();
      formData.append("file", blob, "inspect.jpg");

      // 5. Transmit to backend
      const apiResp = await fetch("http://127.0.0.1:8000/api/v1/detect/image", {
        method: "POST",
        body: formData
      });

      if (!apiResp.ok) {
        const errText = await apiResp.text();
        throw new Error(`API returned HTTP ${apiResp.status}: ${errText}`);
      }

      const data = await apiResp.json();

      // 6. Forward results to DOM for overlay rendering
      chrome.tabs.sendMessage(tab.id, {
        action: "SHOW_RESULT",
        srcUrl: info.srcUrl,
        data: data
      });

    } catch (err) {
      console.error("Verification pipeline error:", err);
      chrome.tabs.sendMessage(tab.id, {
        action: "SHOW_ERROR",
        srcUrl: info.srcUrl,
        error: err.message
      });
    }
  }
});