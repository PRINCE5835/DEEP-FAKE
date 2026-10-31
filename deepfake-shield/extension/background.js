chrome.runtime.onInstalled.addListener(() => {
  // Image context menu
  chrome.contextMenus.create({
    id: "shield-inspect-image",
    title: "🛡️ Verify Image with DEEP-SHIELD",
    contexts: ["image"],
  });

  // Video context menu
  chrome.contextMenus.create({
    id: "shield-inspect-video",
    title: "🛡️ Verify Video with DEEP-SHIELD",
    contexts: ["video"],
  });

  // Audio context menu
  chrome.contextMenus.create({
    id: "shield-inspect-audio",
    title: "🛡️ Verify Audio with DEEP-SHIELD",
    contexts: ["audio"],
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

// Map menu item IDs to API endpoints
const ENDPOINT_BY_MENU = {
  "shield-inspect-image": "detect/image",
  "shield-inspect-video": "detect/video",
  "shield-inspect-audio": "detect/audio",
};

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const menuId = info.menuItemId;
  const endpoint = ENDPOINT_BY_MENU[menuId];
  if (!endpoint || !tab?.id) {
    return;
  }

  if (tab.url.startsWith("chrome://") || tab.url.startsWith("edge://") || tab.url.startsWith("about:")) {
    return;
  }

  try {
    // 1. Ensure content.js is injected
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"],
    });

    // 2. Notify DOM to show loading outline (for images/video)
    if (menuId === "shield-inspect-image" || menuId === "shield-inspect-video") {
      chrome.tabs.sendMessage(tab.id, {
        action: "START_LOADING",
        srcUrl: info.srcUrl,
      });
    }

    // 3. Resolve media Blob
    let blob;
    if (!info.srcUrl) {
      throw new Error("No media URL available for this element. Please try right-clicking directly on the media file.");
    }
    if (info.srcUrl.startsWith("data:")) {
      blob = dataURItoBlob(info.srcUrl);
    } else if (info.srcUrl.startsWith("blob:") || info.srcUrl.startsWith("mediastream:")) {
      throw new Error("Cannot inspect blob/stream media URLs directly. Please download the file and use the extension popup instead.");
    } else {
      const resp = await fetch(info.srcUrl);
      if (!resp.ok) {
        throw new Error(`Failed to fetch media: HTTP ${resp.status}`);
      }
      blob = await resp.blob();
    }

    if (!blob || blob.size === 0) {
      throw new Error("Unable to capture media bytes from element.");
    }

    // 4. Construct FormData with a sensible filename
    const ext = menuId === "shield-inspect-image" ? "jpg" :
                menuId === "shield-inspect-video" ? "mp4" : "ogg";
    const formData = new FormData();
    formData.append("file", blob, `inspect.${ext}`);

    // 5. Transmit to backend
    const apiResp = await fetch(`http://127.0.0.1:8000/api/v1/${endpoint}`, {
      method: "POST",
      body: formData,
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
      data: data,
      mediaType: menuId.replace("shield-inspect-", ""),
    });

  } catch (err) {
    console.error("Verification pipeline error:", err);
    chrome.tabs.sendMessage(tab.id, {
      action: "SHOW_ERROR",
      srcUrl: info.srcUrl,
      error: err.message,
      mediaType: menuId.replace("shield-inspect-", ""),
    });
  }
});
