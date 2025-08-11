// 背景服务：跨域请求代理（解决内容脚本 CORS 限制）
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === 'PROXY_FETCH') {
    const { url, options } = message;
    (async () => {
      try {
        const resp = await fetch(url, options || {});
        const headers = {};
        resp.headers.forEach((v, k) => { headers[k] = v; });
        const bodyText = await resp.text();
        sendResponse({ ok: resp.ok, status: resp.status, headers, bodyText });
      } catch (err) {
        sendResponse({ ok: false, status: 0, error: err?.message || String(err) });
      }
    })();
    return true; // 异步响应
  }
});
