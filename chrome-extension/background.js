// 背景服务：YouTube双语字幕翻译器
console.log('🔧 Background Service Worker 启动');

// 跨域请求代理（解决内容脚本 CORS 限制）
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('📨 收到消息:', message?.type);
  
  if (message && message.type === 'PROXY_FETCH') {
    const { url, options } = message;
    console.log('🌐 代理请求:', url);
    
    (async () => {
      try {
        const resp = await fetch(url, options || {});
        const headers = {};
        resp.headers.forEach((v, k) => { headers[k] = v; });
        const bodyText = await resp.text();
        
        console.log('✅ 代理请求成功:', resp.status);
        sendResponse({ ok: resp.ok, status: resp.status, headers, bodyText });
      } catch (err) {
        console.error('❌ 代理请求失败:', err);
        sendResponse({ ok: false, status: 0, error: err?.message || String(err) });
      }
    })();
    return true; // 异步响应
  }
  
  // 其他消息类型
  if (message && message.type === 'PING') {
    console.log('🏓 收到ping');
    sendResponse({ success: true, timestamp: Date.now() });
    return true;
  }
});

// Service Worker 错误处理
self.addEventListener('error', (event) => {
  console.error('🚨 Service Worker 错误:', event.error);
});

// 安装和激活事件
self.addEventListener('install', (event) => {
  console.log('📦 Service Worker 安装');
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  console.log('🚀 Service Worker 激活');
  event.waitUntil(clients.claim());
});
