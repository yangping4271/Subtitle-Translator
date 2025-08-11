document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('settingsForm');
  const statusDiv = document.getElementById('status');
  
  // 加载保存的设置
  chrome.storage.sync.get(['apiUrl', 'apiKey', 'targetLang', 'model', 'autoTranslate'], function(result) {
    document.getElementById('apiUrl').value = result.apiUrl || 'https://api.openai.com/v1';
    document.getElementById('apiKey').value = result.apiKey || '';
    document.getElementById('targetLang').value = result.targetLang || '简体中文';
    document.getElementById('model').value = result.model || 'gpt-4o-mini';
    document.getElementById('autoTranslate').checked = result.autoTranslate !== false;
  });
  
  // 保存设置
  form.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const settings = {
      apiUrl: document.getElementById('apiUrl').value,
      apiKey: document.getElementById('apiKey').value,
      targetLang: document.getElementById('targetLang').value,
      model: document.getElementById('model').value,
      autoTranslate: document.getElementById('autoTranslate').checked
    };
    
    // 验证设置
    if (!settings.apiUrl || !settings.apiKey) {
      showStatus('请填写完整的API设置！', 'error');
      return;
    }
    
    // 保存到Chrome存储
    chrome.storage.sync.set(settings, function() {
      showStatus('设置保存成功！', 'success');
      
      // 通知content script更新设置
      chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (tabs[0] && tabs[0].url.includes('youtube.com')) {
          chrome.tabs.sendMessage(tabs[0].id, {
            type: 'SETTINGS_UPDATED',
            settings: settings
          });
        }
      });
    });
  });
  
  function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = 'block';
    
    setTimeout(() => {
      statusDiv.style.display = 'none';
    }, 3000);
  }
});