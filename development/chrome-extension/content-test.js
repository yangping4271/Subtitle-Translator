// 最小化测试版本 - 确认扩展是否能在YouTube页面加载
console.log('🧪 测试版Chrome扩展已加载');

// 创建测试按钮来确认扩展工作
const testButton = document.createElement('button');
testButton.textContent = '✅ 扩展已加载';
testButton.style.cssText = `
  position: fixed;
  top: 10px;
  right: 10px;
  z-index: 999999999;
  background: #00ff00;
  color: black;
  padding: 15px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: bold;
  box-shadow: 0 4px 8px rgba(0,0,0,0.3);
`;

testButton.addEventListener('click', () => {
  alert('Chrome扩展正在正常工作！\n\n页面URL: ' + window.location.href + '\n视频ID: ' + new URLSearchParams(window.location.search).get('v'));
});

// 延迟添加确保DOM完全加载
setTimeout(() => {
  document.body.appendChild(testButton);
  console.log('🧪 测试按钮已添加到页面');
}, 2000);

// 测试快捷键
document.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.key === 'b') {
    event.preventDefault();
    alert('Ctrl+B 快捷键工作正常！');
    console.log('🧪 Ctrl+B 快捷键被触发');
  }
});

console.log('🧪 测试扩展初始化完成');