// Animate probability bars on load
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.bar-fill').forEach(bar => {
    const w = bar.style.width;
    bar.style.width = '0';
    setTimeout(() => { bar.style.width = w; }, 100);
  });

  // Highlight active nav link
  const path = window.location.pathname;
  document.querySelectorAll('.navbar ul a').forEach(a => {
    if (a.getAttribute('href') === path) a.style.background = 'rgba(255,255,255,.2)';
  });
});

// Form loading state
document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', () => {
    const btn = form.querySelector('button[type=submit]');
    if (btn) { btn.textContent = '⏳ Analyzing...'; btn.disabled = true; }
  });
});
