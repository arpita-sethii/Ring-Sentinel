const API = '';

async function fetchJSON(url, opts){
  return fetch(API + url, { credentials: 'same-origin', ...opts });
}

const NAV_ITEMS = [
  { href: 'investigate.html', label: 'Investigate' },
  { href: 'risk-api.html', label: 'Risk API' },
  { href: 'model-health.html', label: 'Model Health' },
  { href: 'devices.html', label: 'Devices' },
  { href: 'audit.html', label: 'Audit Log' },
];

async function renderNav(activePage){
  const nav = document.getElementById('app-nav');
  if(!nav) return;

  const itemsHtml = NAV_ITEMS.map(item => `
    <a class="nav-btn${item.href === activePage ? ' active' : ''}" href="${item.href}">
      <span class="label">${item.label}</span>
    </a>
  `).join('');

  nav.innerHTML = `
    <div class="nav-section-label">Ring Sentinel</div>
    ${itemsHtml}
    <div class="nav-auth" id="navAuth">Checking session…</div>
    <button id="app-nav-toggle">&#171; Collapse</button>
  `;

  document.getElementById('app-nav-toggle').onclick = () => {
    const collapsed = nav.classList.toggle('collapsed');
    document.body.classList.toggle('nav-collapsed', collapsed);
    document.getElementById('app-nav-toggle').innerHTML = collapsed ? '&#187;' : '&#171; Collapse';
  };

  const authEl = document.getElementById('navAuth');
  try {
    const res = await fetchJSON('/api/auth/me');
    if(res.ok){
      const data = await res.json();
      authEl.innerHTML = `<span class="who">${data.username}</span><a href="#" id="logoutLink">Log out</a>`;
      document.getElementById('logoutLink').onclick = async (e) => {
        e.preventDefault();
        await fetchJSON('/api/auth/logout', { method: 'POST' });
        window.location.href = 'login.html';
      };
    } else {
      authEl.innerHTML = `<a href="login.html">Log in</a> to take actions`;
    }
  } catch(err){
    authEl.innerHTML = `<a href="login.html">Log in</a> to take actions`;
  }
}
