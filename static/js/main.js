/* =====================================================================
   SmartAIShop — Client-side interactions
   ===================================================================== */

/* ---- Live search (search-as-you-type) ---- */
const searchInput    = document.getElementById('searchInput');
const searchDropdown = document.getElementById('searchDropdown');

if (searchInput) {
  let debounceTimer;

  searchInput.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    const q = searchInput.value.trim();

    if (q.length < 2) {
      searchDropdown.innerHTML = '';
      searchDropdown.classList.remove('open');
      return;
    }

    debounceTimer = setTimeout(() => {
      fetch(`/api/search?q=${encodeURIComponent(q)}`)
        .then(r => r.json())
        .then(items => {
          if (!items.length) {
            searchDropdown.classList.remove('open');
            return;
          }
          searchDropdown.innerHTML = items.map(p => `
            <div class="search-item" onclick="window.location='/product/${p.product_id}'">
              <span>${p.emoji}</span>
              <div>
                <div class="fw-semibold" style="font-size:.88rem">${p.name}</div>
                <div class="text-muted" style="font-size:.76rem">${p.category} · €${parseFloat(p.price).toFixed(2)}</div>
              </div>
            </div>`).join('');
          searchDropdown.classList.add('open');
        });
    }, 280);
  });

  /* Close dropdown when clicking outside */
  document.addEventListener('click', e => {
    if (!searchInput.contains(e.target) && !searchDropdown.contains(e.target)) {
      searchDropdown.classList.remove('open');
    }
  });

  /* Submit on Enter */
  searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      window.location = `/?search=${encodeURIComponent(searchInput.value)}`;
    }
  });
}
