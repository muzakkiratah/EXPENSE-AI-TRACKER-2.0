// ExpenseAI - Frontend interactivity
document.addEventListener("DOMContentLoaded", function () {

  // ---------------------------------------------------------------------
  // Mobile sidebar toggle
  // ---------------------------------------------------------------------
  const sidebar = document.getElementById("sidebar");
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const sidebarBackdrop = document.getElementById("sidebarBackdrop");

  function openSidebar() {
    if (!sidebar) return;
    sidebar.classList.add("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.add("open");
  }

  function closeSidebar() {
    if (!sidebar) return;
    sidebar.classList.remove("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.remove("open");
  }

  if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener("click", function () {
      if (sidebar.classList.contains("open")) {
        closeSidebar();
      } else {
        openSidebar();
      }
    });
  }

  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", closeSidebar);
  }

  // Close the mobile sidebar automatically once a nav link is tapped,
  // so it doesn't stay open behind the page that just loaded.
  if (sidebar) {
    sidebar.querySelectorAll("a.nav-link").forEach((link) => {
      link.addEventListener("click", closeSidebar);
    });
  }

  // If the window is resized back up to desktop width, make sure the
  // mobile-only open state doesn't linger.
  window.addEventListener("resize", function () {
    if (window.innerWidth > 768) closeSidebar();
  });

  // ---------------------------------------------------------------------
  // Search popup
  // ---------------------------------------------------------------------
  const openSearchBtn = document.getElementById("openSearchBtn");
  const closeSearchBtn = document.getElementById("closeSearchBtn");
  const searchOverlay = document.getElementById("searchOverlay");
  const searchInput = document.getElementById("searchInput");
  const quickSearchButtons = document.querySelectorAll(".quick-search-btn");

  function openSearch() {
    if (!searchOverlay) return;
    searchOverlay.classList.add("open");
    setTimeout(() => searchInput && searchInput.focus(), 50);
  }

  function closeSearch() {
    if (!searchOverlay) return;
    searchOverlay.classList.remove("open");
  }

  if (openSearchBtn) openSearchBtn.addEventListener("click", openSearch);
  if (closeSearchBtn) closeSearchBtn.addEventListener("click", closeSearch);

  if (searchOverlay) {
    searchOverlay.addEventListener("click", function (e) {
      if (e.target === searchOverlay) closeSearch();
    });
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeSearch();
  });

  quickSearchButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
      const value = btn.getAttribute("data-value");
      window.location.href = "/search?query=" + encodeURIComponent(value);
    });
  });

  // ---------------------------------------------------------------------
  // Automatic ML category prediction (Add Expense page)
  // ---------------------------------------------------------------------
  const descriptionInput = document.getElementById("description");
  const categorySelect = document.getElementById("category");
  const aiSuggestedTag = document.getElementById("aiSuggestedTag");

  if (descriptionInput && categorySelect && window.EXPENSEAI_PREDICT_URL) {
    let debounceTimer = null;
    let userManuallyChangedCategory = false;

    // If the user manually picks a category, stop auto-overriding it.
    categorySelect.addEventListener("change", function () {
      userManuallyChangedCategory = true;
      if (aiSuggestedTag) aiSuggestedTag.style.display = "none";
    });

    descriptionInput.addEventListener("input", function () {
      clearTimeout(debounceTimer);
      const description = descriptionInput.value.trim();

      if (!description) {
        if (aiSuggestedTag) aiSuggestedTag.style.display = "none";
        return;
      }

      debounceTimer = setTimeout(function () {
        fetch(window.EXPENSEAI_PREDICT_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ description: description }),
        })
          .then((response) => {
            if (!response.ok) throw new Error("Prediction request failed");
            return response.json();
          })
          .then((data) => {
            if (!data || !data.category) return;
            if (userManuallyChangedCategory) return; // respect user's own choice

            categorySelect.value = data.category;
            if (aiSuggestedTag) aiSuggestedTag.style.display = "inline-block";
          })
          .catch((err) => {
            // Fail silently - user can still pick a category manually.
            console.warn("ExpenseAI: category prediction unavailable.", err);
          });
      }, 600);
    });
  }
});
