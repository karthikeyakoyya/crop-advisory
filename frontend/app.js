(function () {
  "use strict";

  const API_BASE = window.location.origin.includes("null") ? "http://localhost:8000" : "";

  const districtSelect = document.getElementById("district");
  const cropSelect = document.getElementById("crop");
  const form = document.getElementById("advisory-form");
  const submitBtn = document.getElementById("submit-btn");
  const formError = document.getElementById("form-error");
  const advancedToggle = document.getElementById("advanced-toggle");
  const advancedPanel = document.getElementById("advanced-panel");
  const advancedChevron = document.getElementById("advanced-chevron");

  const resultsEmpty = document.getElementById("results-empty");
  const resultsBody = document.getElementById("results-body");

  advancedToggle.addEventListener("click", function () {
    const isOpen = advancedPanel.classList.toggle("open");
    advancedChevron.className = isOpen ? "ti ti-chevron-up" : "ti ti-chevron-down";
  });

  function todayISO() {
    return new Date().toISOString().split("T")[0];
  }
  document.getElementById("sowing_date").value = todayISO();

  async function fetchJSON(path) {
    const res = await fetch(API_BASE + path);
    if (!res.ok) throw new Error("Request to " + path + " failed with status " + res.status);
    return res.json();
  }

  async function loadReferenceData() {
    try {
      const [districts, crops] = await Promise.all([
        fetchJSON("/api/districts"),
        fetchJSON("/api/crops"),
      ]);
      districtSelect.innerHTML = districts.districts
        .map((d) => `<option value="${d}">${d}</option>`)
        .join("");
      cropSelect.innerHTML = crops.crops
        .map((c) => `<option value="${c}">${c}</option>`)
        .join("");
    } catch (err) {
      formError.textContent =
        "Could not reach the advisory API. Make sure the FastAPI server is running (see README) and reload this page.";
      formError.style.display = "block";
      console.error(err);
    }
  }

  function numOrNull(id) {
    const val = document.getElementById(id).value;
    return val === "" ? null : parseFloat(val);
  }

  function fmt(n, decimals) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Number(n).toLocaleString(undefined, {
      minimumFractionDigits: decimals || 0,
      maximumFractionDigits: decimals || 0,
    });
  }

  function renderResults(data) {
    document.getElementById("res-location").textContent = data.district + " — " + data.crop;
    document.getElementById("res-dates").textContent =
      "sown " + data.sowing_date + "  ·  harvest " + data.harvest_date;
    document.getElementById("res-yield").textContent = fmt(data.predicted_yield_kg_per_ha);
    document.getElementById("res-band").textContent =
      "kg/ha  ·  80% band " + fmt(data.confidence_interval_80pct.low) + "–" + fmt(data.confidence_interval_80pct.high);

    // Rain ledger gauge: fill = expected rainfall as a % of a generous max
    // scale (2x the "need" reference), marker = the crop's approximate
    // water requirement pulled from the irrigation schedule totals.
    const needTotal = data.irrigation_schedule.reduce((sum, s) => sum + s.irrigation_needed_mm, 0) + data.expected_total_rainfall_mm;
    const maxScale = Math.max(needTotal * 1.3, data.expected_total_rainfall_mm * 1.2, 100);
    const fillPct = Math.min((data.expected_total_rainfall_mm / maxScale) * 100, 100);
    const needPct = Math.min((needTotal / maxScale) * 100, 100);
    const rainFill = document.getElementById("rain-fill");
    const rainNeed = document.getElementById("rain-need-marker");
    rainFill.style.width = "0%";
    requestAnimationFrame(() => { rainFill.style.width = fillPct + "%"; });
    rainNeed.style.left = needPct + "%";
    document.getElementById("rain-expected").textContent = "expected: " + fmt(data.expected_total_rainfall_mm) + " mm";
    document.getElementById("rain-need").textContent = "full-season need: ~" + fmt(needTotal) + " mm";

    const timeline = document.getElementById("irrigation-timeline");
    timeline.innerHTML = data.irrigation_schedule
      .map(
        (s) => `
        <li>
          <div class="stage-date">${s.recommended_start_date}</div>
          <div>
            <div class="stage-name">${s.stage}</div>
            <div class="stage-mm">${fmt(s.irrigation_needed_mm)} mm recommended</div>
            <div class="stage-note">${s.note}</div>
          </div>
        </li>`
      )
      .join("");

    const riskFlags = document.getElementById("risk-flags");
    if (data.risk_flags.length === 0) {
      riskFlags.innerHTML = `<div class="no-risk"><i class="ti ti-circle-check"></i> No elevated drought or excess-rainfall risk detected for this season.</div>`;
    } else {
      riskFlags.innerHTML = data.risk_flags
        .map(
          (f) => `
          <div class="risk-flag ${f.severity}">
            <i class="ti ${f.type === "drought_risk" ? "ti-droplet-off" : "ti-droplet"}"></i>
            <span>${f.message}</span>
          </div>`
        )
        .join("");
    }

    const reasoningList = document.getElementById("reasoning-list");
    reasoningList.innerHTML = data.reasoning.key_drivers_considered.map((d) => `<li>${d}</li>`).join("");
    document.getElementById("reasoning-note").textContent = data.reasoning.note;
    document.getElementById("disclaimer-text").textContent = data.disclaimer;

    resultsEmpty.style.display = "none";
    resultsBody.classList.add("visible");
  }

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    formError.style.display = "none";
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Reading the season…';

    const payload = {
      district: districtSelect.value,
      crop: cropSelect.value,
      season_name: document.getElementById("season_name").value,
      irrigation_source: document.getElementById("irrigation_source").value,
      area_hectares: parseFloat(document.getElementById("area_hectares").value),
      sowing_date: document.getElementById("sowing_date").value,
      soil: {
        soil_ph: numOrNull("soil_ph"),
        soil_moisture_pct: numOrNull("soil_moisture_pct"),
        nitrogen_kg_ha: numOrNull("nitrogen_kg_ha"),
        organic_carbon_pct: numOrNull("organic_carbon_pct"),
      },
    };

    try {
      const res = await fetch(API_BASE + "/api/advisory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.detail || "The advisory API returned an error.");
      }
      renderResults(body);
    } catch (err) {
      formError.textContent = err.message || "Something went wrong generating the outlook. Please try again.";
      formError.style.display = "block";
      console.error(err);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Generate outlook";
    }
  });

  loadReferenceData();
})();
