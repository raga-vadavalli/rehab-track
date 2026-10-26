/* RehaTrack Dashboard -- app.js */

// -- State ----------------------------------------------------------------
let prescriptions   = {};
let selectedKey     = null;   // key in prescriptions currently selected
let isNewMode       = false;  // true when creating a new prescription
let sessionActive   = false;
let sseSource       = null;
let lastSummary     = null;

// -- Helpers --------------------------------------------------------------
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function toast(msg, type = "ok") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className   = `show ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 3200);
}

function showSection(id) {
  document.querySelectorAll("main section").forEach(s => s.classList.remove("visible"));
  document.getElementById(`sec-${id}`).classList.add("visible");
  document.querySelectorAll(".nav-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.page === id);
  });
}

function fmtDuration(s) {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

// -- Nav ------------------------------------------------------------------
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const page = btn.dataset.page;
    if (page === "history") loadHistory();
    showSection(page);
  });
});

// Nav stop button -- always visible when session is active
document.getElementById("nav-stop-btn").addEventListener("click", stopSession);

// -- Prescriptions --------------------------------------------------------
async function loadPrescriptions() {
  try {
    prescriptions = await api("GET", "/prescriptions");
  } catch (e) {
    toast("Failed to load prescriptions: " + e.message, "error");
    prescriptions = {};
  }
  renderPrescList();
}

function renderPrescList() {
  const list = document.getElementById("presc-list");
  const keys = Object.keys(prescriptions);

  if (keys.length === 0) {
    list.innerHTML = '<div class="empty-state">No prescriptions yet. Add one below.</div>';
    return;
  }

  list.innerHTML = keys.map(key => {
    const p = prescriptions[key];
    return `
      <div class="presc-card ${key === selectedKey ? "active" : ""}"
           data-key="${key}">
        <div class="presc-card-info">
          <span class="presc-name">${p.display_name}</span>
          <span class="presc-detail">
            ${p.target_reps} reps &nbsp;·&nbsp;
            ext ${p.extension_angle} deg &nbsp;·&nbsp;
            flex ${p.flexion_angle} deg
          </span>
        </div>
        <div class="presc-card-actions">
          <button class="btn-icon delete-btn" data-key="${key}" title="Delete">✕</button>
        </div>
      </div>`;
  }).join("");

  // Select card
  list.querySelectorAll(".presc-card").forEach(card => {
    card.addEventListener("click", e => {
      if (e.target.classList.contains("delete-btn")) return;
      selectPrescription(card.dataset.key);
    });
  });

  // Delete buttons
  list.querySelectorAll(".delete-btn").forEach(btn => {
    btn.addEventListener("click", () => deletePrescription(btn.dataset.key));
  });
}

function selectPrescription(key) {
  selectedKey = key;
  isNewMode   = false;
  const p = prescriptions[key];
  document.getElementById("form-title").textContent   = `Review -- ${p.display_name}`;
  document.getElementById("f-name").value     = p.display_name;
  document.getElementById("f-reps").value     = p.target_reps;
  document.getElementById("f-ext").value      = p.extension_angle;
  document.getElementById("f-flex").value     = p.flexion_angle;
  const ex = p.exercise || "bicep_curl";
  document.getElementById("f-exercise").value = ex;
  applyExerciseMeta(ex);
  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-save").textContent  = "Save Changes";
  document.getElementById("btn-cancel").style.display = "none";
  clearErrors();
  renderPrescList();  // re-render to highlight active card
}

function enterNewMode() {
  selectedKey = null;
  isNewMode   = true;
  document.getElementById("form-title").textContent = "New Prescription";
  document.getElementById("f-name").value     = "";
  document.getElementById("f-reps").value     = "5";
  document.getElementById("f-exercise").value = "bicep_curl";
  applyExerciseMeta("bicep_curl");
  document.getElementById("f-ext").value   = "150";
  document.getElementById("f-flex").value  = "60";
  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-save").textContent  = "Create";
  document.getElementById("btn-cancel").style.display = "";
  clearErrors();
  renderPrescList();
}

// -- Exercise selector -------------------------------------------------------
const EXERCISE_META = {
  bicep_curl: {
    extLabel:  "Extension Angle (arm straight)",
    flexLabel: "Flexion Angle (arm curled)",
    extHint:   "90\u00b0 \u2013 180\u00b0  (ask your physiotherapist)",
    flexHint:  "10\u00b0 \u2013 90\u00b0  (ask your physiotherapist)",
    extMin: 90,  extMax: 180, extDefault: 150,
    flexMin: 10, flexMax: 90,  flexDefault: 60,
    angleLabel: "Elbow Angle",
  },
  squat: {
    extLabel:  "Extension Angle (leg straight)",
    flexLabel: "Flexion Angle (squat depth)",
    extHint:   "140\u00b0 \u2013 180\u00b0  (ask your physiotherapist)",
    flexHint:  "70\u00b0 \u2013 130\u00b0  (ask your physiotherapist)",
    extMin: 140, extMax: 180, extDefault: 160,
    flexMin: 70, flexMax: 130, flexDefault: 100,
    angleLabel: "Knee Angle",
  },
};

function applyExerciseMeta(exercise) {
  const m = EXERCISE_META[exercise] || EXERCISE_META.bicep_curl;
  const showAngles = exercise === "bicep_curl";
  document.getElementById("group-ext").style.display  = showAngles ? "" : "none";
  document.getElementById("group-flex").style.display = showAngles ? "" : "none";
  if (showAngles) {
    document.getElementById("lbl-ext").textContent   = m.extLabel;
    document.getElementById("lbl-flex").textContent  = m.flexLabel;
    document.getElementById("hint-ext").textContent  = m.extHint;
    document.getElementById("hint-flex").textContent = m.flexHint;
    const extEl  = document.getElementById("f-ext");
    const flexEl = document.getElementById("f-flex");
    extEl.min  = m.extMin;  extEl.max  = m.extMax;
    flexEl.min = m.flexMin; flexEl.max = m.flexMax;
  }
}

document.getElementById("f-exercise").addEventListener("change", e => {
  applyExerciseMeta(e.target.value);
});

document.getElementById("btn-new").addEventListener("click", enterNewMode);
document.getElementById("btn-cancel").addEventListener("click", () => {
  isNewMode   = false;
  selectedKey = null;
  document.getElementById("form-title").textContent = "Select a patient to review";
  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-cancel").style.display = "none";
  renderPrescList();
});

// -- Validation -----------------------------------------------------------
function clearErrors() {
  ["name","reps","ext","flex"].forEach(id => {
    const el = document.getElementById(`err-${id}`);
    el.textContent = "";
    el.classList.remove("visible");
  });
}

function showError(id, msg) {
  const el = document.getElementById(`err-${id}`);
  el.textContent = msg;
  el.classList.add("visible");
}

function validateForm() {
  clearErrors();
  let ok = true;
  const exercise = document.getElementById("f-exercise").value || "bicep_curl";
  const m    = EXERCISE_META[exercise] || EXERCISE_META.bicep_curl;
  const name = document.getElementById("f-name").value.trim();
  const reps = parseFloat(document.getElementById("f-reps").value);
  const ext  = parseFloat(document.getElementById("f-ext").value);
  const flex = parseFloat(document.getElementById("f-flex").value);

  if (!name)              { showError("name", "Name is required.");          ok = false; }
  if (isNaN(reps) || reps < 1 || reps > 50)
    { showError("reps", "Reps must be between 1 and 50.");                   ok = false; }
  if (exercise === "bicep_curl") {
    if (isNaN(ext)  || ext < m.extMin  || ext > m.extMax)
      { showError("ext",  `Extension angle must be ${m.extMin}\u00b0-${m.extMax}\u00b0.`);  ok = false; }
    if (isNaN(flex) || flex < m.flexMin || flex > m.flexMax)
      { showError("flex", `Flexion angle must be ${m.flexMin}\u00b0-${m.flexMax}\u00b0.`);  ok = false; }
    if (ok && flex >= ext)
      { showError("flex", "Flexion angle must be less than extension angle."); ok = false; }
  }
  return ok;
}

function formPayload() {
  const exercise = document.getElementById("f-exercise").value || "bicep_curl";
  const m = EXERCISE_META[exercise] || EXERCISE_META.bicep_curl;
  const ext  = exercise === "bicep_curl"
    ? parseFloat(document.getElementById("f-ext").value)
    : m.extDefault;
  const flex = exercise === "bicep_curl"
    ? parseFloat(document.getElementById("f-flex").value)
    : m.flexDefault;
  return {
    display_name:    document.getElementById("f-name").value.trim(),
    target_reps:     parseInt(document.getElementById("f-reps").value),
    extension_angle: ext,
    flexion_angle:   flex,
    exercise,
    arm: "right",
  };
}

// -- Save -----------------------------------------------------------------
document.getElementById("btn-save").addEventListener("click", async () => {
  if (!validateForm()) return;
  const payload = formPayload();
  try {
    if (isNewMode) {
      const res = await api("POST", "/prescriptions", payload);
      toast(`Created profile for ${payload.display_name}`, "ok");
      await loadPrescriptions();
      selectPrescription(res.key);
      isNewMode = false;
    } else if (selectedKey) {
      await api("PUT", `/prescriptions/${selectedKey}`, payload);
      toast("Prescription saved.", "ok");
      await loadPrescriptions();
      selectPrescription(selectedKey);
    }
  } catch (e) {
    toast("Save failed: " + e.message, "error");
  }
});

// -- Delete ---------------------------------------------------------------
async function deletePrescription(key) {
  if (!confirm(`Delete "${prescriptions[key].display_name}"?`)) return;
  try {
    await api("DELETE", `/prescriptions/${key}`);
    if (selectedKey === key) {
      selectedKey = null;
      document.getElementById("btn-start").disabled = true;
      document.getElementById("form-title").textContent = "Select a patient to review";
    }
    await loadPrescriptions();
    toast("Profile deleted.", "ok");
  } catch (e) {
    toast("Delete failed: " + e.message, "error");
  }
}

// -- Start session ---------------------------------------------------------
document.getElementById("btn-start").addEventListener("click", async () => {
  if (!validateForm()) return;
  const payload = formPayload();
  try {
    await api("POST", "/session/start", {
      prescription_key: selectedKey || "new",
      prescription:     payload,
    });
    sessionActive = true;
    document.getElementById("nav-stop-btn").style.display = "";
    document.getElementById("session-title").textContent =
      `Session -- ${payload.display_name}`;
    document.getElementById("stat-reps").textContent = `0 / ${payload.target_reps}`;
    // Update angle label based on exercise
    const exMeta = EXERCISE_META[payload.exercise] || EXERCISE_META.bicep_curl;
    document.getElementById("stat-angle-label").textContent = exMeta.angleLabel;
    document.getElementById("video-feed").src = "/video_feed?" + Date.now();
    showSection("session");
    startSSE(payload.target_reps);
    toast("Session started.", "ok");
  } catch (e) {
    toast("Could not start session: " + e.message, "error");
  }
});

// -- SSE updates -----------------------------------------------------------
function startSSE(targetReps) {
  if (sseSource) { sseSource.close(); sseSource = null; }
  sseSource = new EventSource("/events");

  sseSource.onmessage = e => {
    const s = JSON.parse(e.data);
    if (!s.active) return;

    // Reps
    document.getElementById("stat-reps").textContent =
      `${s.reps} / ${targetReps}`;

    // Angle
    document.getElementById("stat-angle").textContent =
      s.angle !== null ? `${s.angle} deg` : "--";

    // Phase
    document.getElementById("stat-phase").textContent =
      s.phase ? s.phase.charAt(0).toUpperCase() + s.phase.slice(1) : "Waiting";

    // Elapsed
    document.getElementById("stat-elapsed").textContent =
      fmtDuration(s.elapsed);

    // Feedback
    const fb = document.getElementById("stat-feedback");
    fb.textContent = s.feedback || "";
    fb.className   = "form-bar " + (s.is_valid_form ? "good" : "bad");

    // Auto-complete banner
    if (s.session_done && sessionActive) {
      toast(`Target reached! ${s.reps} reps completed.`, "ok");
    }
  };

  sseSource.onerror = () => {
    // Connection lost -- stop trying silently
    sseSource.close();
  };
}

// -- Stop session ----------------------------------------------------------
async function stopSession() {
  try {
    const summary = await api("POST", "/session/stop");
    sessionActive = false;
    document.getElementById("nav-stop-btn").style.display = "none";
    if (sseSource) { sseSource.close(); sseSource = null; }
    document.getElementById("video-feed").src = "";
    lastSummary = summary;
    renderSummary(summary);
    showSection("summary");
    toast("Session saved.", "ok");
  } catch (e) {
    toast("Stop failed: " + e.message, "error");
  }
}

document.getElementById("btn-stop").addEventListener("click", stopSession);

// -- Summary ---------------------------------------------------------------
function renderSummary(s) {
  const statusBadge = s.status === "complete"
    ? `<span class="badge badge-complete">Complete</span>`
    : `<span class="badge badge-incomplete">Incomplete</span>`;

  document.getElementById("summary-grid").innerHTML = `
    <div class="summary-item">
      <div class="label">Patient</div>
      <div class="value">${s.patient}</div>
    </div>
    <div class="summary-item">
      <div class="label">Status</div>
      <div class="value">${statusBadge}</div>
    </div>
    <div class="summary-item">
      <div class="label">Reps Completed</div>
      <div class="value">${s.completed_reps} / ${s.target_reps}</div>
    </div>
    <div class="summary-item">
      <div class="label">Duration</div>
      <div class="value">${fmtDuration(s.duration_s)}</div>
    </div>
    <div class="summary-item">
      <div class="label">Extension Angle</div>
      <div class="value">${s.extension_angle} deg</div>
    </div>
    <div class="summary-item">
      <div class="label">Flexion Angle</div>
      <div class="value">${s.flexion_angle} deg</div>
    </div>
    <div class="summary-item">
      <div class="label">Date / Time</div>
      <div class="value">${s.timestamp}</div>
    </div>
  `;
}

document.getElementById("btn-new-session").addEventListener("click", () => {
  showSection("prescriptions");
});

document.getElementById("btn-go-history").addEventListener("click", () => {
  loadHistory();
  showSection("history");
});

// -- History ---------------------------------------------------------------
async function loadHistory() {
  const list = document.getElementById("history-list");
  list.innerHTML = '<div class="empty-state">Loading...</div>';
  try {
    const sessions = await api("GET", "/sessions");
    if (sessions.length === 0) {
      list.innerHTML = '<div class="empty-state">No sessions recorded yet.</div>';
      return;
    }
    list.innerHTML = sessions.map(s => {
      const badge = s.status === "complete"
        ? `<span class="badge badge-complete">Complete</span>`
        : `<span class="badge badge-incomplete">Incomplete</span>`;
      return `
        <div class="history-card">
          <div class="history-meta">
            <span class="history-name">${s.patient}</span>
            <span class="history-detail">
              ${s.completed_reps}/${s.target_reps} reps &nbsp;·&nbsp;
              ${fmtDuration(s.duration_s)} &nbsp;·&nbsp;
              ${s.timestamp}
            </span>
          </div>
          ${badge}
        </div>`;
    }).join("");
  } catch (e) {
    list.innerHTML = `<div class="empty-state">Failed to load history.</div>`;
    toast("History load failed: " + e.message, "error");
  }
}

// -- Init ------------------------------------------------------------------
loadPrescriptions();
