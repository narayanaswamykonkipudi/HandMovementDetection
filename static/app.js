(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const els = {
    themeToggle: $("theme-toggle"),
    cameraBadge: $("camera-badge"),
    cameraStatusText: $("camera-status-text"),
    modelBadge: $("model-badge"),
    modelStatusText: $("model-status-text"),
    fps: $("fps-label"),
    hand: $("hand-label"),
    start: $("start-camera"),
    stop: $("stop-camera"),
    reload: $("reload-model"),
    reconnectServer: $("reconnect-server"),
    cameraDevice: $("camera-device"),
    video: $("video"),
    videoSkeleton: $("video-skeleton"),
    videoOverlay: $("video-overlay"),
    videoOverlayText: $("video-overlay-text"),
    bannerDemo: $("banner-demo"),
    bannerNoModel: $("banner-nomodel"),
    noModelDetail: $("nomodel-detail"),
    bannerOffline: $("banner-offline"),
    sign: $("sign-display"),
    confidence: $("confidence-value"),
    confidenceMeter: $("confidence-meter"),
    confidenceFill: $("confidence-fill"),
    holdRingProgress: $("hold-ring-progress"),
    holdRingLabel: $("hold-ring-label"),
    message: $("recognition-message"),
    topList: $("top-list"),
    modelKind: $("model-kind"),
    modelClasses: $("model-classes"),
    modelThresholds: $("model-thresholds"),
    modelF1: $("model-f1"),
    modelSmoothing: $("model-smoothing"),
    textOutput: $("text-output"),
    textContent: $("text-content"),
    textPlaceholder: $("text-placeholder"),
    charCount: $("char-count"),
    speak: $("speak"),
    stopSpeak: $("stop-speak"),
    space: $("space"),
    delete: $("delete"),
    undo: $("undo"),
    addCurrent: $("add-current"),
    copy: $("copy-text"),
    download: $("download-text"),
    clear: $("clear"),
    speechStatus: $("speech-status"),
    voiceSelect: $("voice-select"),
    voiceRate: $("voice-rate"),
    voiceRateValue: $("voice-rate-value"),
    voiceVolume: $("voice-volume"),
    voiceVolumeValue: $("voice-volume-value"),
    imageUpload: $("image-upload"),
    imageResult: $("image-result"),
    toast: $("toast"),
    shortcutHelp: $("shortcut-help"),
    shortcutsDialog: $("shortcuts-dialog"),
    closeShortcuts: $("close-shortcuts"),
    aboutButton: $("about-button"),
    aboutDialog: $("about-dialog"),
    closeAbout: $("close-about"),
    demoLearnMore: $("demo-learn-more"),
    aboutDemoNote: $("about-demo-note"),
    developerDetails: $("developer-details"),
    onboarding: $("onboarding"),
    onboardingDismiss: $("onboarding-dismiss"),
    moreMenu: $("more-menu"),
    alphabetGrid: $("alphabet-grid"),
    alphabetNote: $("alphabet-note"),
  };

  const app = {
    data: null,
    busy: false,
    lastText: "",
    lastSign: null,
    browserSpeaking: false,
    eventSource: null,
    pollTimer: null,
    pollInFlight: false,
    reconnectTimer: null,
    reconnectAttempts: 0,
    failures: 0,
    toastTimer: null,
    offline: false,
    cameraWait: false,
    skeletonTimer: null,
    serverSpeech: null,
    voiceSignature: "",
    deviceLabels: [],
    localVoice: localStorage.getItem("asl-voice") || "",
    localRate: Number(localStorage.getItem("asl-rate") || 170),
    localVolume: Number(localStorage.getItem("asl-volume") || 1),
    voiceTimer: null,
    videoRetry: 0,
    firstState: false,
  };

  const DEV_MODE = new URLSearchParams(window.location.search).get("dev") === "1";
  const ONBOARDING_KEY = "asl-onboarding-complete";

  const SHORTCUTS = {
    Space: "Add a space",
    Backspace: "Delete the last character",
    "Ctrl+Z": "Undo",
    Enter: "Speak committed text",
    Escape: "Stop speech / close dialog",
  };

  const ALPHABET_NOTES = {
    A: "Fist with the thumb resting beside the index finger.",
    B: "Flat hand, fingers together, thumb across the palm.",
    C: "Curved hand forming a C shape.",
    D: "Index finger up; other fingers meet the thumb.",
    E: "Fingers bent toward the palm, thumb across them.",
    F: "Index and thumb touch; other three fingers extend.",
    G: "Index and thumb point sideways, close together.",
    H: "Index and middle fingers extend sideways, together.",
    I: "Pinky extends; other fingers curl into the palm.",
    K: "Index and middle up; thumb touches the middle finger.",
    L: "Index points up and thumb points sideways, forming an L.",
    M: "Three fingers fold over the thumb.",
    N: "Two fingers fold over the thumb.",
    O: "Fingers and thumb curve into an O shape.",
    P: "Like K, with the hand angled downward.",
    Q: "Like G, with the hand angled downward.",
    R: "Index and middle fingers crossed.",
    S: "Fist with the thumb across the front of the fingers.",
    T: "Fist with the thumb tucked between index and middle.",
    U: "Index and middle fingers extend together, pointing up.",
    V: "Index and middle fingers extend apart in a V.",
    W: "Index, middle, and ring fingers extend apart.",
    X: "Index finger bent like a hook; other fingers curl.",
    Y: "Thumb and pinky extend; other fingers curl.",
  };

  class ApiError extends Error {
    constructor(message, status, code, details) {
      super(message);
      this.status = status;
      this.code = code || "request_failed";
      this.details = details || null;
    }
  }

  const browserTTS =
    "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;

  function showToast(message, isError = false) {
    if (!message) return;
    els.toast.textContent = message;
    els.toast.className = `toast is-visible${isError ? " is-error" : ""}`;
    clearTimeout(app.toastTimer);
    app.toastTimer = setTimeout(() => {
      els.toast.className = "toast";
    }, 4200);
  }

  function showBanner(element, visible) {
    if (element) element.classList.toggle("hidden", !visible);
  }

  function percent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return `${Math.round(Number(value) * 100)}%`;
  }

  function numeric(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(digits);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function setStatus(element, textElement, text, kind) {
    element.className = `status status-${kind}`;
    textElement.textContent = text;
  }

  async function api(path, payload = null) {
    const options = {
      method: payload === null ? "GET" : "POST",
      cache: "no-store",
      headers: {},
    };
    if (payload !== null) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(payload);
    }
    let response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      throw new ApiError("The server could not be reached.", 0, "network_error", null);
    }
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (!response.ok || !data || data.ok === false) {
      const message =
        (data && (data.message || data.detail)) ||
        `Request failed (${response.status}).`;
      throw new ApiError(
        typeof message === "string" ? message : "Request validation failed.",
        response.status,
        data && data.error,
        data && data.details
      );
    }
    return data;
  }

  function setBusy(value) {
    app.busy = value;
    updateButtons();
  }

  async function postAction(path, payload = {}, button = null) {
    if (app.busy) return null;
    setBusy(true);
    if (button) button.disabled = true;
    try {
      const result = await api(path, payload);
      if (result.message) showToast(result.message);
      return result;
    } catch (error) {
      showToast(error.message || "Action failed.", true);
      return null;
    } finally {
      setBusy(false);
      if (path.startsWith("/api/text/")) closeMoreMenu();
      await refresh();
    }
  }

  function closeMoreMenu() {
    if (els.moreMenu) els.moreMenu.open = false;
  }

  // ------------------------------------------------------------------ live data
  function schedulePoll(delay) {
    clearTimeout(app.pollTimer);
    app.pollTimer = setTimeout(refresh, delay);
  }

  async function refresh() {
    if (app.pollInFlight) return;
    if (app.eventSource && app.eventSource.readyState === EventSource.OPEN) return;
    app.pollInFlight = true;
    try {
      const response = await fetch("/api/state", { cache: "no-store" });
      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const data = await response.json();
      markOnline();
      applyState(data);
    } catch {
      markOffline();
    } finally {
      app.pollInFlight = false;
      schedulePoll(document.visibilityState === "hidden" ? 1500 : 500);
    }
  }

  function connectEvents(manual = false) {
    clearTimeout(app.reconnectTimer);
    clearTimeout(app.pollTimer);
    if (!("EventSource" in window)) {
      schedulePoll(250);
      return;
    }
    if (app.eventSource) app.eventSource.close();
    const source = new EventSource("/api/events");
    app.eventSource = source;

    source.addEventListener("state", (event) => {
      app.reconnectAttempts = 0;
      markOnline();
      try {
        applyState(JSON.parse(event.data));
      } catch {
        // A malformed frame should not tear down the stream.
      }
    });

    source.addEventListener("open", () => {
      app.failures = 0;
      app.reconnectAttempts = 0;
      markOnline();
    });

    source.addEventListener("error", () => {
      source.close();
      app.eventSource = null;
      markOffline();
      if (manual) app.reconnectAttempts = 0;
      app.reconnectAttempts += 1;
      const delay = Math.min(8000, 800 * 2 ** Math.min(app.reconnectAttempts - 1, 4));
      clearTimeout(app.reconnectTimer);
      app.reconnectTimer = setTimeout(() => connectEvents(false), delay);
      // Keep a lightweight fallback while SSE is unavailable.
      schedulePoll(1000);
    });
  }

  function markOnline() {
    if (app.offline) {
      app.offline = false;
      showBanner(els.bannerOffline, false);
    }
    app.failures = 0;
  }

  function markOffline() {
    app.failures += 1;
    if (app.failures < 2 && !app.offline) return;
    app.offline = true;
    showBanner(els.bannerOffline, true);
    els.modelBadge.className = "status status-error";
    els.modelStatusText.textContent = "Server offline";
    els.cameraBadge.className = "status status-error";
    els.cameraStatusText.textContent = "Server offline";
    els.videoOverlay.classList.remove("hidden");
    els.videoOverlayText.textContent = "Waiting for the server…";
  }

  function refreshDevices() {
    api("/api/cameras")
      .then((data) => {
        let labels = [];
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
          navigator.mediaDevices
            .enumerateDevices()
            .then((devices) => {
              labels = devices
                .filter((device) => device.kind === "videoinput")
                .map((device) => device.label || "");
              app.deviceLabels = labels;
              renderCameraOptions(data.cameras || [], data.current_index ?? 0);
            })
            .catch(() => renderCameraOptions(data.cameras || [], data.current_index ?? 0));
        } else {
          renderCameraOptions(data.cameras || [], data.current_index ?? 0);
        }
      })
      .catch(() => {
        if (!els.cameraDevice.options.length) {
          els.cameraDevice.replaceChildren(new Option("Camera 0", "0"));
        }
      });
  }

  function renderCameraOptions(cameras, currentIndex) {
    const selected = document.activeElement === els.cameraDevice
      ? els.cameraDevice.value
      : String(currentIndex);
    const fragment = document.createDocumentFragment();
    cameras.forEach((camera, index) => {
      const label = app.deviceLabels[index] || camera.label || `Camera ${camera.index}`;
      fragment.appendChild(new Option(label, String(camera.index)));
    });
    els.cameraDevice.replaceChildren(fragment);
    if ([...els.cameraDevice.options].some((option) => option.value === selected)) {
      els.cameraDevice.value = selected;
    }
  }

  // ------------------------------------------------------------------ rendering
  function friendlyCameraError(message) {
    const value = String(message || "").toLowerCase();
    if (value.includes("permission") || value.includes("denied") || value.includes("authoriz")) {
      return "Camera access is blocked. Allow camera access in your browser, then press Start camera.";
    }
    if (value.includes("not found") || value.includes("no camera") || value.includes("index")) {
      return "No camera was found. Connect a camera, then press Start camera.";
    }
    if (value.includes("busy") || value.includes("in use") || value.includes("using")) {
      return "The camera is being used by another app. Close that app, then press Start camera.";
    }
    if (value.includes("timestamp") || value.includes("inference") || value.includes("landmark")) {
      return "Hand tracking stopped. Stop the camera, then press Start camera.";
    }
    return "The camera could not start. Check the camera connection, then press Start camera.";
  }

  function applyState(data) {
    app.data = data;
    app.firstState = true;
    const training = (data.model_metadata && data.model_metadata.training) || {};
    const labels = Array.isArray(training.labels) ? training.labels : [];
    const loaded = Boolean(data.model_loaded);
    const synthetic = Boolean(data.model_is_synthetic);
    const cameraRunning = Boolean(data.camera_running);
    const fps = Number(data.camera_fps || 0);
    const cameraError = data.camera_error || "";
    const hand = Boolean(data.hand_present);
    const stable = data.stable_label || null;
    const raw = data.raw_label || null;
    const accepted = Boolean(data.raw_accepted);

    if (els.modelBadge) {
      if (loaded) {
        setStatus(
          els.modelBadge,
          els.modelStatusText,
          synthetic ? "Demo model" : "Model ready",
          synthetic ? "warn" : "live"
        );
      } else {
        setStatus(
          els.modelBadge,
          els.modelStatusText,
          data.model_error ? "Model error" : "No model",
          "error"
        );
      }
    }
    showBanner(els.bannerDemo, loaded && synthetic);
    showBanner(els.bannerNoModel, !loaded);
    els.aboutDemoNote.classList.toggle("hidden", !(loaded && synthetic));
    if (!loaded) {
      els.noModelDetail.textContent =
        "Start the app with a trained model, or use the demo mode for a quick preview.";
    }

    let cameraKind = "idle";
    let cameraText = "Camera off";
    if (cameraError) {
      cameraKind = "error";
      cameraText = cameraRunning ? "Camera needs attention" : "Camera unavailable";
    } else if (cameraRunning && fps < 0.05) {
      cameraKind = "warn";
      cameraText = "Starting camera?";
    } else if (cameraRunning) {
      cameraKind = "live";
      cameraText = "Camera live";
    }
    setStatus(els.cameraBadge, els.cameraStatusText, cameraText, cameraKind);

    if (els.fps) els.fps.textContent = `${fps.toFixed(1)} fps`;
    if (els.hand) {
      els.hand.textContent = hand
        ? `Hand: ${data.handedness || "detected"} ? ${numeric(data.hand_score, 2)}`
        : "No hand";
    }

    const opening = cameraRunning && fps < 0.05 && !cameraError;
    els.videoOverlay.classList.toggle("hidden", !(opening || app.cameraWait) || app.offline);
    if (opening || app.cameraWait) {
      els.videoOverlayText.textContent = "Opening camera?";
    }
    if (app.firstState) {
      els.videoSkeleton.classList.add("hidden");
      clearTimeout(app.skeletonTimer);
    }
    if (!document.activeElement || document.activeElement !== els.cameraDevice) {
      if ([...els.cameraDevice.options].some((option) => option.value === String(data.camera_index))) {
        els.cameraDevice.value = String(data.camera_index);
      }
    }
    els.cameraDevice.disabled = cameraRunning || app.busy;

    let display = "?";
    if (stable) display = stable;
    else if (hand && accepted && raw) display = raw;
    else if (hand) display = "?";
    els.sign.textContent = display;
    els.sign.classList.toggle("is-idle", !hand);
    els.sign.classList.toggle("is-uncertain", hand && !accepted);
    if (display !== app.lastSign) {
      if (!prefersReducedMotion()) {
        els.sign.classList.remove("is-bump");
        void els.sign.offsetWidth;
        els.sign.classList.add("is-bump");
      }
      app.lastSign = display;
    }

    const confidence = stable ? data.stable_confidence : data.raw_confidence;
    const confidenceRatio =
      confidence === null || confidence === undefined
        ? 0
        : clamp(Number(confidence), 0, 1);
    els.confidence.textContent =
      confidence === null || confidence === undefined ? "?" : percent(confidence);
    els.confidenceFill.style.width = `${(confidenceRatio * 100).toFixed(1)}%`;
    els.confidenceMeter.setAttribute(
      "aria-valuenow",
      String(Math.round(confidenceRatio * 100))
    );

    const controller = data.controller || {};
    const progress = clamp(Number(controller.commit_progress || 0), 0, 1);
    const circumference = 276.46;
    els.holdRingProgress.style.strokeDashoffset = String(circumference * (1 - progress));
    els.holdRingLabel.textContent = stable ? "Added" : `${Math.round(progress * 100)}%`;

    let statusText;
    if (!loaded) {
      statusText = "Recognition is unavailable. Start the app with a trained model.";
    } else if (cameraError) {
      statusText = friendlyCameraError(cameraError);
    } else if (!cameraRunning) {
      statusText = "Camera is off. Press Start camera";
    } else if (opening) {
      statusText = "Opening the camera?";
    } else if (!hand) {
      statusText = "Show your hand to the camera";
    } else if (!accepted) {
      statusText = "Not sure. Hold your hand steady and fully in view";
    } else if (stable && data.repeat_blocked) {
      statusText = "Lower your hand to add the same letter again";
    } else if (stable) {
      statusText = `Added: ${stable}`;
    } else {
      statusText = "Hold still...";
    }
    els.message.textContent = statusText;

    renderTopPredictions(hand ? data.top_predictions || [] : []);

    const metrics = (data.model_metadata && data.model_metadata.validation_metrics) || {};
    els.modelClasses.textContent = labels.length ? String(labels.length) : "?";
    els.modelThresholds.textContent = data.thresholds
      ? `${numeric(data.thresholds.min_confidence, 2)} / ${numeric(data.thresholds.min_margin, 2)}`
      : "?";
    els.modelF1.textContent =
      metrics.macro_f1 === undefined ? "?" : `${numeric(metrics.macro_f1, 3)} (val)`;
    els.modelSmoothing.textContent = controller.smoothing_window
      ? `${controller.smoothing_window} frames ? ${controller.smoothing_min_votes} votes`
      : "?";
    const kind = training.dataset_kind || "unknown";
    els.modelKind.textContent = kind === "synthetic" ? "synthetic demo" : kind;
    els.modelKind.className = `pill${kind === "synthetic" || kind === "mixed" ? " pill-warn" : ""}`;

    const text = data.text || "";
    if (text !== app.lastText) {
      els.textContent.textContent = text;
      els.textPlaceholder.classList.toggle("hidden", text.length > 0);
      els.textOutput.classList.toggle("is-empty", text.length === 0);
      if (text.length > app.lastText.length && !prefersReducedMotion()) {
        els.textOutput.classList.remove("is-bump");
        void els.textOutput.offsetWidth;
        els.textOutput.classList.add("is-bump");
      }
      app.lastText = text;
      els.textOutput.scrollTop = els.textOutput.scrollHeight;
    }
    const maxLength =
      data.limits && data.limits.max_text_length ? data.limits.max_text_length : 280;
    els.charCount.textContent = `${text.length} / ${maxLength}`;

    app.serverSpeech = data.speech || null;
    applySpeechState(app.serverSpeech);
    updateButtons();
  }

  function renderTopPredictions(items) {
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "empty-row";
      li.textContent = "Waiting for a hand…";
      els.topList.replaceChildren(li);
      return;
    }
    const fragment = document.createDocumentFragment();
    items.slice(0, 5).forEach((item) => {
      const li = document.createElement("li");
      const letter = document.createElement("span");
      letter.className = "letter";
      letter.textContent = item.label;
      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(2, Math.min(100, Number(item.probability) * 100)).toFixed(1)}%`;
      bar.appendChild(fill);
      const pct = document.createElement("span");
      pct.className = "percent";
      pct.textContent = percent(item.probability);
      li.append(letter, bar, pct);
      fragment.appendChild(li);
    });
    els.topList.replaceChildren(fragment);
  }

  function applySpeechState(speech) {
    if (!speech) return;
    const serverReady = speech.available && speech.enabled && speech.state !== "unavailable";
    const ready = Boolean(serverReady || browserTTS);
    els.speechStatus.textContent = ready ? "Voice ready" : "Voice unavailable";
    const state = els.speechStatus.closest(".voice-state");
    if (state) state.classList.toggle("is-unavailable", !ready);
    els.speechStatus.title = serverReady
      ? "The app can read your text aloud."
      : ready
        ? "Your browser will read the text aloud."
        : "No speech voice is available on this device.";

    updateVoiceOptions(speech);

    if (!localStorage.getItem("asl-rate") && speech.rate) {
      app.localRate = Number(speech.rate);
      els.voiceRate.value = String(app.localRate);
      els.voiceRateValue.textContent = String(app.localRate);
    }
    if (!localStorage.getItem("asl-volume") && speech.volume !== undefined) {
      app.localVolume = Number(speech.volume);
      els.voiceVolume.value = String(app.localVolume);
      els.voiceVolumeValue.textContent = `${Math.round(app.localVolume * 100)}%`;
    }
  }

  function updateVoiceOptions(speech) {
    const voices = Array.isArray(speech.voices) ? speech.voices : [];
    const browserVoices = browserTTS ? window.speechSynthesis.getVoices() : [];
    const signature = JSON.stringify([
      voices.map((voice) => voice.id),
      browserVoices.map((voice) => voice.name),
      speech.selected_voice,
    ]);
    if (signature === app.voiceSignature) return;
    app.voiceSignature = signature;

    const selected =
      app.localVoice ||
      (speech.selected_voice ? `s:${speech.selected_voice}` : "");
    const fragment = document.createDocumentFragment();
    if (voices.length) {
      const group = document.createElement("optgroup");
      group.label = "Device voices";
      voices.forEach((voice) => {
        group.appendChild(new Option(voice.name || voice.id, `s:${voice.id}`));
      });
      fragment.appendChild(group);
    }
    if (browserVoices.length) {
      const group = document.createElement("optgroup");
      group.label = "Browser voices";
      browserVoices.forEach((voice) => {
        group.appendChild(new Option(`${voice.name} (${voice.lang})`, `b:${voice.name}`));
      });
      fragment.appendChild(group);
    }
    if (!fragment.childNodes.length) {
      fragment.appendChild(new Option("No voices available", ""));
    }
    els.voiceSelect.replaceChildren(fragment);
    if ([...els.voiceSelect.options].some((option) => option.value === selected)) {
      els.voiceSelect.value = selected;
    } else if (speech.selected_voice) {
      const fallback = `s:${speech.selected_voice}`;
      if ([...els.voiceSelect.options].some((option) => option.value === fallback)) {
        els.voiceSelect.value = fallback;
      }
    }
  }

  function updateButtons() {
    const data = app.data;
    const busy = app.busy;
    const set = (button, disabled, reason) => {
      if (!button) return;
      button.disabled = Boolean(disabled);
      if (disabled) button.title = reason;
      else button.removeAttribute("title");
    };

    if (!data) {
      set(els.start, busy, "Please wait a moment.");
      set(els.stop, true, "The camera is off.");
      set(els.reload, busy, "Please wait a moment.");
      set(els.speak, true, "Add a letter first.");
      set(els.stopSpeak, true, "Nothing is speaking.");
      set(els.space, true, "Add a letter first.");
      set(els.delete, true, "There is nothing to delete.");
      set(els.undo, true, "There is nothing to undo.");
      set(els.clear, true, "There is nothing to clear.");
      set(els.addCurrent, true, "Hold a letter steadily first.");
      set(els.copy, true, "There is no text to copy.");
      set(els.download, true, "There is no text to save.");
      return;
    }

    const text = data.text || "";
    const speech = data.speech || {};
    const running = Boolean(data.camera_running);
    const max = (data.limits && data.limits.max_text_length) || 280;
    const speaking = Boolean(speech.speaking || app.browserSpeaking);

    set(els.start, busy || running, running ? "The camera is already on." : "Please wait a moment.");
    set(els.stop, busy || !running, running ? "Stop the camera." : "The camera is already off.");
    set(els.reload, busy, "Please wait a moment.");
    set(els.speak, busy || !text.trim(), !text.trim() ? "Add a letter first." : "Speaking?");
    set(els.stopSpeak, busy || !speaking, speaking ? "Stop speaking." : "Nothing is speaking.");
    set(
      els.space,
      busy || !text || text.endsWith(" ") || text.length >= max,
      !text
        ? "Add a letter first."
        : text.endsWith(" ")
          ? "There is already a space at the end."
          : text.length >= max
            ? "Your text is full."
            : "Add a space."
    );
    set(els.delete, busy || !text, text ? "Delete the last character." : "There is nothing to delete.");
    set(els.undo, busy || !data.can_undo, data.can_undo ? "Undo the last change." : "There is nothing to undo.");
    set(els.clear, busy || !text, text ? "Clear your text. You can undo afterwards." : "There is nothing to clear.");
    set(
      els.addCurrent,
      busy || !data.stable_label,
      data.stable_label
        ? "Add the letter shown in Detected letter."
        : "Hold a letter steadily until it appears in Detected letter."
    );
    set(els.copy, busy || !text, text ? "Copy your text." : "There is no text to copy.");
    set(els.download, busy || !text, text ? "Save your text as a .txt file." : "There is no text to save.");
  }

  // ------------------------------------------------------------------ controls
  function selectedCameraIndex() {
    const value = Number.parseInt(els.cameraDevice.value, 10);
    return Number.isInteger(value) && value >= 0 && value <= 10 ? value : 0;
  }

  function setCameraOpening() {
    app.cameraWait = true;
    els.videoOverlay.classList.remove("hidden");
    els.videoOverlayText.textContent = "Opening camera…";
    els.videoSkeleton.classList.remove("hidden");
    clearTimeout(app.skeletonTimer);
    app.skeletonTimer = setTimeout(() => els.videoSkeleton.classList.add("hidden"), 4000);
  }

  els.start.addEventListener("click", async () => {
    setCameraOpening();
    const result = await postAction(
      "/api/camera/start",
      { index: selectedCameraIndex() },
      els.start
    );
    if (result) {
      els.video.src = `/video_feed?t=${Date.now()}`;
      refreshDevices();
    } else {
      app.cameraWait = false;
      els.videoOverlay.classList.add("hidden");
      els.videoSkeleton.classList.add("hidden");
    }
  });

  els.stop.addEventListener("click", async () => {
    await postAction("/api/camera/stop", {}, els.stop);
    app.cameraWait = false;
    els.videoOverlay.classList.add("hidden");
    els.videoSkeleton.classList.add("hidden");
  });

  els.reload.addEventListener("click", async () => {
    await postAction("/api/model/reload", {}, els.reload);
  });

  els.addCurrent.addEventListener("click", async () => {
    await postAction("/api/text/add", {}, els.addCurrent);
  });
  els.space.addEventListener("click", async () => {
    await postAction("/api/text/space", {}, els.space);
  });
  els.delete.addEventListener("click", async () => {
    await postAction("/api/text/delete", {}, els.delete);
  });
  els.undo.addEventListener("click", async () => {
    await postAction("/api/text/undo", {}, els.undo);
  });
  els.clear.addEventListener("click", async () => {
    const text = (app.data && app.data.text) || "";
    if (text && !window.confirm("Clear all of your text? You can still use Undo afterwards.")) {
      return;
    }
    await postAction("/api/text/clear", {}, els.clear);
  });

  async function copyText() {
    const text = (app.data && app.data.text) || "";
    if (!text) {
      showToast("There is no text to copy yet.", true);
      return;
    }
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const field = document.createElement("textarea");
        field.value = text;
        field.setAttribute("readonly", "");
        field.style.position = "fixed";
        field.style.opacity = "0";
        document.body.appendChild(field);
        field.select();
        document.execCommand("copy");
        field.remove();
      }
      showToast("Text copied.");
    } catch {
      showToast("Could not copy. Select the text manually.", true);
    }
  }

  async function downloadText() {
    const text = (app.data && app.data.text) || "";
    if (!text) {
      showToast("There is no text to save yet.", true);
      return;
    }
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `asl-fingerspelling-${stamp}.txt`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("Text saved.");
  }

  els.copy.addEventListener("click", copyText);
  els.download.addEventListener("click", downloadText);

  els.speak.addEventListener("click", async () => {
    const text = (app.data && app.data.text) || "";
    if (!text.trim()) {
      showToast("Add a letter before pressing Speak.", true);
      return;
    }
    const speech = (app.data && app.data.speech) || {};
    if (!speech.available || !speech.enabled) {
      if (browserTTS) {
        speakWithBrowser(text);
      } else {
        showToast("No voice is available on this device.", true);
      }
      return;
    }
    if (app.busy) return;
    setBusy(true);
    try {
      await api("/api/speech/speak", { text });
      showToast("Speaking.");
    } catch (error) {
      if (browserTTS) {
        speakWithBrowser(text);
        showToast("Using your browser voice instead.", true);
      } else {
        showToast("Could not speak the text. Check your volume and try again.", true);
      }
    } finally {
      setBusy(false);
      await refresh();
    }
  });

  els.stopSpeak.addEventListener("click", async () => {
    if (browserTTS) window.speechSynthesis.cancel();
    app.browserSpeaking = false;
    try {
      await api("/api/speech/stop", {});
    } catch {
      // Browser fallback can still stop locally.
    }
    updateButtons();
    await refresh();
  });

  function speakWithBrowser(text) {
    if (!browserTTS) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = clamp(app.localRate / 180, 0.5, 2);
    utterance.volume = clamp(app.localVolume, 0, 1);
    if (app.localVoice.startsWith("b:")) {
      const name = app.localVoice.slice(2);
      const voice = window.speechSynthesis.getVoices().find((item) => item.name === name);
      if (voice) utterance.voice = voice;
    }
    utterance.onstart = () => {
      app.browserSpeaking = true;
      els.speechStatus.textContent = "Browser voice speaking…";
      updateButtons();
    };
    utterance.onend = () => {
      app.browserSpeaking = false;
      if (app.data) applySpeechState(app.data.speech || null);
      updateButtons();
    };
    utterance.onerror = (event) => {
      app.browserSpeaking = false;
      if (event.error !== "canceled") showToast("Browser voice could not speak. Try again.", true);
      updateButtons();
    };
    app.browserSpeaking = true;
    window.speechSynthesis.speak(utterance);
  }

  // ------------------------------------------------------------------ settings
  function saveVoiceSetting(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      // Storage can be disabled; the UI still works for this session.
    }
  }

  async function pushVoiceSettings() {
    clearTimeout(app.voiceTimer);
    if (!app.serverSpeech || !app.serverSpeech.enabled) return;
    try {
      const payload = { rate: app.localRate, volume: app.localVolume };
      if (app.localVoice.startsWith("s:")) payload.voice = app.localVoice.slice(2);
      await api("/api/speech/settings", payload);
    } catch {
      // Settings remain local; the browser fallback can still use them.
    }
  }

  function queueVoicePush() {
    clearTimeout(app.voiceTimer);
    app.voiceTimer = setTimeout(pushVoiceSettings, 350);
  }

  els.voiceSelect.addEventListener("change", () => {
    app.localVoice = els.voiceSelect.value;
    saveVoiceSetting("asl-voice", app.localVoice);
    if (app.localVoice.startsWith("s:")) {
      pushVoiceSettings();
      showToast("Server voice selected.");
    } else if (app.localVoice.startsWith("b:")) {
      showToast("Browser voice selected.");
    }
  });

  els.voiceRate.addEventListener("input", () => {
    app.localRate = Number(els.voiceRate.value);
    els.voiceRateValue.textContent = String(app.localRate);
    saveVoiceSetting("asl-rate", String(app.localRate));
    queueVoicePush();
  });

  els.voiceVolume.addEventListener("input", () => {
    app.localVolume = Number(els.voiceVolume.value);
    els.voiceVolumeValue.textContent = `${Math.round(app.localVolume * 100)}%`;
    saveVoiceSetting("asl-volume", String(app.localVolume));
    queueVoicePush();
  });

  if (browserTTS) {
    window.speechSynthesis.addEventListener("voiceschanged", () => {
      if (app.serverSpeech) updateVoiceOptions(app.serverSpeech);
    });
  }

  // ------------------------------------------------------------------ image test
  els.imageUpload.addEventListener("change", async () => {
    const file = els.imageUpload.files && els.imageUpload.files[0];
    if (!file) return;
    if (file.size > 8 * 1024 * 1024) {
      showToast("Image exceeds 8 MB.", true);
      return;
    }
    els.imageResult.classList.remove("hidden");
    els.imageResult.textContent = "Extracting landmarks and predicting…";
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/predict/image", {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok || !data) {
        throw new Error((data && (data.message || data.detail)) || "Image prediction failed.");
      }
      renderImageResult(data);
    } catch (error) {
      els.imageResult.textContent = error.message || "Image prediction failed.";
      showToast(error.message || "Image prediction failed.", true);
    } finally {
      els.imageUpload.value = "";
    }
  });

  function renderImageResult(data) {
    els.imageResult.replaceChildren();
    const heading = document.createElement("strong");
    if (!data.hand_detected) {
      heading.textContent = "No hand found";
      els.imageResult.appendChild(heading);
      const note = document.createElement("span");
      note.textContent = "Try a well-lit photo with one whole hand in view.";
      els.imageResult.appendChild(note);
      return;
    }
    heading.textContent = data.label ? `Looks like ${data.label}` : "Letter unclear";
    els.imageResult.appendChild(heading);
    const note = document.createElement("span");
    if (data.accepted) {
      note.textContent = ` ${percent(data.confidence)} confident.`;
    } else {
      note.textContent = " Not sure about this photo. Try better lighting and keep the hand fully in view.";
    }
    els.imageResult.appendChild(note);
    if (data.model_is_synthetic) {
      const warning = document.createElement("p");
      warning.className = "fine-print";
      warning.textContent = "Demo mode: this is for demonstration only, not reliable ASL recognition.";
      els.imageResult.appendChild(warning);
    }
  }

  // ------------------------------------------------------------------ theme
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const next = theme === "light" ? "dark" : "light";
    els.themeToggle.setAttribute("aria-label", `Switch to ${next} theme`);
    els.themeToggle.title = `Switch to ${next} theme`;
  }

  els.themeToggle.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme === "light" ? "light" : "dark";
    const next = current === "light" ? "dark" : "light";
    applyTheme(next);
    saveVoiceSetting("asl-theme", next);
  });

  // ------------------------------------------------------------------ help
  els.shortcutHelp.addEventListener("click", () => {
    if (typeof els.shortcutsDialog.showModal === "function") {
      els.shortcutsDialog.showModal();
    } else {
      els.shortcutsDialog.setAttribute("open", "");
    }
  });

  function closeShortcuts() {
    if (typeof els.shortcutsDialog.close === "function") {
      els.shortcutsDialog.close();
    } else {
      els.shortcutsDialog.removeAttribute("open");
    }
  }

  els.closeShortcuts.addEventListener("click", closeShortcuts);
  els.shortcutsDialog.addEventListener("click", (event) => {
    if (event.target === els.shortcutsDialog) closeShortcuts();
  });

  function openAbout() {
    if (typeof els.aboutDialog.showModal === "function") {
      els.aboutDialog.showModal();
    } else {
      els.aboutDialog.setAttribute("open", "");
    }
  }

  function closeAbout() {
    if (typeof els.aboutDialog.close === "function") {
      els.aboutDialog.close();
    } else {
      els.aboutDialog.removeAttribute("open");
    }
  }

  els.aboutButton.addEventListener("click", openAbout);
  els.demoLearnMore.addEventListener("click", openAbout);
  els.closeAbout.addEventListener("click", closeAbout);
  els.aboutDialog.addEventListener("click", (event) => {
    if (event.target === els.aboutDialog) closeAbout();
  });

  els.alphabetGrid.addEventListener("click", (event) => {
    const button = event.target.closest("[data-letter]");
    if (!button) return;
    els.alphabetGrid.querySelectorAll("[data-letter]").forEach((item) => {
      item.setAttribute("aria-pressed", item === button ? "true" : "false");
    });
    const letter = button.dataset.letter;
    els.alphabetNote.textContent = `${letter}: ${ALPHABET_NOTES[letter] || "Static handshape."}`;
  });

  // ------------------------------------------------------------------ keyboard
  function isTypingTarget(target) {
    if (!target) return false;
    const tag = target.tagName;
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      target.isContentEditable
    );
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (els.shortcutsDialog.open) {
        closeShortcuts();
        return;
      }
      if (els.aboutDialog.open) {
        closeAbout();
        return;
      }
      if (browserTTS) window.speechSynthesis.cancel();
      app.browserSpeaking = false;
      if (app.data && app.data.speech && app.data.speech.speaking) {
        api("/api/speech/stop", {}).catch(() => {});
      }
      updateButtons();
      return;
    }
    if (els.shortcutsDialog.open) return;
    if (isTypingTarget(event.target)) return;
    if (event.target && event.target.tagName === "BUTTON") return;

    const key = event.key;
    if (key === " " || key === "Spacebar") {
      event.preventDefault();
      if (!els.space.disabled) els.space.click();
    } else if (key === "Backspace") {
      event.preventDefault();
      if (!els.delete.disabled) els.delete.click();
    } else if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "z" && !event.shiftKey) {
      event.preventDefault();
      if (!els.undo.disabled) els.undo.click();
    } else if (key === "Enter") {
      event.preventDefault();
      if (!els.speak.disabled) els.speak.click();
    }
  });

  // ------------------------------------------------------------------ connection
  els.reconnectServer.addEventListener("click", () => {
    app.reconnectAttempts = 0;
    clearTimeout(app.reconnectTimer);
    connectEvents(true);
    refresh();
    showToast("Reconnecting…");
  });

  els.video.addEventListener("error", () => {
    if (app.videoRetry >= 3) return;
    app.videoRetry += 1;
    setTimeout(() => {
      els.video.src = `/video_feed?t=${Date.now()}`;
    }, 1200 * app.videoRetry);
  });

  els.video.addEventListener("load", () => {
    app.videoRetry = 0;
  });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      if (!app.eventSource) connectEvents(true);
    }
  });

  // ------------------------------------------------------------------ first run
  function initOnboarding() {
    let complete = false;
    try {
      complete = localStorage.getItem(ONBOARDING_KEY) === "1";
    } catch {
      complete = false;
    }
    if (!complete) els.onboarding.classList.remove("hidden");
    els.onboardingDismiss.addEventListener("click", () => {
      els.onboarding.classList.add("hidden");
      try {
        localStorage.setItem(ONBOARDING_KEY, "1");
      } catch {
        // Storage may be unavailable; hiding the card is enough for this session.
      }
    });
  }

  function initDeveloperMode() {
    if (DEV_MODE) {
      els.developerDetails.classList.remove("hidden");
      document.body.dataset.dev = "true";
    }
  }

  document.addEventListener("click", (event) => {
    if (els.moreMenu && !els.moreMenu.contains(event.target)) els.moreMenu.open = false;
    const voicePanel = $("voice-panel");
    if (voicePanel && !voicePanel.contains(event.target)) voicePanel.open = false;
  });

  // ------------------------------------------------------------------ boot
  initOnboarding();
  initDeveloperMode();
  applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  els.voiceRate.value = String(app.localRate);
  els.voiceRateValue.textContent = String(app.localRate);
  els.voiceVolume.value = String(app.localVolume);
  els.voiceVolumeValue.textContent = `${Math.round(app.localVolume * 100)}%`;
  els.video.src = `/video_feed?t=${Date.now()}`;
  updateButtons();
  refreshDevices();
  connectEvents(true);
  refresh();
  setTimeout(() => els.videoSkeleton.classList.add("hidden"), 2500);
})();
