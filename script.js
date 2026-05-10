const themeStorageKey = 'neurodetect-theme';
const resultStorageKey = 'neurodetect-last-result';
const resultArchiveStorageKey = 'neurodetect-last-result-archive';
const API_BASE_URL = (localStorage.getItem('neurodetect-api-base') || 'http://127.0.0.1:8001').replace(/\/$/, '');

function normalizeClassLabel(label) {
  const map = {
    NonDemented: 'NonDemented',
    VeryMild: 'VeryMild',
    VeryMildDemented: 'VeryMild',
    Mild: 'Mild',
    MildDemented: 'Mild',
    Moderate: 'Moderate',
    ModerateDemented: 'Moderate',
  };
  return map[label] || label;
}

function formatDiagnosisLabel(label) {
  const normalized = normalizeClassLabel(label);
  const display = {
    NonDemented: 'Non Demented',
    VeryMild: 'Very Mild Demented',
    Mild: 'Mild Demented',
    Moderate: 'Moderate Demented',
  };
  return display[normalized] || (label || 'Unknown');
}

function setTheme(theme) {
  const isDark = theme === 'dark';
  document.body.classList.toggle('dark-mode', isDark);
  document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
    button.setAttribute('aria-label', isDark ? 'Light mode' : 'Dark mode');
    button.innerHTML = isDark
      ? '<span aria-hidden="true">☀</span><span class="toggle-text">Light</span>'
      : '<span aria-hidden="true">◐</span><span class="toggle-text">Dark</span>';
  });
  localStorage.setItem(themeStorageKey, theme);
}

function initializeTheme() {
  const savedTheme = localStorage.getItem(themeStorageKey);
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(savedTheme || (prefersDark ? 'dark' : 'light'));
}

function toggleTheme() {
  setTheme(document.body.classList.contains('dark-mode') ? 'light' : 'dark');
}

function createBrainSVG() {
  return `
    <svg viewBox="0 0 240 240" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M108 28c-26 0-48 19-52 44-16 3-28 17-28 34 0 14 8 27 20 33-2 5-4 10-4 16 0 18 15 33 33 33 5 0 10-1 14-4 8 10 20 16 34 16 13 0 25-5 34-14 6 4 12 6 20 6 20 0 36-16 36-36 0-7-2-13-5-18 12-6 20-18 20-32 0-17-12-31-27-34-2-25-24-44-50-44-12 0-23 4-31 11-9-7-20-11-34-11Z" fill="#eaf4ff" stroke="#2e86c1" stroke-width="4"/>
      <path d="M88 62c-13 0-24 11-24 24 0 8 4 15 10 19m10-43c-6 9-7 20-2 30m51-48c-9 7-13 17-12 29m41-14c-5 7-7 15-7 24m36-9c-7 6-11 15-11 24" stroke="#2e86c1" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
      <path d="M64 128c10-7 21-10 34-10m14 0c13 0 24 3 34 10m-78 25c10-6 21-9 34-9m14 0c13 0 24 3 34 9" stroke="#0a1628" stroke-width="6" stroke-linecap="round" opacity="0.2"/>
      <circle cx="120" cy="118" r="56" stroke="#2e86c1" stroke-width="2" stroke-dasharray="6 10" opacity="0.22"/>
    </svg>`;
}

function setupBrandIcons() {
  document.querySelectorAll('[data-brain-icon]').forEach((target) => {
    target.innerHTML = createBrainSVG();
  });
}

function setupNavbar() {
  const navbar = document.querySelector('.navbar');
  const hamburger = document.querySelector('[data-menu-toggle]');
  const mobileNav = document.querySelector('[data-mobile-nav]');

  if (navbar) {
    const onScroll = () => navbar.classList.toggle('scrolled', window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  if (hamburger && mobileNav) {
    hamburger.addEventListener('click', () => {
      const open = mobileNav.classList.toggle('open');
      hamburger.setAttribute('aria-expanded', String(open));
    });
  }

  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', (event) => {
      const targetId = anchor.getAttribute('href').slice(1);
      const target = document.getElementById(targetId);
      if (target) {
        event.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        mobileNav?.classList.remove('open');
        hamburger?.setAttribute('aria-expanded', 'false');
      }
    });
  });
}

function setupReveal() {
  const elements = document.querySelectorAll('.reveal');
  if (!elements.length) return;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.16 });
  elements.forEach((element) => observer.observe(element));
}

function setupUploadPage() {
  const zone = document.querySelector('[data-upload-zone]');
  const input = document.querySelector('[data-upload-input]');
  const preview = document.querySelector('[data-upload-preview]');
  const placeholder = document.querySelector('[data-upload-placeholder]');
  const previewImage = document.querySelector('[data-preview-image]');
  const previewName = document.querySelector('[data-preview-name]');
  const previewSize = document.querySelector('[data-preview-size]');
  const analyzeButton = document.querySelector('[data-analyze-button]');
  const overlay = document.querySelector('[data-scan-overlay]');
  const progressFill = document.querySelector('[data-progress-fill]');
  const progressText = document.querySelector('[data-progress-text]');
  const stepNodes = Array.from(document.querySelectorAll('[data-scan-step]'));
  let selectedFile = null;

  if (!zone || !input) return;

  const showFile = (file) => {
    if (!file) return;
    selectedFile = file;
    placeholder?.classList.add('hidden');
    preview?.classList.remove('hidden');
    previewName.textContent = file.name;
    previewSize.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    if (file.type.startsWith('image/') && previewImage) {
      previewImage.src = URL.createObjectURL(file);
      previewImage.alt = file.name;
    } else if (previewImage) {
      previewImage.src = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160"><rect width="160" height="160" rx="24" fill="#eaf2fb"/><path d="M44 90c0-18 15-33 36-33s36 15 36 33-15 33-36 33-36-15-36-33Z" fill="#cfe2f3" stroke="#2e86c1" stroke-width="4"/><path d="M67 58c8-9 19-14 33-14 22 0 40 15 40 34 0 12-7 23-18 29" fill="none" stroke="#0a1628" stroke-opacity=".2" stroke-width="4" stroke-linecap="round"/></svg>`)}`;
      previewImage.alt = 'MRI preview placeholder';
    }
  };

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (event) => {
    event.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (event) => {
    event.preventDefault();
    zone.classList.remove('dragover');
    const file = event.dataTransfer.files?.[0];
    if (file) {
      input.files = event.dataTransfer.files;
      showFile(file);
    }
  });
  input.addEventListener('change', () => {
    if (input.files?.[0]) showFile(input.files[0]);
  });

  const toDataUrl = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Unable to read image file.'));
    reader.readAsDataURL(file);
  });

  const setScanStep = (stepIndex, progress, text) => {
    progressFill.style.width = `${progress}%`;
    progressText.textContent = text;
    stepNodes.forEach((node, index) => {
      node.classList.toggle('done', index < stepIndex);
    });
  };

  analyzeButton?.addEventListener('click', async () => {
    if (!selectedFile) {
      alert('Please upload an MRI image first.');
      return;
    }

    if (!overlay) return;
    overlay.classList.add('open');
    setScanStep(0, 8, 'Loading model...');

    try {
      const payload = new FormData();
      payload.append('file', selectedFile);

      setScanStep(1, 28, 'Preprocessing image...');
      const response = await fetch(`${API_BASE_URL}/predict/mri`, {
        method: 'POST',
        body: payload,
      });

      setScanStep(2, 72, 'Running analysis...');

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data?.detail || 'Prediction request failed.';
        throw new Error(detail);
      }

      setScanStep(3, 92, 'Generating report...');

      const patientName = document.querySelector('#pname')?.value?.trim() || 'Unknown Patient';
      const ageValue = Number(document.querySelector('#age')?.value || 0);
      const genderValue = document.querySelector('#gender')?.value || 'Unknown';
      const scanDateValue = document.querySelector('#scanDate')?.value || new Date().toISOString().slice(0, 10);
      const notesValue = document.querySelector('#notes')?.value?.trim() || '';

      const originalImage = await toDataUrl(selectedFile);
      const heatmapImage = data.gradcam_image ? `data:image/png;base64,${data.gradcam_image}` : originalImage;
      const patientId = `PT-${Date.now()}`;

      const normalizedClass = normalizeClassLabel(data.predicted_class);
      const normalizedProbabilities = {};
      Object.entries(data.probabilities || {}).forEach(([key, value]) => {
        normalizedProbabilities[normalizeClassLabel(key)] = value;
      });
      const normalizedPrediction = {
        ...data,
        predicted_class: normalizedClass,
        probabilities: normalizedProbabilities,
      };

      const resultPayload = {
        patientId,
        patient: {
          name: patientName,
          age: ageValue > 0 ? ageValue : null,
          gender: genderValue,
          scanDate: scanDateValue,
          notes: notesValue,
        },
        prediction: data,
        images: {
          original: originalImage,
          heatmap: heatmapImage,
        },
        createdAt: new Date().toISOString(),
      };

      resultPayload.prediction = normalizedPrediction;

      // Persist to backend so results can be fetched even if browser storage is cleared.
      fetch(`${API_BASE_URL}/patients`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          name: resultPayload.patient.name,
          age: resultPayload.patient.age,
          gender: resultPayload.patient.gender,
          diagnosis: normalizedPrediction.predicted_class,
          prediction: normalizedPrediction,
        }),
      }).catch(() => null);

      sessionStorage.setItem(resultStorageKey, JSON.stringify(resultPayload));
      localStorage.setItem(resultArchiveStorageKey, JSON.stringify(resultPayload));
      setScanStep(4, 100, 'Done');
      window.setTimeout(() => { window.location.href = `results.html?patient_id=${encodeURIComponent(patientId)}`; }, 400);
    } catch (error) {
      overlay.classList.remove('open');
      const message = error instanceof Error ? error.message : 'Unknown error while running prediction.';
      alert(`Failed to analyze MRI: ${message}`);
    }
  });
}

function setupResultsPage() {
  const readStoredResult = () => {
    const candidates = [
      sessionStorage.getItem(resultStorageKey),
      localStorage.getItem(resultArchiveStorageKey),
    ];

    for (const item of candidates) {
      try {
        if (item) return JSON.parse(item);
      } catch {
        continue;
      }
    }

    return null;
  };

  const patientIdFromQuery = new URLSearchParams(window.location.search).get('patient_id');

  const renderResult = (result) => {
    const stageMap = {
      NonDemented: 'NonDemented',
      VeryMild: 'VeryMild',
      Mild: 'Mild',
      Moderate: 'Moderate',
    };

    if (result?.patient) {
      const { patient } = result;
      const nameNode = document.querySelector('[data-result-patient-name]');
      const ageNode = document.querySelector('[data-result-patient-age]');
      const genderNode = document.querySelector('[data-result-patient-gender]');
      const dateNode = document.querySelector('[data-result-scan-date]');
      const avatar = document.querySelector('.avatar-circle');

      if (nameNode) nameNode.textContent = patient.name || 'Unknown Patient';
      if (ageNode) ageNode.textContent = patient.age ? `Age ${patient.age}` : 'Age N/A';
      if (genderNode) genderNode.textContent = patient.gender || 'Unknown';
      if (dateNode) dateNode.textContent = `Scan Date: ${patient.scanDate || new Date().toISOString().slice(0, 10)}`;
      if (avatar && patient.name) {
        const initials = patient.name
          .split(' ')
          .filter(Boolean)
          .slice(0, 2)
          .map((part) => part[0]?.toUpperCase())
          .join('');
        if (initials) avatar.textContent = initials;
      }
    }

    if (result?.prediction) {
      const prediction = result.prediction;
      const diagnosisNode = document.querySelector('[data-result-diagnosis]');
      const confidenceNode = document.querySelector('[data-result-confidence]');
      const confidenceShortNode = document.querySelector('[data-result-confidence-short]');
      const recommendationNode = document.querySelector('[data-result-recommendation]');
      const riskPill = document.querySelector('[data-result-risk-pill]');

      const predictedClass = normalizeClassLabel(prediction.predicted_class);
      const confidencePct = Math.round((prediction.confidence || 0) * 1000) / 10;
      if (diagnosisNode) diagnosisNode.textContent = formatDiagnosisLabel(predictedClass);
      if (confidenceNode) confidenceNode.textContent = `${confidencePct.toFixed(1)}%`;
      if (confidenceShortNode) confidenceShortNode.textContent = `${Math.round(confidencePct)}%`;
      if (recommendationNode) recommendationNode.textContent = prediction.recommendation || 'No recommendation available.';

      if (riskPill) {
        const classToRisk = {
          NonDemented: ['risk-green', 'Low Risk'],
          VeryMild: ['risk-yellow', 'Very Mild Risk'],
          Mild: ['risk-orange', 'Mild Risk'],
          Moderate: ['risk-red', 'High Risk'],
        };
        const [riskClass, riskLabel] = classToRisk[predictedClass] || ['risk-yellow', 'Risk'];
        riskPill.classList.remove('risk-green', 'risk-yellow', 'risk-orange', 'risk-red');
        riskPill.classList.add(riskClass);
        riskPill.textContent = riskLabel;
      }

      const probabilities = prediction.probabilities || {};
      const normalizedProbabilities = {};
      Object.entries(probabilities).forEach(([key, value]) => {
        normalizedProbabilities[normalizeClassLabel(key)] = value;
      });

      document.querySelectorAll('.progress-fill-dark[data-stage]').forEach((bar) => {
        const stage = bar.dataset.stage;
        const backendKey = stageMap[stage] || stage;
        const value = Math.round(((normalizedProbabilities[backendKey] || 0) * 1000)) / 10;
        const percent = Math.max(0, Math.min(100, value));
        bar.style.width = '0%';
        bar.dataset.value = String(percent);
        requestAnimationFrame(() => { bar.style.width = `${percent}%`; });
        const labelNode = document.querySelector(`[data-stage-percent="${stage}"]`);
        if (labelNode) labelNode.textContent = `${percent.toFixed(1)}%`;
      });
    }

    if (result?.images) {
      const originalNode = document.querySelector('[data-result-original]');
      const heatmapNode = document.querySelector('[data-result-heatmap]');
      if (originalNode && result.images.original) originalNode.src = result.images.original;
      if (heatmapNode && result.images.heatmap) heatmapNode.src = result.images.heatmap;
    }
  };

  const animateFallbackBars = () => {
    document.querySelectorAll('.progress-fill-dark').forEach((bar) => {
      const value = Number(bar.dataset.value || '0');
      bar.style.width = '0%';
      requestAnimationFrame(() => { bar.style.width = `${value}%`; });
    });
  };

  const showNoDataMessage = () => {
    const recommendationNode = document.querySelector('[data-result-recommendation]');
    if (recommendationNode) {
      recommendationNode.textContent = 'No recent backend prediction found. Please run a new scan from the Upload page.';
    }
  };

  const storedResult = readStoredResult();
  if (storedResult?.prediction) {
    renderResult(storedResult);
    return;
  }

  if (!patientIdFromQuery) {
    animateFallbackBars();
    showNoDataMessage();
    return;
  }

  fetch(`${API_BASE_URL}/patients/${encodeURIComponent(patientIdFromQuery)}`)
    .then(async (response) => {
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(body?.detail || 'Could not load patient result.');
      }

      const latest = body?.history?.[body.history.length - 1];
      if (!latest?.prediction) {
        throw new Error('No backend prediction available for this patient.');
      }

      const merged = {
        patientId: body.patient_id,
        patient: {
          name: latest.name || 'Unknown Patient',
          age: latest.age,
          gender: latest.gender,
          scanDate: latest.created_at ? String(latest.created_at).slice(0, 10) : new Date().toISOString().slice(0, 10),
        },
        prediction: latest.prediction,
        images: {},
        createdAt: latest.created_at || new Date().toISOString(),
      };

      sessionStorage.setItem(resultStorageKey, JSON.stringify(merged));
      localStorage.setItem(resultArchiveStorageKey, JSON.stringify(merged));
      renderResult(merged);
    })
    .catch(() => {
      animateFallbackBars();
      showNoDataMessage();
    });
}

function setupDashboardPage() {
  const rows = Array.from(document.querySelectorAll('[data-patient-row]'));
  const filterButtons = Array.from(document.querySelectorAll('[data-filter-risk]'));
  const search = document.querySelector('[data-patient-search]');
  const pages = Array.from(document.querySelectorAll('[data-page]'));
  const donutCanvas = document.querySelector('[data-donut-canvas]');
  const barCanvas = document.querySelector('[data-bar-canvas]');
  const tooltip = document.querySelector('[data-tooltip]');

  const applyFilters = () => {
    const query = search?.value.toLowerCase().trim() || '';
    const activeFilter = document.querySelector('[data-filter-risk].active')?.dataset.filterRisk || 'all';
    rows.forEach((row) => {
      const text = row.textContent.toLowerCase();
      const matchesSearch = !query || text.includes(query);
      const matchesFilter = activeFilter === 'all' || row.dataset.risk === activeFilter;
      row.style.display = matchesSearch && matchesFilter ? '' : 'none';
    });
  };

  filterButtons.forEach((button) => {
    button.addEventListener('click', () => {
      filterButtons.forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      applyFilters();
    });
  });
  search?.addEventListener('input', applyFilters);
  pages.forEach((page) => page.addEventListener('click', () => { pages.forEach((item) => item.classList.remove('active')); page.classList.add('active'); }));

  function drawDonut() {
    if (!donutCanvas) return;
    const ctx = donutCanvas.getContext('2d');
    const ratio = window.devicePixelRatio || 1;
    const size = donutCanvas.getBoundingClientRect().width;
    donutCanvas.width = size * ratio;
    donutCanvas.height = size * ratio;
    ctx.scale(ratio, ratio);
    const cx = size / 2;
    const cy = size / 2;
    const outer = size * 0.42;
    const inner = size * 0.24;
    const data = [
      { value: 45, color: '#21a366' },
      { value: 30, color: '#e3a008' },
      { value: 18, color: '#e05252' },
      { value: 7, color: '#2e86c1' },
    ];
    let start = -Math.PI / 2;
    ctx.clearRect(0, 0, size, size);
    data.forEach((segment) => {
      const angle = (segment.value / 100) * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(cx, cy, outer, start, start + angle);
      ctx.arc(cx, cy, inner, start + angle, start, true);
      ctx.closePath();
      ctx.fillStyle = segment.color;
      ctx.fill();
      start += angle;
    });
    ctx.beginPath();
    ctx.arc(cx, cy, inner - 3, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  }

  function drawBars() {
    if (!barCanvas) return;
    const ctx = barCanvas.getContext('2d');
    const ratio = window.devicePixelRatio || 1;
    const rect = barCanvas.getBoundingClientRect();
    barCanvas.width = rect.width * ratio;
    barCanvas.height = rect.height * ratio;
    ctx.scale(ratio, ratio);
    const data = [12, 16, 14, 20, 18, 22, 19];
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const width = rect.width;
    const height = rect.height;
    const padding = 24;
    const chartHeight = height - 50;
    const barWidth = (width - padding * 2) / data.length - 14;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(46, 134, 193, 0.1)';
    for (let i = 0; i < 4; i += 1) {
      const y = padding + (chartHeight / 4) * i;
      ctx.fillRect(padding, y, width - padding * 2, 1);
    }
    data.forEach((value, index) => {
      const x = padding + index * ((width - padding * 2) / data.length) + 7;
      const barHeight = (value / 25) * chartHeight;
      const y = chartHeight - barHeight + 10;
      ctx.fillStyle = '#2e86c1';
      roundRect(ctx, x, y, barWidth, barHeight, 12);
      ctx.fill();
      ctx.fillStyle = '#607086';
      ctx.font = '12px DM Sans';
      ctx.textAlign = 'center';
      ctx.fillText(days[index], x + barWidth / 2, height - 12);
    });
  }

  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
  }

  function showTooltip(text, x, y) {
    if (!tooltip) return;
    tooltip.textContent = text;
    tooltip.style.left = `${x + 12}px`;
    tooltip.style.top = `${y - 8}px`;
    tooltip.classList.add('show');
  }
  function hideTooltip() { tooltip?.classList.remove('show'); }

  drawDonut();
  drawBars();
  window.addEventListener('resize', () => { drawDonut(); drawBars(); });
  if (barCanvas && tooltip) {
    const data = [12, 16, 14, 20, 18, 22, 19];
    barCanvas.addEventListener('mousemove', (event) => {
      const rect = barCanvas.getBoundingClientRect();
      const index = Math.floor(((event.clientX - rect.left) / rect.width) * data.length);
      if (index >= 0 && index < data.length) showTooltip(`${data[index]} scans`, event.offsetX, event.offsetY);
    });
    barCanvas.addEventListener('mouseleave', hideTooltip);
  }
  rows.forEach((row) => row.addEventListener('click', () => { window.location.href = 'results.html'; }));
}

function setupRiskForm() {
  const form = document.querySelector('[data-risk-form]');
  const panes = Array.from(document.querySelectorAll('[data-step-pane]'));
  const indicators = Array.from(document.querySelectorAll('[data-stepper-item]'));
  const nextButtons = Array.from(document.querySelectorAll('[data-next-step]'));
  const backButtons = Array.from(document.querySelectorAll('[data-back-step]'));
  const ratings = Array.from(document.querySelectorAll('[data-rating-question]'));
  const scoreText = document.querySelector('[data-risk-score]');
  const scoreFill = document.querySelector('[data-score-fill]');
  const finalScore = document.querySelector('[data-final-score]');
  const gauge = document.querySelector('[data-gauge]');
  const resultTitle = document.querySelector('[data-result-title]');
  const resultText = document.querySelector('[data-result-text]');
  const breakdown = document.querySelector('[data-factor-breakdown]');
  let currentStep = 0;
  const answers = new Map();
  if (!form || !panes.length) return;

  const updateScore = () => {
    const score = ratings.reduce((total, row) => total + Number(answers.get(row.dataset.question) || 0), 0);
    const percent = Math.round((score / 25) * 100);
    scoreText.textContent = `Cognitive Risk Score: ${score}/25`;
    scoreFill.style.width = `${percent}%`;
    return score;
  };

  const renderStep = (step, direction = 'forward') => {
    panes.forEach((pane, index) => {
      pane.classList.toggle('active', index === step);
      pane.classList.remove('enter-forward', 'enter-back');
    });
    indicators.forEach((item, index) => {
      item.classList.toggle('active', index === step);
      item.classList.toggle('done', index < step);
    });
    panes[step].classList.add(direction === 'back' ? 'enter-back' : 'enter-forward');
  };

  const setChoice = (group, value, button) => {
    answers.set(group, value);
    const row = button.closest('[data-rating-question]');
    row.querySelectorAll('button').forEach((item) => item.classList.remove('selected'));
    button.classList.add('selected');
    updateScore();
  };

  ratings.forEach((row) => {
    row.querySelectorAll('button').forEach((button) => button.addEventListener('click', () => setChoice(row.dataset.question, Number(button.dataset.value), button)));
  });

  document.querySelectorAll('.pill-option input[type="radio"]').forEach((radio) => {
    const updateState = () => {
      const label = radio.closest('.pill-option');
      const group = radio.name;
      const text = radio.parentElement?.textContent?.trim() || '';
      radio.closest('.pill-group')?.querySelectorAll('.pill-option').forEach((option) => option.classList.remove('selected'));
      if (radio.checked && label) label.classList.add('selected');
      if (['family', 'depression', 'diabetes', 'bloodPressure', 'headInjury', 'smoker'].includes(group)) {
        answers.set(group, text === 'Yes' ? 1 : 0);
        updateScore();
      }
    };
    radio.addEventListener('change', updateState);
    if (radio.checked) updateState();
  });

  const validateStep = (step) => {
    const required = panes[step].querySelectorAll('[required]');
    for (const field of required) {
      if (!field.value) { field.focus(); return false; }
    }
    return true;
  };

  const updateResults = () => {
    const score = updateScore();
    const percent = Math.round((score / 25) * 100);
    finalScore.textContent = `${score}/25`;
    gauge.style.background = `conic-gradient(var(--blue) 0deg ${percent * 3.6}deg, #dce4ee ${percent * 3.6}deg 360deg)`;
    let title = 'Low Risk';
    let text = 'Your responses suggest low risk. Maintain healthy lifestyle.';
    if (score >= 9 && score <= 16) { title = 'Moderate Risk'; text = 'Some risk factors detected. Consider consulting a doctor.'; }
    else if (score >= 17) { title = 'High Risk'; text = 'Multiple risk factors present. We strongly recommend neurological evaluation.'; }
    resultTitle.textContent = title;
    resultText.textContent = text;
    breakdown.innerHTML = `
      <div class="table-row table-header" style="grid-template-columns: 1fr 0.7fr 1fr;"><div>Factor</div><div>Score</div><div>Impact</div></div>
      <div class="table-row" style="grid-template-columns: 1fr 0.7fr 1fr;"><div>Family history</div><div>${answers.get('family') || 0}</div><div>Moderate</div></div>
      <div class="table-row" style="grid-template-columns: 1fr 0.7fr 1fr;"><div>Medical history</div><div>${(answers.get('depression') || 0) + (answers.get('diabetes') || 0) + (answers.get('bloodPressure') || 0)}</div><div>Elevated</div></div>
      <div class="table-row" style="grid-template-columns: 1fr 0.7fr 1fr;"><div>Cognitive responses</div><div>${score}</div><div>Primary driver</div></div>`;
  };

  nextButtons.forEach((button) => button.addEventListener('click', () => {
    if (!validateStep(currentStep)) return;
    currentStep = Math.min(currentStep + 1, panes.length - 1);
    renderStep(currentStep, 'forward');
    if (currentStep === panes.length - 1) updateResults();
  }));

  backButtons.forEach((button) => button.addEventListener('click', () => {
    currentStep = Math.max(currentStep - 1, 0);
    renderStep(currentStep, 'back');
  }));

  form.addEventListener('submit', (event) => { event.preventDefault(); updateResults(); currentStep = 3; renderStep(currentStep, 'forward'); });
  renderStep(currentStep, 'forward');
  updateScore();
}

function init() {
  initializeTheme();
  setupBrandIcons();
  setupNavbar();
  setupReveal();
  setupUploadPage();
  setupResultsPage();
  setupDashboardPage();
  setupRiskForm();
  document.querySelectorAll('[data-theme-toggle]').forEach((button) => button.addEventListener('click', toggleTheme));
}

document.addEventListener('DOMContentLoaded', init);
