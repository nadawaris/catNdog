/**
 * CatVision AI - Client-Side Interactive Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const sampleGallery = document.getElementById('sampleGallery');
  const emptyState = document.getElementById('emptyState');
  const loadingState = document.getElementById('loadingState');
  const resultContent = document.getElementById('resultContent');

  // Result Elements
  const previewImage = document.getElementById('previewImage');
  const badgeEmoji = document.getElementById('badgeEmoji');
  const badgeText = document.getElementById('badgeText');
  const verdictTitle = document.getElementById('verdictTitle');
  const confidenceValue = document.getElementById('confidenceValue');
  const catPercentText = document.getElementById('catPercentText');
  const catProgressBar = document.getElementById('catProgressBar');
  const dogPercentText = document.getElementById('dogPercentText');
  const dogProgressBar = document.getElementById('dogProgressBar');
  const rawSigmoid = document.getElementById('rawSigmoid');

  // Tab Switching
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(tc => tc.classList.remove('active'));

      btn.classList.add('active');
      const content = document.getElementById(targetTab);
      if (content) content.classList.add('active');
    });
  });

  // Load Sample Images
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      if (data.samples && data.samples.length > 0) {
        sampleGallery.innerHTML = '';
        data.samples.forEach(sample => {
          const item = document.createElement('div');
          item.className = 'sample-item';
          item.title = `Click to test ${sample.name}`;
          item.innerHTML = `
            <img src="${sample.url}" alt="${sample.name}" loading="lazy" />
            <span class="sample-tag">${sample.expected}</span>
          `;
          item.addEventListener('click', () => {
            predictSample(sample.name);
          });
          sampleGallery.appendChild(item);
        });
      } else {
        sampleGallery.innerHTML = '<p style="color:var(--text-dim);font-size:0.8rem;grid-column:span 4;">No sample images found</p>';
      }
    } catch (e) {
      console.error('Failed to fetch samples:', e);
      sampleGallery.innerHTML = '<p style="color:var(--text-dim);font-size:0.8rem;grid-column:span 4;">Could not load samples</p>';
    }
  }

  // Drag and Drop Handling
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Upload and Predict File
  async function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPEG/PNG/WEBP).');
      return;
    }

    showLoading();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok && data.success) {
        displayResult(data);
      } else {
        alert(data.error || 'Failed to predict image.');
        hideLoading();
      }
    } catch (err) {
      console.error(err);
      alert('Error communicating with backend server.');
      hideLoading();
    }
  }

  // Predict by Sample Name
  async function predictSample(sampleName) {
    showLoading();
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_name: sampleName })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        displayResult(data);
      } else {
        alert(data.error || 'Prediction failed.');
        hideLoading();
      }
    } catch (err) {
      console.error(err);
      alert('Error running sample prediction.');
      hideLoading();
    }
  }

  function showLoading() {
    emptyState.classList.add('hidden');
    resultContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
  }

  function hideLoading() {
    loadingState.classList.add('hidden');
    emptyState.classList.remove('hidden');
  }

  function displayResult(data) {
    loadingState.classList.add('hidden');
    emptyState.classList.add('hidden');
    resultContent.classList.remove('hidden');

    // Update Image Preview
    previewImage.src = data.preview_base64;

    // Badge and Verdict Title
    badgeEmoji.textContent = data.emoji;
    badgeText.textContent = data.label;
    verdictTitle.textContent = `${data.label} ${data.emoji}`;
    confidenceValue.textContent = `${data.confidence}%`;

    // Probability Bars
    catPercentText.textContent = `${data.cat_percentage}%`;
    dogPercentText.textContent = `${data.dog_percentage}%`;

    // Smooth animation on progress bars
    setTimeout(() => {
      catProgressBar.style.width = `${data.cat_percentage}%`;
      dogProgressBar.style.width = `${data.dog_percentage}%`;
    }, 50);

    // Tensor & Debug Details
    rawSigmoid.textContent = data.raw_probability.toFixed(5);
    
    if (data.debug) {
      document.getElementById('debugModelFile').textContent = data.debug.model_file || 'models/cat_dog_cnn.keras';
      document.getElementById('debugInputSize').textContent = data.debug.input_size || '128 × 128 × 3 (RGB)';
      document.getElementById('debugClassMapping').textContent = data.debug.class_mapping || 'Class 0 = Cat | Class 1 = Dog';
      document.getElementById('debugPreprocessing').textContent = data.debug.preprocessing || 'RGB → Resize(128,128) → float32 / 255.0';
    }
  }

  // Debug Toggle
  const debugToggle = document.getElementById('debugToggle');
  const debugDetails = document.getElementById('debugDetails');
  if (debugToggle && debugDetails) {
    debugToggle.addEventListener('change', (e) => {
      debugDetails.style.display = e.target.checked ? 'block' : 'none';
    });
  }

  // Initialize
  loadSamples();
});
