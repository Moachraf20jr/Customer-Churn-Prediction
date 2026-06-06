// churn stuff
let metricsChart = null;
let impChart = null;

const plots = [
  { file: 'eda_overview.png', label: 'EDA Overview' },
  { file: 'correlation_heatmap.png', label: 'Correlation Heatmap' },
  { file: 'churn_correlation_bar.png', label: 'Churn Correlation' },
  { file: 'model_comparison.png', label: 'Model Comparison' },
  { file: 'roc_curves_comparison.png', label: 'ROC Curves' },
  { file: 'confusion_matrix_Random_Forest_tuned.png', label: 'Confusion Matrix' },
  { file: 'Contract Length_vs_churn.png', label: 'Churn by Contract Length' },
  { file: 'Subscription Type_vs_churn.png', label: 'Churn by Subscription' },
  { file: 'Gender_vs_churn.png', label: 'Churn by Gender' },
  { file: 'feature_importance_Random_Forest.png', label: 'Feature Importance' },
];

const presets = {
  'high-risk': { Age: 28, Gender: 'Male', Tenure: 3, 'Usage Frequency': 2, 'Support Calls': 9, 'Payment Delay': 28, 'Subscription Type': 'Basic', 'Contract Length': 'Monthly', 'Total Spend': 120, 'Last Interaction': 28 },
  'low-risk': { Age: 45, Gender: 'Female', Tenure: 48, 'Usage Frequency': 25, 'Support Calls': 0, 'Payment Delay': 1, 'Subscription Type': 'Premium', 'Contract Length': 'Annual', 'Total Spend': 900, 'Last Interaction': 3 },
  'medium-risk': { Age: 35, Gender: 'Male', Tenure: 12, 'Usage Frequency': 10, 'Support Calls': 4, 'Payment Delay': 15, 'Subscription Type': 'Standard', 'Contract Length': 'Quarterly', 'Total Spend': 400, 'Last Interaction': 14 },
};

$(function() {
  // theme - was annoying to setup
  if (localStorage.getItem('theme') == 'light') {
    $('html').attr('data-theme', 'light')
    $('#theme-toggle').text('☀️')
  }

  $('#theme-toggle').click(function() {
    if ($('html').attr('data-theme') == 'light') {
      $('html').removeAttr('data-theme')
      $(this).text('🌙')
      localStorage.setItem('theme', 'dark')
    } else {
      $('html').attr('data-theme', 'light')
      $(this).text('☀️')
      localStorage.setItem('theme', 'light')
    }
  })

  // tab switching
  $('.tab').click(function() {
    $('.tab, .tab-content').removeClass('active')
    $(this).addClass('active')
    $('#tab-' + $(this).data('tab')).addClass('active')
  })

  loadModelInfo()
  loadPlots()
  loadHistory()
  checkStatus()

  // form
  $('#predict-form').submit(function(e) {
    e.preventDefault()
    doPredict()
  })

  $('#reset-form-btn').click(function() {
    $('#prediction-result').hide()
    $('#predict-form')[0].reset()
    $('#gauge-fill').css('transform', 'rotate(-180deg)')
    $('#gauge-value').text('0%').css('color', '')
    $('input, select').removeClass('success error')
  })

  $('#clear-history-btn').click(function() {
    if (!localStorage.getItem('churn_history')) return alert('Nothing to clear')
    if (confirm('Clear history?')) {
      localStorage.removeItem('churn_history')
      loadHistory()
    }
  })

  $('.quick-btn').click(function() {
    var d = presets[$(this).data('preset')]
    if (!d) return
    for (var k in d) {
      var el = $('[name="' + k + '"]')
      if (el.length) el.val(d[k])
    }
  })

  $('#refresh-metrics').click(function() { loadModelInfo() })

  $('#copy-result-btn').click(function() {
    var txt = $('#result-content').text().trim().replace(/\s+/g, ' ')
    if (txt) { navigator.clipboard.writeText(txt); alert('Copied!') }
  })

  $('#export-result-btn').click(function() {
    var txt = $('#result-content').text().trim()
    if (!txt) return alert('Nothing to export')
    var b = new Blob([txt], { type: 'text/plain' })
    var a = document.createElement('a')
    a.href = URL.createObjectURL(b)
    a.download = 'prediction.txt'
    a.click()
  })

  setInterval(checkStatus, 10000)
})

// TODO: replace this with something better
function showToast(msg, type) {
  var colors = { success: '#22c55e', error: '#ef4444', info: '#38bdf8' }
  var el = $('<div>').css({
    position: 'fixed', bottom: '20px', right: '20px', zIndex: 9999,
    background: colors[type] || '#38bdf8', color: '#0b1121',
    padding: '10px 18px', borderRadius: '8px', fontSize: '13px',
    boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
  }).text(msg)
  $('body').append(el)
  setTimeout(function() { el.fadeOut(300, function() { $(this).remove() }) }, 3500)
}

function checkStatus() {
  $.get('/api/health').done(function() {
    $('#status-dot').removeClass().addClass('dot online')
    $('#status-text').text('Online')
  }).fail(function() {
    $('#status-dot').removeClass().addClass('dot offline')
    $('#status-text').text('Offline')
  })
}

function loadModelInfo() {
  $('#loading-overlay').show()
  $.get('/api/model-info').done(function(data) {
    console.log('model loaded:', data.model_name)
    $('#model-badge').html('<strong>' + data.model_name + '</strong> · ' +
      (data.metrics.accuracy * 100).toFixed(1) + '% accuracy')
    renderMetrics(data.metrics)
    renderImportance(data.feature_importance)
    renderFeatureTable(data.feature_importance)
  }).fail(function() {
    showToast('Failed to load model info', 'error')
  }).always(function() {
    $('#loading-overlay').hide()
  })
}

function renderMetrics(m) {
  var items = [
    { label: 'Accuracy', value: m.accuracy },
    { label: 'Precision', value: m.precision, cls: 'green' },
    { label: 'Recall', value: m.recall },
    { label: 'F1 Score', value: m.f1_score, cls: 'yellow' },
    { label: 'ROC-AUC', value: m.roc_auc, cls: 'purple' },
  ]
  var html = ''
  items.forEach(function(it) {
    html += '<div class="metric-card"><div class="label">' + it.label + '</div>' +
            '<div class="value ' + (it.cls || '') + '">' + (it.value * 100).toFixed(1) + '%</div></div>'
  })
  $('#metrics-cards').html(html)

  if (metricsChart) metricsChart.destroy()
  var ctx = document.getElementById('metricsChart').getContext('2d')
  metricsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
      datasets: [{
        data: [m.accuracy, m.precision, m.recall, m.f1_score, m.roc_auc],
        backgroundColor: ['#38bdf8', '#34d399', '#fbbf24', '#facc15', '#a78bfa'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 1 } }
    }
  })
}

function renderImportance(data) {
  if (!data || !data.length) return
  data.sort(function(a, b) { return parseFloat(a.Importance) - parseFloat(b.Importance) })
  if (impChart) impChart.destroy()
  var ctx = document.getElementById('importanceChart').getContext('2d')
  impChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(function(d) { return d.Feature }),
      datasets: [{
        data: data.map(function(d) { return parseFloat(d.Importance) }),
        backgroundColor: '#38bdf8',
        borderWidth: 0
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } }
    }
  })
}

function renderFeatureTable(data) {
  if (!data || !data.length) return
  data.sort(function(a, b) { return parseFloat(b.Importance) - parseFloat(a.Importance) })
  var max = parseFloat(data[0].Importance)
  var html = ''
  data.forEach(function(d, i) {
    var pct = (parseFloat(d.Importance) / max * 100).toFixed(0)
    html += '<tr><td>' + d.Feature + (i === 0 ? ' ⭐' : '') + '</td>' +
            '<td><div class="importance-bar"><div class="bar-fill" style="width:' + pct + '%"></div></div></td>' +
            '<td style="text-align:right">' + (parseFloat(d.Importance) * 100).toFixed(2) + '%</td></tr>'
  })
  $('#feature-table tbody').html(html)
  $('#feature-count').text(data.length + ' features')
}

function loadPlots() {
  var html = ''
  plots.forEach(function(p) {
    html += '<div class="plot-card">' +
            '<img src="/results/plots/' + p.file + '" alt="' + p.label + '" loading="lazy" onerror="this.style.display=\'none\'">' +
            '<div class="plot-label">' + p.label + '</div></div>'
  })
  $('#plot-gallery').html(html)

  // lightbox - click image to see bigger
  $('#plot-gallery').on('click', 'img', function() {
    $('#lightbox-img').attr('src', $(this).attr('src'))
    var card = $(this).closest('.plot-card')
    $('#lightbox-label').text(card.find('.plot-label').text() || '')
    $('#lightbox').show()
  })
  $('#lightbox-close, #lightbox').click(function(e) {
    if (e.target === this) $('#lightbox').hide()
  })
}

function doPredict() {
  var data = {}
  $('#predict-form').serializeArray().forEach(function(f) { data[f.name] = f.value })

  var btn = $('button[type=submit]')
  btn.prop('disabled', true).text('Predicting...')
  $('#result-content').html('<div style="padding:30px;text-align:center;color:var(--text-muted)">loading...</div>')
  $('#prediction-result').show()

  $.ajax({
    url: '/api/predict',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify([data])
  }).done(function(res) {
    if (res.error) {
      $('#result-content').html('<div style="color:var(--danger);padding:16px">' + res.error + '</div>')
      return
    }
    var p = res.predictions[0]
    if (!p) {
      $('#result-content').html('<div style="padding:16px">no result</div>')
      return
    }
    showResult(p, data)
    saveHistory(data, p)
  }).fail(function(xhr) {
    var msg = 'connection error'
    try { msg = JSON.parse(xhr.responseText).error } catch(e) {}
    $('#result-content').html('<div style="color:var(--danger);padding:16px">' + msg + '</div>')
  }).always(function() {
    btn.prop('disabled', false).text('Predict Churn')
  })
}

function showResult(p, data) {
  var prob = p.probability
  var deg = -180 + (prob * 180)
  $('#gauge-fill').css('transform', 'rotate(' + deg + 'deg)')
  $('#gauge-value').text((prob * 100).toFixed(0) + '%')
    .css('color', prob >= 0.7 ? 'var(--danger)' : prob >= 0.3 ? 'var(--warning)' : 'var(--success)')

  var isChurn = p.prediction === 'Churn'
  var cls = isChurn ? 'churn' : 'no-churn'

  $('#result-content').html(
    '<div class="result-status ' + cls + '">' + (isChurn ? 'Will Churn' : 'Will Stay') + '</div>' +
    '<div class="result-details">' +
      '<div class="result-detail-item"><div class="detail-label">Probability</div>' +
      '<div class="detail-value">' + (prob * 100).toFixed(2) + '%</div></div>' +
      '<div class="result-detail-item"><div class="detail-label">Risk</div>' +
      '<div class="detail-value">' + p.risk_level + '</div></div>' +
    '</div>'
  )
}

function saveHistory(input, pred) {
  var hist = JSON.parse(localStorage.getItem('churn_history') || '[]')
  hist.unshift({
    timestamp: new Date().toLocaleString(),
    prediction: pred.prediction,
    probability: pred.probability,
    risk_level: pred.risk_level,
    data: input
  })
  if (hist.length > 20) hist = hist.slice(0, 20)
  localStorage.setItem('churn_history', JSON.stringify(hist))
  loadHistory()
}

function loadHistory() {
  var hist = JSON.parse(localStorage.getItem('churn_history') || '[]')
  $('#history-count').text(hist.length)
  if (hist.length === 0) {
    $('#history-list').html('<div class="history-empty">No predictions yet.</div>')
    return
  }
  var html = ''
  hist.forEach(function(h, i) {
    var cls = h.prediction === 'Churn' ? 'churn' : 'no-churn'
    html += '<div class="history-item" data-idx="' + i + '">' +
            '<div><div class="h-result ' + cls + '">' + h.prediction + '</div>' +
            '<div class="h-time">' + (h.probability * 100).toFixed(1) + '% · ' + h.timestamp + '</div></div>' +
            '<span class="h-risk risk-' + h.risk_level.toLowerCase() + '">' + h.risk_level + '</span></div>'
  })
  $('#history-list').html(html)
  $('.history-item').click(function() {
    var idx = $(this).data('idx')
    var entry = JSON.parse(localStorage.getItem('churn_history') || '[]')[idx]
    if (entry && entry.data) {
      for (var k in entry.data) {
        var el = $('[name="' + k + '"]')
        if (el.length) el.val(entry.data[k])
      }
    }
  })
}
